"""POST /oracle_router/feedback and GET /oracle_router/state."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.oracle_router_endpoints import (
    get_oracle_router_state,
    submit_oracle_router_feedback,
)
from litellm.router_strategy.oracle_router.oracle_router import OracleRouter
from litellm.router_strategy.oracle_router.verifier import ReportedVerifier
from litellm.types.management_endpoints.oracle_router_endpoints import OracleRouterFeedbackRequest
from litellm.types.router import OracleRouterConfig, TaggedPreRoutingStrategy

ADMIN = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="admin-hash")
OWNER = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="owner-hash")
STRANGER = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="other-hash")


class _Maker:
    def __init__(self) -> None:
        self.updates = []

    @property
    def models(self):
        return ("smart", "fast")

    async def select(self, context):
        return "smart"

    def update(self, context, model, score):
        self.updates.append((model, score))

    def snapshot(self):
        return {"type": "test"}


class _LLMRouter:
    def __init__(self, oracle: OracleRouter) -> None:
        self.oracle_routers = {oracle.router_name: [TaggedPreRoutingStrategy(tags=(), strategy=oracle)]}


@pytest.fixture
def oracle():
    maker = _Maker()
    router = OracleRouter("oracle", OracleRouterConfig(available_models=["smart", "fast"]), maker, ReportedVerifier())
    router.maker = maker
    return router


async def _bind(oracle: OracleRouter, program_id: str, api_key_hash: str | None = "owner-hash"):
    metadata = {"program_id": program_id, **({"user_api_key_hash": api_key_hash} if api_key_hash else {})}
    await oracle.async_pre_routing_hook(
        model="oracle", request_kwargs={"metadata": metadata}, messages=[{"role": "user", "content": "t"}]
    )


@pytest.mark.asyncio
async def test_feedback_completes_the_program_and_the_router_learns(oracle):
    await _bind(oracle, "task-1")
    oracle.observe_request("task-1", cost=0.2, response_text="done")
    with patch("litellm.proxy.proxy_server.llm_router", _LLMRouter(oracle)):
        response = await submit_oracle_router_feedback(
            OracleRouterFeedbackRequest(program_id="task-1", score=1.0), OWNER
        )
    assert response.model_dump() == {
        "program_id": "task-1",
        "router_name": "oracle",
        "model": "smart",
        "pending_verifications": 1,
    }
    assert oracle.binding("task-1") is None
    await oracle.drain()
    assert oracle.maker.updates == [("smart", 1.0)]


@pytest.mark.asyncio
async def test_feedback_from_another_key_is_forbidden_but_an_admin_may_complete(oracle):
    await _bind(oracle, "task-1")
    with patch("litellm.proxy.proxy_server.llm_router", _LLMRouter(oracle)):
        with pytest.raises(HTTPException) as denied:
            await submit_oracle_router_feedback(OracleRouterFeedbackRequest(program_id="task-1", score=1.0), STRANGER)
        assert denied.value.status_code == 403 and oracle.binding("task-1") is not None
        response = await submit_oracle_router_feedback(
            OracleRouterFeedbackRequest(program_id="task-1", score=0.0), ADMIN
        )
    assert response.program_id == "task-1" and oracle.binding("task-1") is None
    await oracle.drain()


@pytest.mark.asyncio
async def test_feedback_for_an_unknown_program_or_without_routers_is_404(oracle):
    with patch("litellm.proxy.proxy_server.llm_router", _LLMRouter(oracle)):
        with pytest.raises(HTTPException) as unknown:
            await submit_oracle_router_feedback(OracleRouterFeedbackRequest(program_id="nope"), OWNER)
    assert unknown.value.status_code == 404
    with patch("litellm.proxy.proxy_server.llm_router", None):
        with pytest.raises(HTTPException) as none:
            await submit_oracle_router_feedback(OracleRouterFeedbackRequest(program_id="nope"), OWNER)
    assert none.value.status_code == 404


@pytest.mark.asyncio
async def test_state_is_admin_only_and_lists_every_router(oracle):
    await _bind(oracle, "task-1")
    with patch("litellm.proxy.proxy_server.llm_router", _LLMRouter(oracle)):
        with pytest.raises(HTTPException) as denied:
            await get_oracle_router_state(OWNER)
        assert denied.value.status_code == 403
        state = await get_oracle_router_state(ADMIN)
    assert len(state.routers) == 1
    assert state.routers[0]["router_name"] == "oracle" and state.routers[0]["active_programs"] == 1
    assert state.routers[0]["decision_maker"] == {"type": "test"}


def test_feedback_request_validates_score_range():
    with pytest.raises(ValidationError, match="score"):
        OracleRouterFeedbackRequest(program_id="p", score=1.5)
    with pytest.raises(ValidationError, match="program_id"):
        OracleRouterFeedbackRequest(program_id="")
