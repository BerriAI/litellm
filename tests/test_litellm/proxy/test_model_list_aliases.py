"""
Tests for key and team `model_aliases` on the model listing endpoints: GET /v1/models
(`model_list`, OpenAI and Anthropic shapes) and GET /v1/models/{id} (`model_info`).
An alias the caller can complete on is listed next to its target and resolves by name.
"""

import pytest
from starlette.requests import Request

from litellm import Router
from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _deployment(model_name: str, model: str = "openai/gpt-4.1-mini"):
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "api_key": "sk-fake"},
        "model_info": {"id": f"{model_name}-id"},
    }


@pytest.fixture
def router(monkeypatch) -> Router:
    router = Router(model_list=[_deployment("gpt-4.1-mini"), _deployment("gpt-4.1", model="openai/gpt-4.1")])
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "llm_model_list", router.model_list)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    return router


def _team_member(team_id: str = "team1", **aliases: dict[str, str] | None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-test",
        user_id="u",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id=team_id,
        team_models=["gpt-4.1-mini"],
        models=["gpt-4.1-mini"],
        **aliases,
    )


def _anthropic_request() -> Request:
    return Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/v1/models",
            "query_string": b"",
            "headers": [(b"anthropic-version", b"2023-06-01")],
        }
    )


async def _v1_models(user_api_key_dict: UserAPIKeyAuth, **kwargs) -> list[str]:
    response = await proxy_server.model_list(user_api_key_dict=user_api_key_dict, **kwargs)
    return [m["id"] for m in response["data"]]


@pytest.mark.asyncio
async def test_v1_models_lists_team_alias_next_to_its_target_in_both_shapes(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "claude-sonnet-4-5"]
    assert await _v1_models(caller, request=_anthropic_request()) == ["gpt-4.1-mini", "claude-sonnet-4-5"]


@pytest.mark.asyncio
async def test_v1_models_lists_key_alias_and_hides_alias_to_a_model_the_caller_cannot_list(router):
    caller = _team_member(aliases={"mini": "gpt-4.1-mini", "big": "gpt-4.1"})

    assert await _v1_models(caller) == ["gpt-4.1-mini", "mini"]


@pytest.mark.asyncio
async def test_v1_models_ignores_a_malformed_alias_map(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": 5})

    assert await _v1_models(caller) == ["gpt-4.1-mini"]


@pytest.mark.asyncio
async def test_v1_models_by_id_resolves_a_team_alias_to_its_target_metadata(router):
    caller = _team_member(team_model_aliases={"claude-sonnet-4-5": "gpt-4.1-mini"})

    response = await proxy_server.model_info(model_id="claude-sonnet-4-5", user_api_key_dict=caller)
    assert response["id"] == "claude-sonnet-4-5"
    assert response["owned_by"] == "openai"
