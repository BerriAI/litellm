from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.response_api_endpoints.endpoints import (
    _prepare_cross_deployment_responses_replay,
)
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesVisibleHistory,
)


@pytest.mark.asyncio
async def test_cross_deployment_replay_runs_guardrails_on_assembled_history():
    history = ResponsesVisibleHistory(
        input=(
            {"type": "message", "role": "user", "content": "stored user text"},
            {"type": "message", "role": "assistant", "content": "stored assistant text"},
        ),
        litellm_session_id="session-1",
    )
    history_loader = AsyncMock(return_value=history)
    proxy_logging = AsyncMock()
    proxy_logging.pre_call_hook = AsyncMock(side_effect=lambda **kwargs: kwargs["data"])

    prepared = await _prepare_cross_deployment_responses_replay(
        model="luna",
        request_kwargs={
            "previous_response_id": "resp-terra",
            "input": "current user text",
            "litellm_metadata": {"user_api_key_hash": "hashed-key"},
        },
        proxy_logging_obj=proxy_logging,
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        history_loader=history_loader,
    )

    guarded_input = proxy_logging.pre_call_hook.await_args.kwargs["data"]["input"]
    assert guarded_input == [
        {"type": "message", "role": "user", "content": "stored user text"},
        {"type": "message", "role": "assistant", "content": "stored assistant text"},
        {"type": "message", "role": "user", "content": "current user text"},
    ]
    assert prepared["input"] == guarded_input
    assert "previous_response_id" not in prepared


@pytest.mark.asyncio
async def test_cross_deployment_replay_fails_closed_without_history():
    history_loader = AsyncMock(return_value=None)

    with pytest.raises(HTTPException, match="visible history is unavailable"):
        await _prepare_cross_deployment_responses_replay(
            model="luna",
            request_kwargs={"previous_response_id": "resp-terra", "input": "current"},
            proxy_logging_obj=AsyncMock(),
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
            history_loader=history_loader,
        )
