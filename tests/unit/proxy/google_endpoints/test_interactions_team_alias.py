"""Regression tests for team model_aliases on POST /v1beta/interactions."""

import json
from typing import Any, Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response
from starlette.requests import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    resolve_inference_model,
)
from litellm.proxy.google_endpoints.endpoints import create_interaction


def _make_request(payload: dict[str, Any]) -> Request:
    body: Final = json.dumps(payload).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope: Final = {
        "type": "http",
        "method": "POST",
        "path": "/v1beta/interactions",
        "headers": [(b"content-type", b"application/json")],
    }
    return Request(scope, receive)


@pytest.mark.asyncio
async def test_create_interaction_passes_model_none_to_preserve_body_aliases():
    """create_interaction must pass model=None so body model aliases are not reverted."""
    request: Final = _make_request({"model": "video", "input": "hello"})
    fastapi_response: Final = Response()
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key="sk-test",
        models=["video"],
        team_model_aliases={"video": "gemini-omni"},
    )

    with patch.object(
        ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        new_callable=AsyncMock,
    ) as mock_process:
        mock_process.return_value = {"id": "int-1"}
        result: Final = await create_interaction(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )

    assert result == {"id": "int-1"}
    mock_process.assert_awaited_once()
    assert mock_process.call_args.kwargs["model"] is None
    assert mock_process.call_args.kwargs["route_type"] == "acreate_interaction"


def test_resolve_inference_model_preserves_aliased_body_model_when_endpoint_model_none():
    """When endpoint_model is None, resolve_inference_model keeps the aliased body model."""
    resolved: Final = resolve_inference_model(
        "gemini-omni",
        {},
        None,
        None,
        kind="completion",
    )
    assert resolved == "gemini-omni"
