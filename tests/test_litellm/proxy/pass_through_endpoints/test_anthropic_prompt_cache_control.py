import asyncio
import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Response
from starlette.requests import Request

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    llm_passthrough_factory_proxy_route,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    ANTHROPIC_PROMPT_CACHE_TTL_ENV,
    ANTHROPIC_PROMPT_CACHE_TTL_HEADER,
    _apply_anthropic_prompt_cache_control_to_request,
    apply_anthropic_prompt_cache_control,
    filter_anthropic_prompt_cache_control_headers,
)
from litellm.types.utils import LlmProviders


class DummyState:
    pass


class DummyRequest:
    def __init__(self, body, headers=None, method="POST"):
        self.method = method
        self.headers = headers or {}
        self.scope = {}
        self.state = DummyState()
        self._body = body

    async def body(self):
        import orjson

        return orjson.dumps(self._body)


def test_apply_anthropic_prompt_cache_control_injects_top_level_one_hour(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is True
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_apply_anthropic_prompt_cache_control_preserves_client_control(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
        "messages": [{"role": "user", "content": "hello"}],
    }
    original = copy.deepcopy(body)

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is False
    assert body == original


def test_apply_anthropic_prompt_cache_control_preserves_nested_client_control(
    monkeypatch,
):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hello",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    original = copy.deepcopy(body)

    changed = apply_anthropic_prompt_cache_control(body, headers={})

    assert changed is False
    assert body == original


def test_apply_anthropic_prompt_cache_control_header_can_disable_env_default(
    monkeypatch,
):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    body = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }

    changed = apply_anthropic_prompt_cache_control(
        body,
        headers={ANTHROPIC_PROMPT_CACHE_TTL_HEADER: "off"},
    )

    assert changed is False
    assert "cache_control" not in body


def test_filter_anthropic_prompt_cache_control_headers_strips_control_headers():
    headers = {
        "authorization": "Bearer sk-test",
        ANTHROPIC_PROMPT_CACHE_TTL_HEADER: "1h",
        "x-anthropic-prompt-cache-workload": "eval",
    }

    filtered = filter_anthropic_prompt_cache_control_headers(headers)

    assert filtered == {"authorization": "Bearer sk-test"}


def test_anthropic_passthrough_mutates_messages_request(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    request = DummyRequest(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"content-type": "application/json"},
    )

    asyncio.run(
        _apply_anthropic_prompt_cache_control_to_request(
            request,
            LlmProviders.ANTHROPIC.value,
            "/v1/messages",
        )
    )

    assert request.state.litellm_pass_through_custom_body["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


def test_non_anthropic_passthrough_is_unchanged(monkeypatch):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "1h")
    request = DummyRequest(
        {
            "model": "some-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"content-type": "application/json"},
    )

    asyncio.run(
        _apply_anthropic_prompt_cache_control_to_request(
            request,
            LlmProviders.MISTRAL.value,
            "/v1/messages",
        )
    )

    assert not hasattr(request.state, "litellm_pass_through_custom_body")


def _build_starlette_request(body, headers=None):
    request_body = json.dumps(body).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": request_body, "more_body": False}

    raw_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/anthropic/v1/messages",
        "raw_path": b"/anthropic/v1/messages",
        "query_string": b"",
        "headers": raw_headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    return Request(scope, receive)


@pytest.mark.asyncio
async def test_anthropic_passthrough_e2e_forwards_injected_cache_control(
    monkeypatch,
):
    monkeypatch.setenv(ANTHROPIC_PROMPT_CACHE_TTL_ENV, "auto")
    captured = {}

    async def upstream_handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            status_code=200,
            json={
                "id": "msg_cache_e2e",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-20250514",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 11,
                    "cache_creation_input_tokens": 64,
                    "cache_read_input_tokens": 32,
                    "cache_creation": {"ephemeral_1h_input_tokens": 64},
                    "output_tokens": 7,
                },
            },
            request=request,
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
    mock_client_obj = MagicMock()
    mock_client_obj.client = async_client

    async def pre_call_hook(**kwargs):
        data = dict(kwargs["data"])
        data.pop("litellm_logging_obj", None)
        return data

    request = _build_starlette_request(
        {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "stable evaluation prefix"}],
        },
        headers={
            "content-type": "application/json",
            "x-anthropic-prompt-cache-ttl": "auto",
            "x-anthropic-prompt-cache-workload": "benchmark",
        },
    )
    provider_config = MagicMock()
    provider_config.get_api_base.return_value = "https://api.anthropic.com"
    provider_config.validate_environment.return_value = {"x-api-key": "test-key"}

    try:
        with (
            patch(
                "litellm.utils.ProviderConfigManager.get_provider_model_info",
                return_value=provider_config,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router.get_credentials",
                return_value="test-key",
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.is_registered_pass_through_route",
                return_value=True,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.InitPassThroughEndpointHelpers.get_registered_pass_through_route",
                return_value=None,
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
                return_value=mock_client_obj,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj.pre_call_hook",
                new=AsyncMock(side_effect=pre_call_hook),
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_logging_obj.post_call_response_headers_hook",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.pass_through_endpoint_logging.pass_through_async_success_handler",
                new=AsyncMock(),
            ),
        ):
            response = await llm_passthrough_factory_proxy_route(
                custom_llm_provider=LlmProviders.ANTHROPIC.value,
                endpoint="/v1/messages",
                request=request,
                fastapi_response=Response(),
                user_api_key_dict=MagicMock(),
            )
    finally:
        await async_client.aclose()

    assert json.loads(response.body)["usage"]["cache_read_input_tokens"] == 32
    assert captured["body"]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "x-anthropic-prompt-cache-ttl" not in captured["headers"]
    assert "x-anthropic-prompt-cache-workload" not in captured["headers"]
