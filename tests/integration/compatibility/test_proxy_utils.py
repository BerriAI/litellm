from __future__ import annotations

from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue

_TOOL_PARAMETERS: Final = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_FUNCTION_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "integration_lookup",
        "description": "Look up a synthetic value",
        "parameters": _TOOL_PARAMETERS,
    },
}
_FAKE_API_KEY: Final = "my-fake-api-key"
_OPENAI_API_BASE: Final = "https://api.openai.com/v1/"
_ANTHROPIC_API_BASE: Final = "https://api.anthropic.com/v1/messages"


class _RawRequestResponse(BaseModel):
    raw_request_api_base: str | None = None
    raw_request_body: dict[str, JsonValue] | None = None
    raw_request_headers: dict[str, str] | None = None
    error: str | None = None


def _transform_request(
    gateway: Gateway,
    key: str,
    model: str,
    stream: bool,
) -> tuple[httpx.Response, _RawRequestResponse]:
    request_body: Final = {
        "model": model,
        "messages": [{"role": "user", "content": "integration transform request"}],
        "tools": [_FUNCTION_TOOL],
        "tool_choice": "auto",
        "max_tokens": 17,
        "stream": stream,
    }
    response: Final = gateway.request(
        "POST",
        "/utils/transform_request",
        {"call_type": "completion", "request_body": request_body},
        key=key,
    )
    assert response.status_code == 200, response.text
    parsed: Final = _RawRequestResponse.model_validate_json(response.content)
    assert parsed.error is None, response.text
    return response, parsed


def test_transform_request_returns_exact_provider_request_and_rejects_unsafe_bodies(gateway: Gateway) -> None:
    openai_body: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "integration transform request"}],
        "max_tokens": 17,
        "tools": [_FUNCTION_TOOL],
        "tool_choice": "auto",
        "extra_body": {},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    openai_non_streaming_body: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "integration transform request"}],
        "max_tokens": 17,
        "tools": [_FUNCTION_TOOL],
        "tool_choice": "auto",
        "extra_body": {},
    }
    anthropic_body: Final = {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "integration transform request"}]}],
        "max_tokens": 17,
        "tools": [
            {
                "type": "custom",
                "name": "integration_lookup",
                "description": "Look up a synthetic value",
                "input_schema": _TOOL_PARAMETERS,
            }
        ],
        "tool_choice": {"type": "auto"},
    }

    def no_destination_traffic(request: Request) -> Reply:
        return Reply()

    with gateway.scenario() as scenario, wire_server(no_destination_traffic) as destination:
        key: Final = scenario.key()
        openai_stream_result: Final = _transform_request(gateway, key, "openai/gpt-4o-mini", True)
        openai_stream_response, openai_stream = openai_stream_result
        assert openai_stream.raw_request_api_base == _OPENAI_API_BASE, openai_stream_response.text
        assert openai_stream.raw_request_body == openai_body, openai_stream_response.text
        assert openai_stream.raw_request_headers == {"Authorization": "Be****ey"}, openai_stream_response.text
        assert _FAKE_API_KEY not in openai_stream_response.text, "Transform response exposed its synthetic credential"

        openai_non_stream_result: Final = _transform_request(gateway, key, "openai/gpt-4o-mini", False)
        openai_non_stream_response, openai_non_stream = openai_non_stream_result
        assert openai_non_stream.raw_request_api_base == _OPENAI_API_BASE, openai_non_stream_response.text
        assert openai_non_stream.raw_request_body == openai_non_streaming_body, openai_non_stream_response.text
        assert _FAKE_API_KEY not in openai_non_stream_response.text, (
            "Transform response exposed its synthetic credential"
        )

        anthropic_result: Final = _transform_request(gateway, key, "anthropic/claude-opus-4-8", False)
        anthropic_response, anthropic = anthropic_result
        assert anthropic.raw_request_api_base == _ANTHROPIC_API_BASE, anthropic_response.text
        assert anthropic.raw_request_body == anthropic_body, anthropic_response.text
        assert anthropic.raw_request_headers == {
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": "my****ey",
        }, anthropic_response.text
        assert _FAKE_API_KEY not in anthropic_response.text, "Transform response exposed its synthetic credential"

        unsafe_requests: Final = (
            (
                "api_base",
                destination.url,
                "Rejected Request: api_base is not allowed in request body. Clientside passthrough requires "
                "explicit admin opt-in via either `general_settings.allow_client_side_credentials = true` "
                "(proxy-wide) or `configurable_clientside_auth_params` on the deployment in your proxy "
                "config.yaml. Relevant Issue: "
                "https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997",
            ),
            (
                "base_url",
                destination.url,
                "Rejected Request: base_url is not allowed in request body. Clientside passthrough requires "
                "explicit admin opt-in via either `general_settings.allow_client_side_credentials = true` "
                "(proxy-wide) or `configurable_clientside_auth_params` on the deployment in your proxy "
                "config.yaml. Relevant Issue: "
                "https://huntr.com/bounties/4001e1a2-7b7a-4776-a3ae-e6692ec3d997",
            ),
            (
                "model_list",
                [{"model_name": "unsafe", "litellm_params": {"api_base": destination.url}}],
                "Rejected Request: model_list is not allowed in the request body.",
            ),
        )
        for field, value, error in unsafe_requests:
            rejected: Final = gateway.request(
                "POST",
                "/utils/transform_request",
                {
                    "call_type": "completion",
                    "request_body": {
                        "model": "openai/gpt-4o-mini",
                        "messages": [{"role": "user", "content": "unsafe transform request"}],
                        field: value,
                    },
                },
                key=key,
            )
            assert rejected.status_code == 400, rejected.text
            assert rejected.json() == {"detail": {"error": error}}, rejected.text
            assert _FAKE_API_KEY not in rejected.text, "Unsafe response exposed its synthetic credential"
            assert destination.drain() == ()
        assert destination.drain() == ()


def test_transform_request_reports_masked_authorization_for_non_streaming_openai(gateway: Gateway) -> None:
    pytest.skip("BUG: /utils/transform_request returns empty raw_request_headers for a non-streaming openai completion")
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response, parsed = _transform_request(gateway, key, "openai/gpt-4o-mini", False)
        assert parsed.raw_request_api_base == _OPENAI_API_BASE, response.text
        assert parsed.raw_request_body == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "integration transform request"}],
            "max_tokens": 17,
            "tools": [_FUNCTION_TOOL],
            "tool_choice": "auto",
            "extra_body": {},
        }, response.text
        assert parsed.raw_request_headers == {"Authorization": "Be****ey"}, response.text
        assert _FAKE_API_KEY not in response.text, "Transform response exposed its synthetic credential"


def test_transform_request_reports_stream_flag_for_streaming_anthropic(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: /utils/transform_request omits stream: true from the anthropic raw_request_body when stream is true"
    )
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        response, parsed = _transform_request(gateway, key, "anthropic/claude-opus-4-8", True)
        assert parsed.raw_request_api_base == _ANTHROPIC_API_BASE, response.text
        assert parsed.raw_request_headers == {
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": "my****ey",
        }, response.text
        assert parsed.raw_request_body == {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "integration transform request"}]}],
            "max_tokens": 17,
            "tools": [
                {
                    "type": "custom",
                    "name": "integration_lookup",
                    "description": "Look up a synthetic value",
                    "input_schema": _TOOL_PARAMETERS,
                }
            ],
            "tool_choice": {"type": "auto"},
            "stream": True,
        }, response.text
        assert _FAKE_API_KEY not in response.text, "Transform response exposed its synthetic credential"
