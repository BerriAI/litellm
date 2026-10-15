from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.rust_bridge.responses.route_host import connection_defaults, decline_reason, map_failure, response
from litellm.types.llms.openai import ResponsesAPIResponse


def test_response_validates_into_the_public_responses_model() -> None:
    built: Final = response(
        MappingProxyType(
            {
                "id": "resp_native",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_native",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "native", "annotations": []}],
                    }
                ],
            }
        )
    )

    assert isinstance(built, ResponsesAPIResponse)
    assert built.id == "resp_native"
    assert built.output[0].content[0].text == "native"


def test_response_rejects_a_payload_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        response(MappingProxyType({"object": "response"}))


@pytest.mark.parametrize(
    ("global_key", "provider_key", "expected"),
    (
        ("global", "provider", "global"),
        (None, "provider", "provider"),
        ("", "provider", "provider"),
        (None, None, None),
    ),
)
def test_connection_defaults_preserve_openai_precedence(
    monkeypatch: pytest.MonkeyPatch, global_key: str | None, provider_key: str | None, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "api_key", global_key)
    monkeypatch.setattr(litellm, "openai_key", provider_key)
    monkeypatch.setattr(litellm, "api_base", "https://configured.invalid/v1")
    assert connection_defaults("openai") == (expected, litellm.api_base)


@pytest.mark.parametrize(("global_key", "expected"), (("global", "global"), (None, None)))
def test_connection_defaults_keep_the_openai_key_away_from_openrouter(
    monkeypatch: pytest.MonkeyPatch, global_key: str | None, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "api_key", global_key)
    monkeypatch.setattr(litellm, "openai_key", "openai-only")
    monkeypatch.setattr(litellm, "api_base", None)
    assert connection_defaults("openrouter") == (expected, None)


class _UpstreamFailure(Exception):
    headers: Final = ()


@pytest.mark.parametrize(
    ("api_base", "base_url", "expected"),
    (
        (None, "https://alias.invalid/v1", "https://alias.invalid/v1"),
        ("", "https://alias.invalid/v1", "https://alias.invalid/v1"),
        ("https://base.invalid/v1", "https://alias.invalid/v1", "https://base.invalid/v1"),
    ),
)
def test_failure_preserves_the_explicit_endpoint(api_base: str | None, base_url: str, expected: str) -> None:
    upstream: Final = _UpstreamFailure(429, '{"error":{"message":"rate limited"}}')
    mapped: Final = map_failure(upstream, {"model": "openai/test-model", "api_base": api_base, "base_url": base_url})
    assert isinstance(mapped, litellm.RateLimitError)
    assert str(mapped.response.request.url) == expected
    assert mapped.__context__ is upstream


_PATTERNED_TOOL: Final = MappingProxyType(
    {
        "type": "function",
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "pattern": "^\\p{L}+$"}}},
    }
)
_PLAIN_TOOL: Final = MappingProxyType(
    {"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}
)


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    (
        pytest.param({"input": "hi", "temperature": 1}, None, id="supported_temperature"),
        pytest.param(
            {"input": "hi", "temperature": 0.2},
            "native Responses does not implement the model's sampling rules",
            id="gpt_5_sampling_rules",
        ),
        pytest.param({"input": "hi", "tools": [dict(_PLAIN_TOOL)]}, None, id="valid_tool_schema"),
        pytest.param(
            {"input": "hi", "tools": [dict(_PATTERNED_TOOL)]},
            "native Responses does not implement tool schema repair",
            id="python_incompatible_pattern",
        ),
        pytest.param(
            {"input": [{"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}]},
            "native Responses does not implement prompt cache breakpoints",
            id="prompt_cache_breakpoints_on_openai",
        ),
        pytest.param(
            {
                "input": [{"role": "user", "content": "hi", "cache_control": {"type": "ephemeral"}}],
                "api_base": "http://127.0.0.1:9/v1",
            },
            None,
            id="cache_control_on_a_compatible_host",
        ),
    ),
)
def test_model_rules_stay_on_python(request_fields: dict[str, object], expected: str | None) -> None:
    assert decline_reason(MappingProxyType({"model": "openai/gpt-6.1-sol", **request_fields})) == expected
