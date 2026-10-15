from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.rust_bridge.responses.route_host import (
    connection_defaults,
    decline_reason,
    map_failure,
    provider_connection,
    response,
)
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
            "native Responses does not implement tool schema validation and repair",
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


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    (
        pytest.param({"api_key": "azure-key"}, None, id="api_key"),
        pytest.param(
            {"api_key": None, "extra_headers": {"api-key": "caller-key"}}, None, id="forwarded_api_key_header"
        ),
        pytest.param(
            {"api_key": None}, "native Azure Responses does not implement Azure AD token auth", id="azure_ad_token_auth"
        ),
        pytest.param(
            {"api_key": "azure-key", "model": "azure/o_series/my-o3"},
            "native Azure Responses does not implement the o-series config",
            id="o_series",
        ),
        pytest.param(
            {"api_key": "azure-key", "context_management": [{"type": "compaction"}]},
            "native Azure Responses does not implement context_management",
            id="context_management",
        ),
        pytest.param({"api_key": "azure-key", "api_version": "2025-03-01-preview"}, None, id="api_version"),
    ),
)
def test_azure_requests_outside_the_native_config_stay_on_python(
    monkeypatch: pytest.MonkeyPatch, request_fields: dict[str, object], expected: str | None
) -> None:
    for name in ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "azure_key", None)
    request: Final = {"model": "azure/my-deployment", "input": "hi", **request_fields}
    assert decline_reason(MappingProxyType(request)) == expected


def test_azure_connection_defaults_read_the_azure_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "azure_key", "azure-global")
    monkeypatch.setattr(litellm, "api_base", None)
    assert connection_defaults("azure") == ("azure-global", None)


@pytest.mark.parametrize(
    ("request_fields", "xai_key", "expected"),
    [
        pytest.param({"api_key": "xai-explicit"}, None, None, id="explicit_key"),
        pytest.param({}, "xai-global", None, id="xai_key"),
        pytest.param({}, None, "native xAI Responses does not implement OAuth", id="no_key"),
    ],
)
def test_xai_without_a_key_stays_on_python(
    monkeypatch: pytest.MonkeyPatch, request_fields: dict[str, object], xai_key: str | None, expected: str | None
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "xai_key", xai_key)
    assert decline_reason({"model": "xai/grok-4", "input": "hi", **request_fields}) == expected


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        pytest.param({"api_base": "http://vllm:8000"}, None, id="api_base"),
        pytest.param({"base_url": "http://vllm:8000"}, None, id="base_url"),
        pytest.param({}, "native hosted vLLM Responses needs an api_base", id="no_api_base"),
    ],
)
def test_hosted_vllm_without_an_api_base_stays_on_python(
    monkeypatch: pytest.MonkeyPatch, request_fields: dict[str, object], expected: str | None
) -> None:
    monkeypatch.delenv("HOSTED_VLLM_API_BASE", raising=False)
    assert decline_reason({"model": "hosted_vllm/llama", "input": "hi", **request_fields}) == expected


def test_provider_connection_matches_get_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_API_BASE", raising=False)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "xai_key", "xai-global")
    assert provider_connection("xai/grok-4", None, None, None) == ("xai-global", "https://api.x.ai/v1")
    assert provider_connection("hosted_vllm/llama", None, None, "http://vllm:8000") == (
        "fake-api-key",
        "http://vllm:8000",
    )
    assert provider_connection("gpt-4.1", None, None, None) == (None, None)


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        pytest.param({"api_key": "fw-key"}, None, id="api_key"),
        pytest.param({}, "native Fireworks Responses needs an api key", id="no_key"),
        pytest.param(
            {"api_key": "fw-key", "litellm_metadata": {"session_id": "sess-1"}},
            "native Fireworks Responses does not implement session affinity",
            id="session_affinity",
        ),
        pytest.param(
            {"api_key": "fw-key", "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]},
            "native Responses does not implement emulated file_search",
            id="emulated_file_search",
        ),
    ],
)
def test_fireworks_requests_outside_the_native_config_stay_on_python(
    monkeypatch: pytest.MonkeyPatch, request_fields: dict[str, object], expected: str | None
) -> None:
    for name in ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY", "FIREWORKSAI_API_KEY", "FIREWORKS_AI_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert decline_reason({"model": "fireworks_ai/qwen3", "input": "hi", **request_fields}) == expected


def test_openai_file_search_stays_native() -> None:
    request: Final = {
        "model": "gpt-4.1",
        "input": "hi",
        "api_key": "sk-test",
        "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}],
    }
    assert decline_reason(request) is None
