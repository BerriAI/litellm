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
    provider_params,
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


@pytest.mark.parametrize(
    ("request_fields", "environment", "expected"),
    [
        pytest.param({}, {}, None, id="personal_access_token"),
        pytest.param(
            {"model": "databricks/databricks-claude-sonnet"},
            {},
            "Databricks serves Responses only on GPT endpoints",
            id="non_gpt_endpoint",
        ),
        pytest.param(
            {"api_key": None},
            {},
            "native Databricks Responses does not implement Databricks SDK auth",
            id="no_token",
        ),
        pytest.param(
            {"api_key": None, "api_base": None},
            {"DATABRICKS_API_KEY": "dapi-env"},
            "native Databricks Responses does not implement Databricks SDK auth",
            id="no_api_base",
        ),
        pytest.param(
            {},
            {"DATABRICKS_CLIENT_ID": "id", "DATABRICKS_CLIENT_SECRET": "secret"},
            "native Databricks Responses does not implement OAuth M2M",
            id="oauth_m2m",
        ),
    ],
)
def test_databricks_requests_outside_the_native_config_stay_on_python(
    monkeypatch: pytest.MonkeyPatch,
    request_fields: dict[str, object],
    environment: dict[str, str],
    expected: str | None,
) -> None:
    for name in ("DATABRICKS_API_KEY", "DATABRICKS_API_BASE", "DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    request: Final = {
        "model": "databricks/databricks-gpt-5-nano",
        "input": "hi",
        "api_key": "dapi-explicit",
        "api_base": "https://w.cloud.databricks.com/serving-endpoints",
        **request_fields,
    }
    assert decline_reason(request) == expected


def test_databricks_connection_defaults_ignore_the_global_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", "global")
    monkeypatch.setattr(litellm, "api_base", "https://global.invalid")
    assert connection_defaults("databricks") == (None, None)


_MANTLE_MODEL_COST: Final = {
    "bedrock_mantle/openai.gpt-openai-path": {"mode": "responses", "use_openai_responses_path": True},
    "bedrock_mantle/openai.gpt-v1-path": {"mode": "chat", "supported_endpoints": ["/v1/responses"]},
    "bedrock_mantle/openai.gpt-chat-only": {"mode": "chat"},
}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        pytest.param("bedrock_mantle/openai.gpt-openai-path", (("use_openai_responses_path", True),), id="openai"),
        pytest.param(
            "bedrock_mantle/us-west-2/openai.gpt-v1-path", (("use_openai_responses_path", False),), id="regional_v1"
        ),
        pytest.param("openai/gpt-openai-path", (), id="other_provider"),
    ],
)
def test_provider_params_carry_the_mantle_path_from_the_cost_map(
    monkeypatch: pytest.MonkeyPatch, model: str, expected: tuple[tuple[str, object], ...]
) -> None:
    monkeypatch.setattr(litellm, "model_cost", _MANTLE_MODEL_COST)
    assert provider_params(model, None) == expected


@pytest.mark.parametrize(
    ("request_fields", "expected"),
    [
        pytest.param({}, None, id="responses_model"),
        pytest.param({"model": "bedrock_mantle/openai.gpt-v1-path"}, None, id="supported_endpoint"),
        pytest.param(
            {"aws_region_name": "us-west-2", "aws_role_name": "arn:aws:iam::1:role/r", "aws_bedrock_project_id": "p"},
            None,
            id="aws_settings",
        ),
        pytest.param(
            {"aws_session_tags": {"team": "a"}},
            "native inference does not implement aws_session_tags",
            id="aws_session_tags",
        ),
        pytest.param(
            {"model": "bedrock_mantle/openai.gpt-chat-only"}, "native HTTP responses provider", id="chat_only_model"
        ),
        pytest.param(
            {"service_tier": "flex"},
            "native Responses does not implement the model's sampling rules",
            id="service_tier",
        ),
        pytest.param(
            {"reasoning": {"summary": "detailed"}},
            "native Responses does not implement the model's sampling rules",
            id="openai_path_reasoning_summary",
        ),
        pytest.param(
            {"model": "bedrock_mantle/openai.gpt-v1-path", "reasoning": {"summary": "detailed"}},
            None,
            id="v1_path_reasoning_summary",
        ),
    ],
)
def test_bedrock_mantle_requests_outside_the_native_config_stay_on_python(
    monkeypatch: pytest.MonkeyPatch, request_fields: dict[str, object], expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "model_cost", _MANTLE_MODEL_COST)
    request: Final = {"model": "bedrock_mantle/openai.gpt-openai-path", "input": "hi", **request_fields}
    assert decline_reason(request) == expected


def test_aws_settings_stay_on_python_for_other_providers() -> None:
    request: Final = {"model": "gpt-4.1", "input": "hi", "api_key": "sk-test", "aws_region_name": "us-west-2"}
    assert decline_reason(request) == "native inference does not implement aws_region_name"


def test_bedrock_mantle_connection_defaults_ignore_the_global_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "api_key", "global")
    monkeypatch.setattr(litellm, "api_base", "https://global.invalid")
    assert connection_defaults("bedrock_mantle") == (None, None)


_STREAMING_MODEL_COST: Final = MappingProxyType(
    {
        "gpt-batch-only": {
            "litellm_provider": "openai",
            "mode": "responses",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
            "supports_native_streaming": False,
        },
        "gpt-streams": {
            "litellm_provider": "openai",
            "mode": "responses",
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
        },
    }
)


@pytest.mark.parametrize(
    ("model", "stream", "expected"),
    [
        pytest.param("openai/gpt-streams", True, None, id="native_stream"),
        pytest.param(
            "openai/gpt-batch-only", True, "native Responses does not implement faked streams", id="faked_stream"
        ),
        pytest.param("openai/gpt-batch-only", False, None, id="faked_stream_model_without_stream"),
    ],
)
def test_streams_python_fakes_stay_on_python(
    monkeypatch: pytest.MonkeyPatch, model: str, stream: bool, expected: str | None
) -> None:
    monkeypatch.setattr(litellm, "model_cost", dict(_STREAMING_MODEL_COST))
    assert decline_reason({"model": model, "input": "hi", "api_key": "sk-test", "stream": stream}) == expected


@pytest.mark.parametrize(
    ("stream", "expected"), [(True, "native Responses streaming does not implement caching"), (False, None)]
)
def test_cached_streams_stay_on_python(monkeypatch: pytest.MonkeyPatch, stream: bool, expected: str | None) -> None:
    monkeypatch.setattr(litellm, "cache", litellm.Cache())
    assert decline_reason({"model": "gpt-4.1", "input": "hi", "api_key": "sk-test", "stream": stream}) == expected
