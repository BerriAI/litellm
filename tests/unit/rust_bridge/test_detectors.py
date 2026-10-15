from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import detectors
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_TOOL_TYPES

GATEWAY_TOOL: Final = {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
REMOTE_TOOL: Final = {"type": "mcp", "server_label": "docs", "server_url": "https://mcp.example.com/sse"}
ENCODED_ID: Final = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-1", "rs_1")  # pyright: ignore[reportPrivateUsage]  # the encoder litellm stamps on earlier turns
WRAPPED_CONTENT: Final = ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id("opaque", "deployment-1")


MAPPED_FILE: Final = {
    "model_file_id_mapping": {"litellm-file": {"deployment-1": "file-abc"}},
    "model_info": {"id": "deployment-1"},
}
MAPPED_FILE_INPUT: Final = [{"role": "user", "content": [{"type": "input_file", "file_id": "litellm-file"}]}]
CACHED_TEXT: Final = {"type": "input_text", "text": "hi", "cache_control": {"type": "ephemeral"}}
THINKING_TURN: Final = {"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "signature": "s"}]}
REDACTED: Final = {"type": "redacted_thinking", "data": "x"}
TOOL_SEARCH_TYPE: Final = sorted(ANTHROPIC_TOOL_SEARCH_TOOL_TYPES)[0]


def function_tool(parameters: object) -> Mapping[str, object]:
    return {"type": "function", "name": "f", "parameters": parameters}


def call(**kwargs: object) -> NativeCall:
    return NativeCall(args=(), kwargs=kwargs, base={"model": "gpt-5"})


@pytest.mark.parametrize(
    ("detector", "kwargs", "expected"),
    (
        (detectors.mock_response, {"mock_response": "hi"}, True),
        (detectors.mock_response, {"mock_response": None}, False),
        (detectors.caller_client, {"client": object()}, True),
        (detectors.caller_client, {}, False),
        (detectors.streaming, {"stream": True}, True),
        (detectors.streaming, {"stream": False}, False),
        (detectors.timeout_variants, {"request_timeout": 5}, True),
        (detectors.timeout_variants, {"stream": True, "stream_timeout": 5}, True),
        (detectors.timeout_variants, {"stream_timeout": 5}, False),
        (detectors.timeout_variants, {"timeout": httpx.Timeout(60, connect=2)}, True),
        (detectors.timeout_variants, {"timeout": 60}, False),
        (detectors.proxy_mcp_tools, {"tools": [GATEWAY_TOOL]}, True),
        (detectors.proxy_mcp_tools, {"tools": [REMOTE_TOOL]}, False),
        (detectors.proxy_mcp_tools, {"tools": None}, False),
        (detectors.prompt_management, {"prompt_id": "welcome"}, True),
        (detectors.prompt_management, {"prompt_variables": {"name": "x"}}, False),
        (detectors.text_format, {"text_format": dict}, True),
        (detectors.text_format, {}, False),
        (detectors.chat_completions_bridge, {"use_chat_completions_api": True}, True),
        (detectors.chat_completions_bridge, {"use_chat_completions_api": False}, False),
        (detectors.chat_completions_bridge, {"model": "openai/chat_completions/gpt-5"}, True),
        (detectors.chat_completions_bridge, {"model": "openai/gpt-5"}, False),
        (detectors.retries_fallbacks, {"num_retries": 0}, True),
        (detectors.retries_fallbacks, {"max_retries": 2}, True),
        (detectors.retries_fallbacks, {"fallbacks": ["gpt-5-mini"]}, True),
        (detectors.retries_fallbacks, {"retry_policy": {"TimeoutErrorRetries": 2}}, True),
        (detectors.retries_fallbacks, {}, False),
        (detectors.prompt_management, {"knowledge_bases": ["kb-1"]}, True),
        (detectors.prompt_management, {"vector_store_ids": []}, False),
        (detectors.managed_resource_ids, {"input": MAPPED_FILE_INPUT, **MAPPED_FILE}, True),
        (detectors.managed_resource_ids, {"input": MAPPED_FILE_INPUT}, False),
        (detectors.managed_resource_ids, {"input": "hi"}, False),
        (detectors.cache_control_marks, {"input": [{"role": "user", "content": [CACHED_TEXT]}]}, True),
        (detectors.cache_control_marks, {"tools": [{"type": "function", "name": "f", "cache_control": {}}]}, True),
        (detectors.cache_control_marks, {"input": [{"role": "user", "content": [{"type": "input_text"}]}]}, False),
        (detectors.anthropic_tool_search, {"tools": [{"type": TOOL_SEARCH_TYPE, "name": "tool_search"}]}, True),
        (detectors.anthropic_tool_search, {"tools": [{"type": "function", "name": "f"}]}, False),
        (detectors.tool_schema_repair, {"tools": [function_tool('{"type": "object"}')]}, True),
        (detectors.tool_schema_repair, {"tools": [function_tool({"type": "string", "pattern": r"\p{L}"})]}, True),
        (detectors.tool_schema_repair, {"tools": [function_tool({"type": "object"})]}, False),
        (detectors.tool_schema_repair, {}, False),
        (detectors.model_param_gating, {"max_output_tokens": 1}, True),
        (detectors.model_param_gating, {"max_output_tokens": 1024}, False),
        (detectors.cache_control_injection, {"enable_prompt_caching": True}, True),
        (detectors.cache_control_injection, {"cache_control_injection_points": [{"location": "message"}]}, True),
        (detectors.cache_control_injection, {"cache_control_injection_points": []}, False),
        (detectors.thinking_signature_retry, {"messages": [THINKING_TURN]}, True),
        (detectors.thinking_signature_retry, {"messages": [{"role": "assistant", "content": [REDACTED]}]}, True),
        (detectors.thinking_signature_retry, {"messages": [{"role": "assistant", "content": "done"}]}, False),
        (detectors.thinking_signature_retry, {"messages": [{"role": "user", "content": [{"type": "text"}]}]}, False),
        (detectors.drop_params, {"drop_params": True}, True),
        (detectors.drop_params, {"drop_params": False}, False),
        (detectors.drop_params, {"drop_params": " True "}, True),
        (detectors.drop_params, {"drop_params": "false"}, False),
        (detectors.drop_params, {"drop_params": "maybe"}, False),
        (detectors.drop_params, {"additional_drop_params": ["metadata.user_id"]}, True),
        (detectors.drop_params, {"additional_drop_params": []}, False),
        (detectors.deployment_response_id, {"litellm_metadata": {"model_info": {"id": "deployment-1"}}}, True),
        (detectors.deployment_response_id, {"litellm_metadata": {"model_info": {}}}, False),
        (detectors.deployment_response_id, {"litellm_metadata": None}, False),
        (detectors.encoded_item_ids, {"input": [{"type": "reasoning", "id": ENCODED_ID}]}, True),
        (detectors.encoded_item_ids, {"input": [{"type": "reasoning", "encrypted_content": WRAPPED_CONTENT}]}, True),
        (detectors.encoded_item_ids, {"input": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}]}, False),
        (detectors.encoded_item_ids, {"input": "hi"}, False),
    ),
)
def test_request_detectors(
    detector: Callable[[NativeCall], bool], kwargs: Mapping[str, object], expected: bool
) -> None:
    assert detector(call(**kwargs)) is expected


@pytest.mark.parametrize(
    ("detector", "setting", "value", "kwargs"),
    (
        (detectors.drop_params, "drop_params", True, {}),
        (detectors.modify_params, "modify_params", True, {}),
        (detectors.retries_fallbacks, "num_retries", 3, {}),
        (detectors.cache_control_injection, "enable_anthropic_prompt_caching", True, {}),
        (detectors.stream_cost_injection, "include_cost_in_streaming_usage", True, {"stream": True}),
    ),
)
def test_global_settings_are_detected(
    monkeypatch: pytest.MonkeyPatch,
    detector: Callable[[NativeCall], bool],
    setting: str,
    value: object,
    kwargs: Mapping[str, object],
) -> None:
    for name in ("drop_params", "modify_params", "enable_anthropic_prompt_caching", "include_cost_in_streaming_usage"):
        monkeypatch.setattr(litellm, name, False)
    monkeypatch.setattr(litellm, "num_retries", None)
    assert not detector(call(**kwargs))
    monkeypatch.setattr(litellm, setting, value)
    assert detector(call(**kwargs))


def test_stream_cost_injection_needs_a_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    assert not detectors.stream_cost_injection(call())


def test_openai_workload_identity_applies_only_without_a_static_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    identity: Final = {
        "openai_identity_provider_id": "idp-1",
        "openai_service_account_id": "sa-1",
        "openai_identity_token_file": "/var/run/token",
    }
    assert detectors.openai_workload_identity(call(**identity))
    assert not detectors.openai_workload_identity(call(**identity, api_key="sk-static"))
    assert not detectors.openai_workload_identity(call())


def test_anthropic_workload_identity_applies_only_without_a_static_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    federation: Final = {"anthropic_federation_rule_id": "rule-1", "anthropic_organization_id": "org-1"}
    assert detectors.anthropic_workload_identity(call(**federation))
    assert not detectors.anthropic_workload_identity(call(**federation, api_key="sk-ant-static"))
    assert not detectors.anthropic_workload_identity(call(anthropic_federation_rule_id="rule-1"))


class _Logger(CustomLogger):
    pass


class _Rewrite(CustomLogger):
    async def async_pre_request_hook(self, model: str, messages: list, kwargs: dict) -> dict | None:  # pyright: ignore[reportMissingTypeArgument]  # mirrors the CustomLogger signature
        return kwargs


class _InheritedRewrite(_Rewrite):
    pass


class _Agentic(CustomLogger):
    async def async_should_run_agentic_loop(self, *args: object, **kwargs: object) -> tuple[bool, dict]:  # pyright: ignore[reportIncompatibleMethodOverride, reportMissingTypeArgument]  # test double only needs the override
        return False, {}


@pytest.mark.parametrize(
    ("detector", "callbacks", "expected"),
    (
        (detectors.pre_request_hook, [_Rewrite()], True),
        (detectors.pre_request_hook, [_InheritedRewrite()], True),
        (detectors.pre_request_hook, [_Logger(), _Agentic()], False),
        (detectors.pre_request_hook, ["not-a-logger"], False),
        (detectors.agentic_loop_hook, [_Agentic()], True),
        (detectors.agentic_loop_hook, [_Logger(), _Rewrite()], False),
    ),
)
def test_callback_detectors_need_an_override(
    monkeypatch: pytest.MonkeyPatch, detector: Callable[[NativeCall], bool], callbacks: list[object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "callbacks", callbacks)
    assert detector(call()) is expected


def _logging_with(callback: CustomLogger) -> Logging:
    logging_obj: Final = Logging(
        model="claude-sonnet-4-5",
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="function-1",
    )
    logging_obj.add_dynamic_callback(callback)
    return logging_obj


@pytest.mark.parametrize(
    ("detector", "kwargs", "expected"),
    (
        (detectors.agentic_loop_hook, {"success_callback": [_Agentic()]}, True),
        (detectors.agentic_loop_hook, {"litellm_logging_obj": _logging_with(_Agentic())}, True),
        (detectors.agentic_loop_hook, {"success_callback": [_Logger()]}, False),
        (detectors.pre_request_hook, {"success_callback": [_Rewrite()]}, False),
    ),
    ids=("success-callback", "logging-object", "plain-logger", "pre-request-reads-globals-only"),
)
def test_callback_detectors_read_the_request_callbacks_the_handler_reads(
    monkeypatch: pytest.MonkeyPatch,
    detector: Callable[[NativeCall], bool],
    kwargs: Mapping[str, object],
    expected: bool,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    assert detector(call(**kwargs)) is expected


def test_websearch_interception_needs_the_interception_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger

    monkeypatch.setattr(litellm, "callbacks", [_Rewrite()])
    assert not detectors.websearch_interception(call())
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger()])
    assert detectors.websearch_interception(call())


@pytest.mark.parametrize(
    ("cache", "kwargs", "expected"),
    (
        (None, {}, False),
        (Cache(), {}, True),
        (Cache(), {"cache": {"ttl": 60}}, True),
        (Cache(), {"cache": {"no-cache": True}}, False),
    ),
)
def test_response_cache_is_skipped_only_by_no_cache(
    monkeypatch: pytest.MonkeyPatch, cache: Cache | None, kwargs: Mapping[str, object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "cache", cache)
    assert detectors.response_cache(call(**kwargs)) is expected


@pytest.mark.parametrize(
    ("environment", "setting", "expected"),
    ((None, None, False), ("DEFAULT@SECLEVEL=1", None, True), (None, "DEFAULT@SECLEVEL=1", True)),
)
def test_tls_policy_reads_the_security_level_environment_then_the_setting(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, setting: str | None, expected: bool
) -> None:
    monkeypatch.delenv("SSL_SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("SSL_ECDH_CURVE", raising=False)
    monkeypatch.setattr(litellm, "ssl_ecdh_curve", None)
    if environment is not None:
        monkeypatch.setenv("SSL_SECURITY_LEVEL", environment)
    monkeypatch.setattr(litellm, "ssl_security_level", setting)
    assert detectors.tls_policy(call()) is expected


@pytest.mark.parametrize(
    ("environment", "setting", "expected"),
    ((None, "X25519", False), (None, " P-256 ", False), ("secp521r1", None, True), (None, "secp521r1", True)),
)
def test_tls_policy_flags_only_curves_native_tls_lacks(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, setting: str | None, expected: bool
) -> None:
    monkeypatch.delenv("SSL_SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("SSL_ECDH_CURVE", raising=False)
    monkeypatch.setattr(litellm, "ssl_security_level", None)
    if environment is not None:
        monkeypatch.setenv("SSL_ECDH_CURVE", environment)
    monkeypatch.setattr(litellm, "ssl_ecdh_curve", setting)
    assert detectors.tls_policy(call()) is expected
