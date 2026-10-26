from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime
from functools import reduce
from typing import Final

import httpx
import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import detectors
from litellm.rust_bridge.chat_completions import detectors as chat_detectors
from litellm.rust_bridge.messages import detectors as messages_detectors
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.responses import detectors as responses_detectors
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE, ANTHROPIC_TOOL_SEARCH_TOOL_TYPES

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
    ("model", "provider", "tools", "expected"),
    (
        ("deepseek/test-model", None, [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE}], True),
        ("anthropic/test-model", None, [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE}], False),
        ("test-model", "anthropic", [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE}], False),
        ("unresolved-model", None, [{"type": ANTHROPIC_ADVISOR_TOOL_TYPE}], True),
        ("deepseek/test-model", None, [{"type": "function", "name": "f"}], False),
        ("deepseek/test-model", None, ["not-a-tool"], False),
        ("deepseek/test-model", None, [], False),
        ("deepseek/test-model", None, None, False),
    ),
)
def test_messages_interception_matches_tools_and_resolved_provider(
    model: str, provider: str | None, tools: object, expected: bool
) -> None:
    assert messages_detectors.intercepted(call(model=model, custom_llm_provider=provider, tools=tools)) is expected


def _wrap_nested_level(value: object, depth: int) -> object:
    return {"nested": value} if depth % 2 else [value]


@pytest.mark.parametrize(
    ("detector", "kwargs", "expected"),
    (
        (detectors.mock_response, {"mock_response": "hi"}, True),
        (detectors.mock_response, {"mock_response": None}, False),
        (detectors.mock_response, {"mock_tool_calls": [{"id": "call_1"}]}, False),
        (chat_detectors.chat_mock_response, {"mock_response": "hi"}, True),
        (chat_detectors.chat_mock_response, {"mock_tool_calls": [{"id": "call_1"}]}, True),
        (chat_detectors.chat_mock_response, {"mock_timeout": True}, True),
        (chat_detectors.chat_mock_response, {"mock_tool_calls": [], "mock_timeout": False}, False),
        (responses_detectors.client_headers, {"headers": {"x-tenant": "a"}}, True),
        (responses_detectors.client_headers, {"headers": {}}, False),
        (responses_detectors.client_headers, {"headers": None}, False),
        (
            chat_detectors.model_list,
            {"model_list": [{"model_name": "gpt-5", "litellm_params": {"model": "gpt-5"}}]},
            True,
        ),
        (chat_detectors.model_list, {"model_list": None}, False),
        (chat_detectors.chat_retries_fallbacks, {"context_window_fallback_dict": {"gpt-5": "gpt-5-mini"}}, True),
        (chat_detectors.chat_retries_fallbacks, {"fallbacks": ["gpt-5-mini"]}, True),
        (detectors.retries_fallbacks, {"context_window_fallback_dict": {"gpt-5": "gpt-5-mini"}}, False),
        (detectors.prompt_management, {"model": "litellm_agent/gpt-5"}, True),
        (detectors.prompt_management, {"model": "gpt-5"}, False),
        (detectors.caller_client, {"client": object()}, True),
        (detectors.caller_client, {"shared_session": object()}, True),
        (detectors.caller_client, {"client": None, "shared_session": None}, False),
        (detectors.caller_client, {}, False),
        (detectors.extra_body, {"extra_body": {"stream": True}}, True),
        (detectors.extra_body, {"extra_body": {"stream": False}}, True),
        (detectors.extra_body, {"extra_body": {}}, False),
        (detectors.extra_body, {"extra_body": None}, False),
        (detectors.extra_body, {}, False),
        (
            responses_detectors.provider_affinity,
            {"provider_affinity_header": "X-Session-ID", "session_id": "session-1"},
            True,
        ),
        (responses_detectors.provider_affinity, {"provider_affinity_header": "X-Session-ID"}, True),
        (
            responses_detectors.provider_affinity,
            {"provider_affinity_header": "X-Session-ID", "extra_headers": {"x-session-id": "session-1"}},
            True,
        ),
        (responses_detectors.provider_affinity, {"provider_affinity_header": "Authorization"}, True),
        (responses_detectors.provider_affinity, {"provider_affinity_header": "invalid header"}, True),
        (responses_detectors.provider_affinity, {"provider_affinity_header": ""}, True),
        (responses_detectors.provider_affinity, {"provider_affinity_header": None, "session_id": "session-1"}, False),
        (responses_detectors.provider_affinity, {}, False),
        (responses_detectors.provider_affinity, {"api_base": "https://opencode.ai/inference/go/openai/v1"}, True),
        (messages_detectors.provider_affinity, {"api_base": "https://opencode.ai/inference/go/anthropic/v1"}, True),
        (messages_detectors.provider_affinity, {"api_base": "https://other.example/v1"}, False),
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
        (responses_detectors.text_format, {"text_format": dict}, True),
        (responses_detectors.text_format, {}, False),
        (responses_detectors.chat_completions_bridge, {"use_chat_completions_api": True}, True),
        (responses_detectors.chat_completions_bridge, {"use_chat_completions_api": False}, False),
        (responses_detectors.chat_completions_bridge, {"model": "openai/chat_completions/gpt-5"}, True),
        (responses_detectors.chat_completions_bridge, {"model": "openai/gpt-5"}, False),
        (detectors.retries_fallbacks, {"num_retries": 0}, True),
        (detectors.retries_fallbacks, {"max_retries": 2}, True),
        (detectors.retries_fallbacks, {"fallbacks": ["gpt-5-mini"]}, True),
        (detectors.retries_fallbacks, {"retry_policy": {"TimeoutErrorRetries": 2}}, True),
        (detectors.retries_fallbacks, {}, False),
        (detectors.prompt_management, {"knowledge_bases": ["kb-1"]}, True),
        (detectors.prompt_management, {"vector_store_ids": []}, False),
        (responses_detectors.managed_resource_ids, {"input": MAPPED_FILE_INPUT, **MAPPED_FILE}, True),
        (responses_detectors.managed_resource_ids, {"input": MAPPED_FILE_INPUT}, False),
        (responses_detectors.managed_resource_ids, {"input": "hi"}, False),
        (responses_detectors.cache_control_marks, {"input": [{"role": "user", "content": [CACHED_TEXT]}]}, True),
        (
            responses_detectors.cache_control_marks,
            {"tools": [{"type": "function", "name": "f", "cache_control": {}}]},
            True,
        ),
        (
            responses_detectors.cache_control_marks,
            {"input": [{"role": "user", "content": [{"type": "input_text"}]}]},
            False,
        ),
        (
            responses_detectors.anthropic_tool_search,
            {"tools": [{"type": TOOL_SEARCH_TYPE, "name": "tool_search"}]},
            True,
        ),
        (responses_detectors.anthropic_tool_search, {"tools": [{"type": "function", "name": "f"}]}, False),
        (responses_detectors.tool_schema_repair, {"tools": [function_tool('{"type": "object"}')]}, True),
        (
            responses_detectors.tool_schema_repair,
            {"tools": [function_tool({"type": "string", "pattern": r"\p{L}"})]},
            True,
        ),
        (responses_detectors.tool_schema_repair, {"tools": [function_tool({"type": "object"})]}, False),
        (responses_detectors.tool_schema_repair, {}, False),
        (responses_detectors.model_param_gating, {"max_output_tokens": 1}, True),
        (responses_detectors.model_param_gating, {"max_output_tokens": 1024}, False),
        (detectors.cache_control_injection, {"enable_prompt_caching": True}, True),
        (detectors.cache_control_injection, {"cache_control_injection_points": [{"location": "message"}]}, True),
        (detectors.cache_control_injection, {"cache_control_injection_points": []}, False),
        (messages_detectors.thinking_signature_retry, {"messages": [THINKING_TURN]}, True),
        (
            messages_detectors.thinking_signature_retry,
            {"messages": [{"role": "assistant", "content": [REDACTED]}]},
            True,
        ),
        (messages_detectors.thinking_signature_retry, {"messages": [{"role": "assistant", "content": "done"}]}, False),
        (
            messages_detectors.thinking_signature_retry,
            {"messages": [{"role": "user", "content": [{"type": "text"}]}]},
            False,
        ),
        (detectors.drop_params, {"drop_params": True}, True),
        (detectors.drop_params, {"drop_params": False}, False),
        (detectors.drop_params, {"drop_params": " True "}, True),
        (detectors.drop_params, {"drop_params": "false"}, False),
        (detectors.drop_params, {"drop_params": "maybe"}, False),
        (detectors.drop_params, {"additional_drop_params": ["metadata.user_id"]}, True),
        (detectors.drop_params, {"additional_drop_params": []}, False),
        (
            responses_detectors.deployment_response_id,
            {"litellm_metadata": {"model_info": {"id": "deployment-1"}}},
            True,
        ),
        (responses_detectors.deployment_response_id, {"litellm_metadata": {"model_info": {}}}, False),
        (responses_detectors.deployment_response_id, {"litellm_metadata": None}, False),
        (responses_detectors.encoded_item_ids, {"input": [{"type": "reasoning", "id": ENCODED_ID}]}, True),
        (
            responses_detectors.encoded_item_ids,
            {"input": [{"type": "reasoning", "encrypted_content": WRAPPED_CONTENT}]},
            True,
        ),
        (
            responses_detectors.encoded_item_ids,
            {"input": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x"}]},
            False,
        ),
        (responses_detectors.encoded_item_ids, {"input": "hi"}, False),
    ),
)
def test_request_detectors(
    detector: Callable[[NativeCall], bool], kwargs: Mapping[str, object], expected: bool
) -> None:
    assert detector(call(**kwargs)) is expected


def test_cache_control_marks_handles_deeply_nested_input() -> None:
    nested_input: Final = reduce(_wrap_nested_level, range(5_000), {"cache_control": {}})
    assert responses_detectors.cache_control_marks(call(input=nested_input))


@pytest.mark.parametrize(
    ("input_value", "expected"),
    (
        pytest.param(
            [{"type": "reasoning", "id": "rs_00000000-0000-4000-8000-000000000000", "summary": list[object]()}],
            True,
            id="bridge-minted-reasoning",
        ),
        pytest.param(
            [{"type": "reasoning", "encrypted_content": '[{"type":"thinking","thinking":"x","signature":"s"}]'}],
            True,
            id="bridge-thinking-blocks",
        ),
        pytest.param(
            [{"type": "function_call", "id": "foreign-id", "call_id": "call_1", "name": "f", "arguments": "{}"}],
            True,
            id="foreign-tool-call-id",
        ),
        pytest.param(
            [{"type": "reasoning", "id": "rs_1", "summary": list[object](), "status": None}],
            True,
            id="null-reasoning-status",
        ),
        pytest.param(
            [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": "{}"}],
            False,
            id="native-tool-call-id",
        ),
        pytest.param(
            [{"type": "custom_tool_call", "call_id": "call_1", "name": "f", "input": "x", "namespace": "ns"}],
            True,
            id="custom-tool-call-namespace",
        ),
        pytest.param(
            [{"type": "custom_tool_call", "call_id": "call_1", "name": "f", "input": "x"}],
            False,
            id="custom-tool-call",
        ),
        pytest.param([{"type": "reasoning", "id": "rs_1", "summary": list[object]()}], False, id="native-reasoning"),
        pytest.param([{"role": "user", "content": "hi"}], False, id="message"),
        pytest.param("hi", False, id="string"),
        pytest.param([], False, id="empty"),
        pytest.param(None, False, id="absent"),
    ),
)
def test_input_rewrite_preserves_caller_input(input_value: object, expected: bool) -> None:
    original: Final = deepcopy(input_value)
    assert responses_detectors.input_rewrite(call(input=input_value)) is expected
    assert input_value == original


@pytest.mark.parametrize(
    ("detector", "setting", "value", "kwargs"),
    (
        (detectors.drop_params, "drop_params", True, {}),
        (detectors.modify_params, "modify_params", True, {}),
        (detectors.retries_fallbacks, "num_retries", 3, {}),
        (detectors.cache_control_injection, "enable_anthropic_prompt_caching", True, {}),
        (messages_detectors.stream_cost_injection, "include_cost_in_streaming_usage", True, {"stream": True}),
        (messages_detectors.stream_cost_injection, "include_cost_in_streaming_usage", "true", {"stream": True}),
        (chat_detectors.chat_retries_fallbacks, "model_fallbacks", [{"gpt-5": ["gpt-5-mini"]}], {}),
        (chat_detectors.chat_retries_fallbacks, "num_retries", 3, {}),
        (chat_detectors.model_alias_map, "model_alias_map", {"gpt-5": "openai/gpt-5"}, {}),
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
    for name in ("num_retries", "model_fallbacks"):
        monkeypatch.setattr(litellm, name, None)
    monkeypatch.setattr(litellm, "model_alias_map", {})
    assert not detector(call(**kwargs))
    monkeypatch.setattr(litellm, setting, value)
    assert detector(call(**kwargs))


def test_stream_cost_injection_needs_a_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", True)
    assert not messages_detectors.stream_cost_injection(call())


def test_model_alias_map_applies_only_to_the_requested_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "model_alias_map", {"fast": "openai/gpt-5-mini"})
    assert chat_detectors.model_alias_map(call(model="fast"))
    assert not chat_detectors.model_alias_map(call())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    (
        pytest.param({"temperature": 0.2}, True, id="gpt-5-temperature"),
        pytest.param({"max_tokens": 16}, False, id="gpt-5-max-tokens"),
        pytest.param({"model": "gpt-4o", "temperature": 0.2}, False, id="gpt-4o-temperature"),
        pytest.param({"model": "anthropic/claude-sonnet-4-5", "logit_bias": {"1": 1}}, True, id="unsupported-param"),
        pytest.param({"model": "anthropic/claude-sonnet-4-5", "n": 1}, False, id="langchain-default-n"),
        pytest.param({"model": "anthropic/claude-sonnet-4-5", "n": 2}, True, id="unsupported-n"),
        pytest.param({"model": "anthropic/claude-sonnet-4-5", "max_tokens": 16, "stream": True}, False, id="supported"),
        pytest.param({"model": "unknown-model", "temperature": 0.2}, False, id="unresolved-provider"),
        pytest.param({}, False, id="no-params"),
    ),
)
def test_chat_model_param_gating_follows_the_provider_param_rules(
    monkeypatch: pytest.MonkeyPatch, kwargs: Mapping[str, object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    assert chat_detectors.chat_model_param_gating(call(**kwargs)) is expected


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
    assert responses_detectors.openai_workload_identity(call(**identity))
    assert not responses_detectors.openai_workload_identity(call(**identity, api_key="sk-static"))
    assert not responses_detectors.openai_workload_identity(call())


def test_anthropic_workload_identity_applies_only_without_a_static_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    federation: Final = {"anthropic_federation_rule_id": "rule-1", "anthropic_organization_id": "org-1"}
    assert messages_detectors.anthropic_workload_identity(call(**federation))
    assert not messages_detectors.anthropic_workload_identity(call(**federation, api_key="sk-ant-static"))
    assert not messages_detectors.anthropic_workload_identity(call(anthropic_federation_rule_id="rule-1"))


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


class _ChatAgentic(CustomLogger):
    async def async_should_run_chat_completion_agentic_loop(
        self, *args: object, **kwargs: object
    ) -> tuple[bool, dict[str, object]]:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return False, {}


class _Deployment(CustomLogger):
    async def async_pre_call_deployment_hook(self, *args: object, **kwargs: object) -> dict[str, object] | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # test double only needs the override
        return None


@pytest.mark.parametrize(
    ("detector", "callbacks", "expected"),
    (
        (messages_detectors.pre_request_hook, [_Rewrite()], True),
        (messages_detectors.pre_request_hook, [_InheritedRewrite()], True),
        (messages_detectors.pre_request_hook, [_Logger(), _Agentic()], False),
        (messages_detectors.pre_request_hook, ["not-a-logger"], False),
        (detectors.agentic_loop_hook, [_Agentic()], True),
        (detectors.agentic_loop_hook, [_Logger(), _Rewrite()], False),
        (detectors.agentic_loop_hook, [_ChatAgentic()], False),
        (chat_detectors.chat_agentic_loop_hook, [_ChatAgentic()], True),
        (chat_detectors.chat_agentic_loop_hook, [_Agentic()], True),
        (chat_detectors.chat_agentic_loop_hook, [_Logger(), _Deployment()], False),
        (responses_detectors.deployment_hook, [_Deployment()], True),
        (responses_detectors.deployment_hook, [_Logger(), _Agentic(), _Rewrite()], False),
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
        (messages_detectors.pre_request_hook, {"success_callback": [_Rewrite()]}, False),
        (chat_detectors.chat_agentic_loop_hook, {"litellm_logging_obj": _logging_with(_ChatAgentic())}, True),
        (responses_detectors.deployment_hook, {"success_callback": [_Deployment()]}, True),
    ),
    ids=(
        "success-callback",
        "logging-object",
        "plain-logger",
        "pre-request-reads-globals-only",
        "chat-agentic-logging-object",
        "deployment-success-callback",
    ),
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
    assert not messages_detectors.websearch_interception(call())
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger()])
    assert messages_detectors.websearch_interception(call())


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
