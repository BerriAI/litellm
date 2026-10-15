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

GATEWAY_TOOL: Final = {"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}
REMOTE_TOOL: Final = {"type": "mcp", "server_label": "docs", "server_url": "https://mcp.example.com/sse"}
ENCODED_ID: Final = ResponsesAPIRequestUtils._build_encrypted_item_id("deployment-1", "rs_1")  # pyright: ignore[reportPrivateUsage]  # the encoder litellm stamps on earlier turns
WRAPPED_CONTENT: Final = ResponsesAPIRequestUtils.wrap_encrypted_content_with_model_id("opaque", "deployment-1")


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
        (detectors.retries_fallbacks, {"num_retries": 0}, True),
        (detectors.retries_fallbacks, {"max_retries": 2}, True),
        (detectors.retries_fallbacks, {"fallbacks": ["gpt-5-mini"]}, True),
        (detectors.retries_fallbacks, {}, False),
        (detectors.drop_modify_params, {"drop_params": True}, True),
        (detectors.drop_modify_params, {"drop_params": False}, False),
        (detectors.drop_modify_params, {"drop_params": " True "}, True),
        (detectors.drop_modify_params, {"drop_params": "false"}, False),
        (detectors.drop_modify_params, {"drop_params": "maybe"}, False),
        (detectors.drop_modify_params, {"additional_drop_params": ["metadata.user_id"]}, True),
        (detectors.drop_modify_params, {"additional_drop_params": []}, False),
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


@pytest.mark.parametrize("setting", ("drop_params", "modify_params"))
def test_global_drop_modify_params_are_detected(monkeypatch: pytest.MonkeyPatch, setting: str) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "modify_params", False)
    assert not detectors.drop_modify_params(call())
    monkeypatch.setattr(litellm, setting, True)
    assert detectors.drop_modify_params(call())


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
def test_ssl_security_level_reads_the_environment_then_the_setting(
    monkeypatch: pytest.MonkeyPatch, environment: str | None, setting: str | None, expected: bool
) -> None:
    monkeypatch.delenv("SSL_SECURITY_LEVEL", raising=False)
    if environment is not None:
        monkeypatch.setenv("SSL_SECURITY_LEVEL", environment)
    monkeypatch.setattr(litellm, "ssl_security_level", setting)
    assert detectors.ssl_security_level(call()) is expected
