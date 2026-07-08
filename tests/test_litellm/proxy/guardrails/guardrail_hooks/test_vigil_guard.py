import json
import logging
import ssl
from types import SimpleNamespace
from typing import Any, List

import httpx
import pytest

from litellm.exceptions import GuardrailRaisedException
from litellm.exceptions import Timeout as LiteLLMTimeout
from litellm.proxy.guardrails.guardrail_hooks.vigil_guard import (
    VigilGuardGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.vigil_guard.vigil_guard import (
    _DEFAULT_VIGIL_TIMEOUT,
    VigilGuardMissingConfig,
)
from litellm.types.guardrails import LitellmParams, SupportedGuardrailIntegrations
from litellm.types.proxy.guardrails.guardrail_hooks.vigil_guard import (
    VigilGuardGuardrailConfigModel,
)

_ENDPOINT = "https://vigil.test/v1/guard/analyze"


def _resp(body: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=body,
        request=httpx.Request("POST", _ENDPOINT),
    )


class FakeHandler:
    def __init__(self, items: List[Any]):
        self._items = list(items)
        self.calls: List[SimpleNamespace] = []

    async def post(self, *, url, headers, json, timeout=None):  # noqa: A002
        self.calls.append(
            SimpleNamespace(url=url, headers=headers, json=json, timeout=timeout)
        )
        if not self._items:
            raise AssertionError("FakeHandler ran out of programmed responses")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_guardrail(
    handler: FakeHandler,
    *,
    unreachable_fallback="fail_closed",
    api_base="https://vigil.test",
    api_key="vg_secret_key_123",
    guardrail_name="vigil-guard",
    timeout=None,
) -> VigilGuardGuardrail:
    return VigilGuardGuardrail(
        api_base=api_base,
        api_key=api_key,
        unreachable_fallback=unreachable_fallback,
        timeout=timeout,
        async_handler=handler,
        guardrail_name=guardrail_name,
        event_hook="pre_call",
        default_on=True,
    )


def _transient_exceptions() -> List[BaseException]:
    req = httpx.Request("POST", _ENDPOINT)
    return [
        httpx.ConnectError("boom", request=req),
        httpx.ConnectTimeout("boom", request=req),
        httpx.ReadTimeout("boom", request=req),
        httpx.RemoteProtocolError("boom", request=req),
        LiteLLMTimeout(message="t", model="m", llm_provider="vigil_guard"),
    ]


def test_requires_api_base(monkeypatch):
    monkeypatch.delenv("VIGIL_GUARD_URL", raising=False)
    monkeypatch.delenv("VIGIL_GUARD_API_KEY", raising=False)
    with pytest.raises(VigilGuardMissingConfig):
        VigilGuardGuardrail(api_key="k", async_handler=FakeHandler([]))


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("VIGIL_GUARD_API_KEY", raising=False)
    with pytest.raises(VigilGuardMissingConfig):
        VigilGuardGuardrail(
            api_base="https://vigil.test", async_handler=FakeHandler([])
        )


def test_trailing_slash_stripped():
    g = _make_guardrail(FakeHandler([]), api_base="https://vigil.test/")
    assert g.api_base == "https://vigil.test"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("VIGIL_GUARD_URL", "https://env.vigil.test")
    monkeypatch.setenv("VIGIL_GUARD_API_KEY", "env_key")
    g = VigilGuardGuardrail(
        async_handler=FakeHandler([]),
        guardrail_name="vg",
        event_hook="pre_call",
        default_on=True,
    )
    assert g.api_base == "https://env.vigil.test"
    assert g.api_key == "env_key"


def test_default_unreachable_fallback_is_fail_closed():
    g = _make_guardrail(FakeHandler([]), unreachable_fallback=None)
    assert g.unreachable_fallback == "fail_closed"


def test_explicit_fail_open_is_stored():
    g = _make_guardrail(FakeHandler([]), unreachable_fallback="fail_open")
    assert g.unreachable_fallback == "fail_open"


def test_unknown_fallback_defaults_to_fail_closed():
    g = _make_guardrail(FakeHandler([]), unreachable_fallback="weird")
    assert g.unreachable_fallback == "fail_closed"


async def test_allowed_preserves_full_input_shape_and_logs_allow():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    structured = [{"role": "user", "content": "hello"}]
    inputs = {"texts": ["hello"], "structured_messages": structured, "model": "gpt-4o"}
    request_data = {"metadata": {}}
    out = await g.apply_guardrail(
        inputs=inputs, request_data=request_data, input_type="request", logging_obj=None
    )
    assert out["texts"] == ["hello"]
    assert out["structured_messages"] is structured
    assert out["model"] == "gpt-4o"
    assert out is not inputs
    assert inputs["structured_messages"] is structured
    assert len(handler.calls) == 1
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert entries[0]["guardrail_response"] == "allow"


async def test_sanitized_replaces_text():
    handler = FakeHandler(
        [_resp({"decision": "SANITIZED", "sanitizedText": "[REDACTED]"})]
    )
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["my ssn is 123"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["[REDACTED]"]


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            {
                "decision": "SANITIZED",
                "sanitizedText": "S",
                "outputText": "O",
            },
            "S",
        ),
        ({"decision": "SANITIZED", "outputText": "O"}, "O"),
        ({"decision": "SANITIZED", "sanitizedText": 123, "outputText": "O"}, "O"),
        ({"decision": "SANITIZED", "sanitizedText": ""}, ""),
        ({"decision": "SANITIZED"}, "orig"),
    ],
)
async def test_sanitized_precedence(body, expected):
    handler = FakeHandler([_resp(body)])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["orig"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == [expected]


async def test_blocked_raises_guardrail_exception_with_400():
    handler = FakeHandler([_resp({"decision": "BLOCKED", "blockMessage": "nope"})])
    g = _make_guardrail(handler)
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": ["bad"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.guardrail_name == "vigil-guard"
    assert exc_info.value.message == "nope"


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            {
                "decision": "BLOCKED",
                "blockMessage": "bm",
                "decisionReason": "dr",
                "categories": ["c1"],
            },
            "bm",
        ),
        ({"decision": "BLOCKED", "blockMessage": "   ", "decisionReason": "dr"}, "dr"),
        (
            {"decision": "BLOCKED", "decisionReason": "dr", "categories": ["c1", "c2"]},
            "dr",
        ),
        ({"decision": "BLOCKED", "categories": ["c1", "c2"]}, "c1, c2"),
        ({"decision": "BLOCKED"}, "Blocked by policy"),
    ],
)
async def test_block_reason_precedence(body, expected):
    handler = FakeHandler([_resp(body)])
    g = _make_guardrail(handler)
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.message == expected


async def test_block_reason_is_clamped_to_500_chars():
    handler = FakeHandler([_resp({"decision": "BLOCKED", "blockMessage": "x" * 600})])
    g = _make_guardrail(handler)
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert "x" * 500 in exc_info.value.message
    assert "x" * 501 not in exc_info.value.message


async def test_empty_and_whitespace_texts_skip_analyze():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["", "   ", "real"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["", "   ", "real"]
    assert len(handler.calls) == 1
    assert handler.calls[0].json["text"] == "real"


async def test_no_scannable_text_returns_inputs_unchanged():
    handler = FakeHandler([])
    g = _make_guardrail(handler)
    inputs = {"texts": ["", "  "], "structured_messages": [{"role": "user"}]}
    out = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert out is inputs
    assert len(handler.calls) == 0


async def test_multi_text_preserves_length_and_order():
    handler = FakeHandler(
        [
            _resp({"decision": "ALLOWED"}),
            _resp({"decision": "SANITIZED", "sanitizedText": "B-clean"}),
            _resp({"decision": "ALLOWED"}),
        ]
    )
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["A", "B", "C"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["A", "B-clean", "C"]
    assert len(handler.calls) == 3


async def test_one_blocked_text_blocks_the_whole_call():
    handler = FakeHandler(
        [
            _resp({"decision": "ALLOWED"}),
            _resp({"decision": "BLOCKED", "blockMessage": "bad second"}),
        ]
    )
    g = _make_guardrail(handler)
    with pytest.raises(GuardrailRaisedException):
        await g.apply_guardrail(
            inputs={"texts": ["ok", "bad"]}, request_data={}, input_type="request"
        )


async def test_request_source_is_user_input():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert handler.calls[0].json["source"] == "user_input"


async def test_response_source_is_model_output():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="response"
    )
    assert handler.calls[0].json["source"] == "model_output"


async def test_sanitized_returns_canonical_shape_and_logs_mask():
    handler = FakeHandler(
        [_resp({"decision": "SANITIZED", "sanitizedText": "[REDACTED]"})]
    )
    g = _make_guardrail(handler)
    tools = [{"type": "function", "function": {"name": "f"}}]
    inputs = {
        "texts": ["my ssn is 123"],
        "images": ["img1"],
        "tools": tools,
        "tool_calls": [{"id": "1"}],
        "structured_messages": [{"role": "user", "content": "my ssn is 123"}],
        "model": "gpt-4o",
    }
    request_data = {"metadata": {}}
    out = await g.apply_guardrail(
        inputs=inputs, request_data=request_data, input_type="request"
    )
    assert out["texts"] == ["[REDACTED]"]
    assert out["images"] == ["img1"]
    assert out["tools"] == tools
    assert set(out.keys()) == {"texts", "images", "tools"}
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert entries[0]["guardrail_response"] == "mask"


async def test_empty_images_and_tools_are_preserved_when_present():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["x"], "images": [], "tools": []},
        request_data={},
        input_type="request",
    )
    assert set(out.keys()) == {"texts", "images", "tools"}
    assert out["images"] == []
    assert out["tools"] == []


async def test_logging_obj_none_supported():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request", logging_obj=None
    )
    assert out["texts"] == ["x"]


async def test_standard_guardrail_logging_remains_active():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    request_data = {"metadata": {}}
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data=request_data, input_type="request"
    )
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    assert entries[0]["guardrail_name"] == "vigil-guard"
    assert entries[0]["guardrail_status"] == "success"


async def test_request_url_headers_and_body():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler, api_base="https://vigil.test", api_key="vg_secret")
    await g.apply_guardrail(
        inputs={"texts": ["hello"]}, request_data={}, input_type="request"
    )
    call = handler.calls[0]
    assert call.url == "https://vigil.test/v1/guard/analyze"
    assert call.headers["Authorization"] == "Bearer vg_secret"
    assert call.headers["Content-Type"] == "application/json"
    assert call.json["text"] == "hello"
    assert call.json["mode"] == "full"
    assert set(call.json.keys()) == {"text", "source", "mode", "metadata"}
    assert "metadata" in call.json


async def test_default_timeout_forwarded_when_unset():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    assert g.timeout == _DEFAULT_VIGIL_TIMEOUT
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert handler.calls[0].timeout == _DEFAULT_VIGIL_TIMEOUT


async def test_configured_timeout_forwarded_to_handler():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler, timeout=30)
    expected = httpx.Timeout(30, connect=5.0)
    assert g.timeout == expected
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert handler.calls[0].timeout == expected


def test_short_timeout_caps_connect():
    g = _make_guardrail(FakeHandler([]), timeout=2)
    assert g.timeout == httpx.Timeout(2, connect=2.0)


def test_initialize_guardrail_forwards_timeout():
    lp = LitellmParams(
        guardrail="vigil_guard",
        mode="pre_call",
        api_base="https://vigil.test",
        api_key="k",
        timeout="30",
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "vg"})
    assert cb.timeout == httpx.Timeout(30, connect=5.0)


async def test_api_key_only_in_header_never_in_payload():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler, api_key="super_secret_key")
    await g.apply_guardrail(
        inputs={"texts": ["hello"]},
        request_data={"metadata": {"user_id": "u"}},
        input_type="request",
    )
    call = handler.calls[0]
    assert "super_secret_key" not in json.dumps(call.json)
    assert call.headers["Authorization"] == "Bearer super_secret_key"


@pytest.mark.parametrize("code", [429, 502, 503, 504])
async def test_retry_once_on_transient_status(code):
    handler = FakeHandler([_resp({}, status_code=code), _resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["x"]
    assert len(handler.calls) == 2


@pytest.mark.parametrize("exc", _transient_exceptions())
async def test_retry_once_on_transient_exception(exc):
    handler = FakeHandler([exc, _resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    out = await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["x"]
    assert len(handler.calls) == 2


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("boom"), RuntimeError),
        (
            httpx.WriteError("boom", request=httpx.Request("POST", _ENDPOINT)),
            GuardrailRaisedException,
        ),
    ],
)
async def test_no_retry_on_non_transient_exception(exc, expected):
    handler = FakeHandler([exc])
    g = _make_guardrail(handler)
    with pytest.raises(expected):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert len(handler.calls) == 1


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
async def test_no_retry_on_non_429_4xx(code):
    handler = FakeHandler([_resp({}, status_code=code)])
    g = _make_guardrail(handler)
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.status_code == 400
    assert len(handler.calls) == 1


async def test_fail_closed_raises_after_exhausted_retry(caplog):
    handler = FakeHandler([_resp({}, status_code=503), _resp({}, status_code=503)])
    g = _make_guardrail(handler)
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(GuardrailRaisedException) as exc_info,
    ):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.status_code == 400
    assert len(handler.calls) == 2
    assert any("fail_closed" in record.message for record in caplog.records)
    assert any("vigil-guard" in record.message for record in caplog.records)


@pytest.mark.parametrize("exc", _transient_exceptions())
async def test_fail_closed_raises_controlled_block_on_transport_error(exc, caplog):
    handler = FakeHandler([exc, exc])
    g = _make_guardrail(handler)
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(GuardrailRaisedException) as exc_info,
    ):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.guardrail_name == "vigil-guard"
    assert exc_info.value.__cause__ is exc
    assert any("fail_closed" in record.message for record in caplog.records)


async def test_fail_open_returns_inputs_unchanged_on_backend_error(caplog):
    handler = FakeHandler([_resp({}, status_code=503), _resp({}, status_code=503)])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    structured = [{"role": "user", "content": "x"}]
    inputs = {"texts": ["x"], "structured_messages": structured}
    request_data = {"metadata": {}}
    with caplog.at_level(logging.ERROR):
        out = await g.apply_guardrail(
            inputs=inputs, request_data=request_data, input_type="request"
        )
    assert out is not inputs
    assert out["texts"] == ["x"]
    assert out["structured_messages"] == structured
    assert len(handler.calls) == 2
    assert any("fail_open" in record.message for record in caplog.records)
    assert any("vigil-guard" in record.message for record in caplog.records)
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert entries[0]["guardrail_response"] == "allow"


@pytest.mark.parametrize("exc", [ssl.SSLError("tls failed"), OSError("network down")])
async def test_fail_open_returns_inputs_unchanged_on_transport_error(exc):
    handler = FakeHandler([exc])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    inputs = {"texts": ["x"]}
    out = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert out is not inputs
    assert out["texts"] == ["x"]
    assert len(handler.calls) == 1


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("bug"),
        KeyError("bug"),
        AttributeError("bug"),
    ],
)
async def test_fail_open_does_not_swallow_programming_errors(exc):
    handler = FakeHandler([exc])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    with pytest.raises(type(exc)):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert len(handler.calls) == 1


async def test_invalid_decision_fail_closed_raises(caplog):
    handler = FakeHandler([_resp({"decision": "MAYBE"})])
    g = _make_guardrail(handler)
    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(GuardrailRaisedException) as exc_info,
    ):
        await g.apply_guardrail(
            inputs={"texts": ["x"]}, request_data={}, input_type="request"
        )
    assert exc_info.value.status_code == 400
    assert "MAYBE" not in exc_info.value.message
    assert any("MAYBE" in record.message for record in caplog.records)


async def test_invalid_decision_fail_open_returns_inputs():
    handler = FakeHandler([_resp({"decision": "MAYBE"})])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    out = await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data={}, input_type="request"
    )
    assert out["texts"] == ["x"]


async def test_fail_open_multi_text_preserves_earlier_sanitization():
    handler = FakeHandler(
        [
            _resp({"decision": "SANITIZED", "sanitizedText": "[REDACTED]"}),
            _resp({}, status_code=503),
            _resp({}, status_code=503),
        ]
    )
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    request_data = {"metadata": {}}
    out = await g.apply_guardrail(
        inputs={"texts": ["my ssn is 123", "second"]},
        request_data=request_data,
        input_type="request",
    )
    assert out["texts"] == ["[REDACTED]", "second"]
    assert len(handler.calls) == 3
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert entries[0]["guardrail_response"] == "mask"


def _tool_call(arguments, name="f", tc_id="1"):
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


async def test_response_tool_call_arguments_allowed_unchanged():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"q": "weather"}')]
    out = await g.apply_guardrail(
        inputs={"texts": [], "tool_calls": tcs}, request_data={}, input_type="response"
    )
    assert handler.calls[0].json["text"] == '{"q": "weather"}'
    assert handler.calls[0].json["source"] == "model_output"
    assert out["tool_calls"] == tcs


async def test_response_tool_call_arguments_sanitized_in_place():
    handler = FakeHandler(
        [_resp({"decision": "SANITIZED", "sanitizedText": '{"email": "[EMAIL]"}'})]
    )
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"email": "john@example.com"}', name="send_mail")]
    inputs = {"texts": [], "tool_calls": tcs}
    out = await g.apply_guardrail(inputs=inputs, request_data={}, input_type="response")
    assert out["tool_calls"][0]["function"]["arguments"] == '{"email": "[EMAIL]"}'
    assert out["tool_calls"][0]["function"]["name"] == "send_mail"
    # original inputs are not mutated in place
    assert inputs["tool_calls"][0]["function"]["arguments"] == (
        '{"email": "john@example.com"}'
    )


async def test_response_tool_call_arguments_blocked_raises():
    handler = FakeHandler(
        [_resp({"decision": "BLOCKED", "blockMessage": "tool blocked"})]
    )
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"x": "bad"}')]
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": [], "tool_calls": tcs},
            request_data={},
            input_type="response",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "tool blocked"


async def test_request_tool_calls_are_not_scanned():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"x": "y"}')]
    await g.apply_guardrail(
        inputs={"texts": ["hello"], "tool_calls": tcs},
        request_data={},
        input_type="request",
    )
    assert len(handler.calls) == 1
    assert handler.calls[0].json["text"] == "hello"


async def test_tool_call_scan_backend_failure_fail_closed_raises():
    handler = FakeHandler([_resp({}, status_code=503), _resp({}, status_code=503)])
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"x": "y"}')]
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": [], "tool_calls": tcs},
            request_data={},
            input_type="response",
        )
    assert exc_info.value.status_code == 400
    assert len(handler.calls) == 2


async def test_tool_call_scan_backend_failure_fail_open_passes_through():
    handler = FakeHandler([_resp({}, status_code=503), _resp({}, status_code=503)])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    tcs = [_tool_call('{"x": "y"}')]
    out = await g.apply_guardrail(
        inputs={"texts": [], "tool_calls": tcs}, request_data={}, input_type="response"
    )
    assert out["tool_calls"] == tcs


async def test_response_tool_call_unrecognized_decision_fail_closed_raises():
    handler = FakeHandler([_resp({"decision": "MAYBE"})])
    g = _make_guardrail(handler)
    tcs = [_tool_call('{"x": "y"}')]
    with pytest.raises(GuardrailRaisedException) as exc_info:
        await g.apply_guardrail(
            inputs={"texts": [], "tool_calls": tcs},
            request_data={},
            input_type="response",
        )
    assert exc_info.value.status_code == 400


async def test_response_tool_call_unrecognized_decision_fail_open_passes_through():
    handler = FakeHandler([_resp({"decision": "MAYBE"})])
    g = _make_guardrail(handler, unreachable_fallback="fail_open")
    tcs = [_tool_call('{"x": "y"}')]
    out = await g.apply_guardrail(
        inputs={"texts": [], "tool_calls": tcs}, request_data={}, input_type="response"
    )
    assert out["tool_calls"] == tcs


async def test_metadata_allowlist_and_clamping():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    request_data = {
        "model": "gpt-4o",
        "metadata": {
            "user_id": "u1",
            "tenant_id": "t1",
            "secret_unlisted": "should_not_forward",
            "session_id": "s" * 600,
            "org_id": ["a"] * 20,
            "request_id": True,
            "conversation_id": 7,
        },
    }
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data=request_data, input_type="request"
    )
    md = handler.calls[0].json["metadata"]
    assert md["model"] == "gpt-4o"
    assert md["user_id"] == "u1"
    assert md["tenant_id"] == "t1"
    assert "secret_unlisted" not in md
    assert len(md["session_id"]) == 500
    assert len(md["org_id"]) == 10
    assert "request_id" not in md
    assert md["conversation_id"] == 7


async def test_metadata_source_precedence_and_litellm_metadata_fallback():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    request_data = {
        "user_id": "top",
        "metadata": {"user_id": "nested"},
        "litellm_metadata": {"tenant_id": "lm-tenant"},
    }
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data=request_data, input_type="request"
    )
    md = handler.calls[0].json["metadata"]
    assert md["user_id"] == "top"
    assert md["tenant_id"] == "lm-tenant"


async def test_metadata_uses_later_source_when_earlier_value_is_unclampable():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    request_data = {
        "user_id": {"drop": "dicts are not forwarded"},
        "metadata": {"user_id": "nested"},
    }
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data=request_data, input_type="request"
    )
    assert handler.calls[0].json["metadata"]["user_id"] == "nested"


async def test_metadata_array_items_are_clamped_and_filtered():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    request_data = {
        "metadata": {
            "org_id": ["z" * 600, 123, True, {"drop": 1}, None],
        },
    }
    await g.apply_guardrail(
        inputs={"texts": ["x"]}, request_data=request_data, input_type="request"
    )
    assert handler.calls[0].json["metadata"]["org_id"] == ["z" * 500, 123]


async def test_metadata_array_with_no_supported_items_is_dropped():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"metadata": {"org_id": [{"drop": 1}, None]}},
        input_type="request",
    )
    assert "org_id" not in handler.calls[0].json["metadata"]


async def test_call_id_forwarded_from_logging_obj():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    logging_obj = SimpleNamespace(litellm_call_id="call-123")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={},
        input_type="request",
        logging_obj=logging_obj,
    )
    assert handler.calls[0].json["metadata"]["litellm_call_id"] == "call-123"


async def test_call_id_forwarded_from_request_data():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"litellm_call_id": "rd-1"},
        input_type="request",
        logging_obj=None,
    )
    assert handler.calls[0].json["metadata"]["litellm_call_id"] == "rd-1"


async def test_call_id_forwarded_from_request_metadata():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"metadata": {"litellm_call_id": "md-1"}},
        input_type="request",
        logging_obj=None,
    )
    assert handler.calls[0].json["metadata"]["litellm_call_id"] == "md-1"


async def test_call_id_logging_obj_takes_precedence():
    handler = FakeHandler([_resp({"decision": "ALLOWED"})])
    g = _make_guardrail(handler)
    logging_obj = SimpleNamespace(litellm_call_id="log-1")
    await g.apply_guardrail(
        inputs={"texts": ["x"]},
        request_data={"litellm_call_id": "rd-1"},
        input_type="request",
        logging_obj=logging_obj,
    )
    assert handler.calls[0].json["metadata"]["litellm_call_id"] == "log-1"


def test_enum_value():
    assert SupportedGuardrailIntegrations.VIGIL_GUARD.value == "vigil_guard"


def test_config_model_ui_name_and_instantiation():
    assert VigilGuardGuardrailConfigModel.ui_friendly_name() == "Vigil Guard"
    model = VigilGuardGuardrailConfigModel(api_base="https://x", api_key="k")
    assert model.api_base == "https://x"


def test_get_config_model_returns_config_model():
    g = _make_guardrail(FakeHandler([]))
    assert g.get_config_model() is VigilGuardGuardrailConfigModel


def test_registries_expose_initializer_and_class():
    assert "vigil_guard" in guardrail_initializer_registry
    assert guardrail_class_registry["vigil_guard"] is VigilGuardGuardrail


def test_litellm_params_includes_config_model():
    assert VigilGuardGuardrailConfigModel in LitellmParams.__mro__


def test_config_driven_initialization_creates_callback():
    lp = LitellmParams(
        guardrail="vigil_guard",
        mode="pre_call",
        api_base="https://vigil.test",
        api_key="k",
    )
    cb = initialize_guardrail(lp, {"guardrail_name": "vg"})
    assert isinstance(cb, VigilGuardGuardrail)
    assert cb.unreachable_fallback == "fail_closed"
