"""
Unit tests for the TypeSafe (Jev) compaction guardrail.

Tests cover:
- exchanges scored below relevance_threshold have their tool rows blanked while
  assistant tool-call rows and kept exchanges pass through verbatim, without
  mutating the caller's message list
- protected rows (system, last user, and the last tool exchange via the
  last-assistant rule) are never sent to Jev even when long
- exchanges under min_chars_to_evaluate are skipped
- request shape: POST {api_base}/v1/systemone with Bearer auth, one noul
  question per candidate keyed e<i>, task = last user text, results truncated
  to max_result_chars_in_state
- identity return when there are no candidates or nothing is dropped
- fail_open forwards uncompacted on service failure; fail_closed raises
- response input_type passthrough and initialize_guardrail wiring
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.typesafe import (
    TypeSafeGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.typesafe.typesafe import DROPPED_RESULT_TEXT
from litellm.types.guardrails import SupportedGuardrailIntegrations
from litellm.types.utils import GenericGuardrailAPIInputs

FAKE_API_BASE = "https://typesafe.example.com"
FAKE_API_KEY = "ts_test-key"

SYSTEM_TEXT = "You are a research assistant."
USER_TEXT = "Which 2026 EV has the longest range?"
TOOL_OUTPUT_LONG = "Result: EV range comparison. " * 40
TOOL_OUTPUT_SHORT = "short"


def _exchange(call_id: str, tool_text: str, name: str = "web_search") -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": '{"query": "ev"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": name, "content": tool_text},
    ]


def _messages(*, tail: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    base = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": USER_TEXT},
    ]
    return base + (tail or [])


def _make_guardrail(
    handler: MagicMock | None = None,
    *,
    max_result_chars_in_state: int | None = None,
    unreachable_fallback: str | None = None,
) -> TypeSafeGuardrail:
    return TypeSafeGuardrail(
        api_base=FAKE_API_BASE,
        api_key=FAKE_API_KEY,
        guardrail_name="typesafe",
        default_on=True,
        async_handler=handler or _make_handler({"e0": 0.9}),
        max_result_chars_in_state=max_result_chars_in_state,
        unreachable_fallback=unreachable_fallback,
    )


def _make_handler(answers: dict[str, float], status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {qid: {"type": "noul", "noul": score} for qid, score in answers.items()},
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    response.text = ""
    handler = MagicMock()
    handler.post = AsyncMock(return_value=response)
    return handler


def _inputs(messages: list[dict[str, object]]) -> GenericGuardrailAPIInputs:
    return GenericGuardrailAPIInputs(structured_messages=messages)


async def _apply(
    guardrail: TypeSafeGuardrail, messages: list[dict[str, object]], input_type: str = "request"
) -> GenericGuardrailAPIInputs:
    return await guardrail.apply_guardrail(
        inputs=_inputs(messages),
        request_data={},
        input_type=input_type,  # pyright: ignore[reportArgumentType]  # test uses the same literal domain
        logging_obj=None,
    )


@pytest.mark.asyncio
async def test_low_noul_exchange_blanked_high_kept_and_input_not_mutated():
    handler = _make_handler({"e0": 0.1, "e1": 0.95})
    guardrail = _make_guardrail(handler)
    messages = _messages(
        tail=[
            *_exchange("call_1", TOOL_OUTPUT_LONG),
            *_exchange("call_2", TOOL_OUTPUT_LONG),
            {"role": "assistant", "content": "still thinking"},
        ]
    )
    snapshot = [dict(m) for m in messages]

    result = await _apply(guardrail, messages)
    out = result["structured_messages"]

    assert out[3]["content"] == DROPPED_RESULT_TEXT
    assert out[3]["tool_call_id"] == "call_1"
    assert out[3]["role"] == "tool"
    assert out[5]["content"] == TOOL_OUTPUT_LONG
    assert out[2] == messages[2]
    assert out[4] == messages[4]
    assert out[6]["content"] == "still thinking"
    assert messages == snapshot


@pytest.mark.asyncio
async def test_last_exchange_and_protected_rows_never_evaluated():
    handler = _make_handler({"e0": 0.05})
    guardrail = _make_guardrail(handler)
    messages = _messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), *_exchange("call_2", TOOL_OUTPUT_LONG)])

    result = await _apply(guardrail, messages)

    payload = handler.post.call_args.kwargs["json"]
    assert list(payload["questions"]) == ["e0"]
    assert list(payload["state"]["tool_exchanges"]) == ["e0"]
    assert payload["state"]["task"] == USER_TEXT
    assert payload["state"]["system"] == SYSTEM_TEXT
    out = result["structured_messages"]
    assert out[3]["content"] == DROPPED_RESULT_TEXT
    assert out[5]["content"] == TOOL_OUTPUT_LONG


@pytest.mark.asyncio
async def test_short_exchange_not_sent():
    handler = _make_handler({"e0": 0.9})
    guardrail = _make_guardrail(handler)
    messages = _messages(
        tail=[
            *_exchange("call_1", TOOL_OUTPUT_SHORT),
            *_exchange("call_2", TOOL_OUTPUT_LONG),
            {"role": "assistant", "content": "done"},
        ]
    )
    result = await _apply(guardrail, messages)
    payload = handler.post.call_args.kwargs["json"]
    assert list(payload["questions"]) == ["e0"]
    exchange = payload["state"]["tool_exchanges"]["e0"]
    assert exchange["result"] == TOOL_OUTPUT_LONG
    assert result is not None


@pytest.mark.asyncio
async def test_request_body_shape_and_truncation():
    handler = _make_handler({"e0": 0.9})
    guardrail = _make_guardrail(handler, max_result_chars_in_state=50)
    messages = _messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "done"}])
    await _apply(guardrail, messages)

    kwargs = handler.post.call_args.kwargs
    assert kwargs["url"].endswith("/v1/systemone")
    assert kwargs["url"].startswith(FAKE_API_BASE)
    assert kwargs["headers"]["Authorization"] == f"Bearer {FAKE_API_KEY}"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    payload = kwargs["json"]
    assert payload["model"] == "jev-latest"
    assert list(payload["questions"]) == ["e0"]
    assert payload["questions"]["e0"]["type"] == "noul"
    assert "e0" in payload["questions"]["e0"]["instructions"]
    assert payload["state"]["task"] == USER_TEXT
    exchange = payload["state"]["tool_exchanges"]["e0"]
    assert len(exchange["result"]) == 50
    assert exchange["result"].startswith(TOOL_OUTPUT_LONG[:10])
    assert exchange["result"].endswith(TOOL_OUTPUT_LONG[-11:])
    assert list(exchange["tool_calls"]) == [{"name": "web_search", "arguments": '{"query": "ev"}'}]


@pytest.mark.asyncio
async def test_no_candidates_returns_identity_and_skips_http():
    handler = _make_handler({})
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[{"role": "assistant", "content": "plain answer"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs
    handler.post.assert_not_called()


@pytest.mark.asyncio
async def test_all_above_threshold_returns_identity():
    handler = _make_handler({"e0": 0.9})
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_fail_open_returns_inputs_on_exception():
    handler = MagicMock()
    handler.post = AsyncMock(side_effect=Exception("connection refused"))
    guardrail = _make_guardrail(handler, unreachable_fallback="fail_open")
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_fail_closed_raises_http_exception():
    handler = MagicMock()
    handler.post = AsyncMock(side_effect=Exception("connection refused"))
    guardrail = _make_guardrail(handler, unreachable_fallback="fail_closed")
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_fail_open_on_non_2xx():
    handler = _make_handler({"e0": 0.9}, status=500)
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_response_input_type_passthrough():
    handler = _make_handler({"e0": 0.05})
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG)]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response", logging_obj=None)
    assert result is inputs
    handler.post.assert_not_called()


def test_initialize_guardrail_applies_optional_params_and_registry_keys():
    from litellm.types.guardrails import LitellmParams

    litellm_params = LitellmParams(
        guardrail="typesafe",
        mode="pre_call",
        api_key=FAKE_API_KEY,
        api_base=FAKE_API_BASE,
        optional_params={
            "relevance_threshold": 0.5,
            "min_chars_to_evaluate": 10,
            "max_result_chars_in_state": 100,
        },
    )
    callback = initialize_guardrail(litellm_params, {"guardrail_name": "jev-compaction"})
    assert isinstance(callback, TypeSafeGuardrail)
    assert callback.relevance_threshold == 0.5
    assert callback.min_chars_to_evaluate == 10
    assert callback.max_result_chars_in_state == 100
    assert callback.unreachable_fallback == "fail_open"
    assert guardrail_initializer_registry[SupportedGuardrailIntegrations.TYPESAFE.value] is initialize_guardrail
    assert guardrail_class_registry[SupportedGuardrailIntegrations.TYPESAFE.value] is TypeSafeGuardrail


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires an API key"):
        TypeSafeGuardrail(api_key=None)


def test_get_config_model_and_ui_name():
    from litellm.types.proxy.guardrails.guardrail_hooks.typesafe import (
        TypeSafeGuardrailConfigModel,
    )

    assert TypeSafeGuardrail.get_config_model() is TypeSafeGuardrailConfigModel
    assert TypeSafeGuardrailConfigModel.ui_friendly_name() == "TypeSafe (Jev) Compaction"


@pytest.mark.asyncio
async def test_non_list_and_non_dict_messages_return_identity():
    guardrail = _make_guardrail()
    not_a_list = GenericGuardrailAPIInputs(structured_messages={"role": "user"})
    assert (
        await guardrail.apply_guardrail(inputs=not_a_list, request_data={}, input_type="request", logging_obj=None)
        is not_a_list
    )
    with_bad_row = _inputs(_messages(tail=[["not", "a", "dict"]]))
    assert (
        await guardrail.apply_guardrail(inputs=with_bad_row, request_data={}, input_type="request", logging_obj=None)
        is with_bad_row
    )


def test_odd_tool_call_shapes_yield_no_entries():
    from litellm.proxy.guardrails.guardrail_hooks.typesafe.typesafe import _tool_call_entries

    assert _tool_call_entries({"tool_calls": "not-a-list"}) == ()
    assert _tool_call_entries({"tool_calls": None}) == ()
    assert list(_tool_call_entries({"tool_calls": [42]})) == []
    entries = _tool_call_entries({"tool_calls": [{"function": {"name": "web_search", "arguments": "{}"}}]})
    assert list(entries) == [{"name": "web_search", "arguments": "{}"}]


@pytest.mark.asyncio
async def test_short_max_chars_uses_prefix_slice():
    handler = _make_handler({"e0": 0.9})
    guardrail = _make_guardrail(handler, max_result_chars_in_state=5)
    await _apply(
        guardrail, _messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}])
    )
    result = handler.post.call_args.kwargs["json"]["state"]["tool_exchanges"]["e0"]["result"]
    assert result == TOOL_OUTPUT_LONG[:5]


@pytest.mark.asyncio
async def test_unreadable_json_body_fails_open():
    handler = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.text = "not json"
    response.json.side_effect = ValueError("no json")
    handler.post = AsyncMock(return_value=response)
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_malformed_answers_shape_fails_open():
    handler = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.text = '{"answers": "oops"}'
    response.json.return_value = {"answers": "oops"}
    handler.post = AsyncMock(return_value=response)
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_http_status_error_includes_status_and_undecodable_body():
    import httpx

    response = MagicMock()
    response.status_code = 503
    type(response).text = PropertyMock(side_effect=httpx.DecodingError("bad codec"))
    handler = MagicMock()
    handler.post = AsyncMock(side_effect=httpx.HTTPStatusError("unavailable", request=MagicMock(), response=response))
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)
    assert result is inputs


@pytest.mark.asyncio
async def test_cancelled_jev_call_propagates():
    import asyncio

    handler = MagicMock()
    handler.post = AsyncMock(side_effect=asyncio.CancelledError())
    guardrail = _make_guardrail(handler)
    inputs = _inputs(_messages(tail=[*_exchange("call_1", TOOL_OUTPUT_LONG), {"role": "assistant", "content": "x"}]))
    with pytest.raises(asyncio.CancelledError):
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request", logging_obj=None)


def test_optional_params_defaults_and_event_hook_coercion():
    from litellm.proxy.guardrails.guardrail_hooks.typesafe import _coerce_event_hook, _optional_params
    from litellm.types.guardrails import GuardrailEventHooks, LitellmParams

    assert _coerce_event_hook("pre_call") is GuardrailEventHooks.pre_call
    assert _coerce_event_hook(["pre_call", "post_call"]) == [
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.post_call,
    ]
    litellm_params = LitellmParams(guardrail="typesafe", mode="pre_call", api_key=FAKE_API_KEY)
    params = _optional_params(litellm_params)
    assert params.relevance_threshold is None


def test_typesafe_initializer_discoverable_via_hook_registries():
    from litellm.proxy.guardrails.guardrail_registry import get_guardrail_initializer_from_hooks

    initializers = get_guardrail_initializer_from_hooks()
    assert initializers["typesafe"] is initialize_guardrail
