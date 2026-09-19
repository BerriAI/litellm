"""Pin ProxyLogging streaming + response-headers helpers.

Covers ``_wrap_streaming_iterator_with_enrichment``,
``async_post_call_streaming_hook``,
``async_post_call_streaming_iterator_hook``, ``_fire_deferred_stream_logging``,
``is_a2a_streaming_response``, ``_init_response_taking_too_long_task``,
``post_call_response_headers_hook``, ``_build_litellm_call_info``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    BaseAnthropicMessagesStreamingIterator,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import Choices, Message, ModelResponseStream, Usage


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# is_a2a_streaming_response
# ---------------------------------------------------------------------------


def test_is_a2a_streaming_response_truth_matrix(proxy_logging):
    snapshot = {
        "all_three_keys_present": proxy_logging.is_a2a_streaming_response(
            {"jsonrpc": "2.0", "id": "1", "result": {"x": 1}, "extra": "y"}
        ),
        "missing_result": proxy_logging.is_a2a_streaming_response(
            {"jsonrpc": "2.0", "id": "1"}
        ),
        "missing_jsonrpc": proxy_logging.is_a2a_streaming_response(
            {"id": "1", "result": {}}
        ),
        "empty_dict": proxy_logging.is_a2a_streaming_response({}),
    }
    assert snapshot == {
        "all_three_keys_present": True,
        "missing_result": False,
        "missing_jsonrpc": False,
        "empty_dict": False,
    }


def test_is_a2a_streaming_response_invalid_input_raises(proxy_logging):
    with pytest.raises(TypeError):
        proxy_logging.is_a2a_streaming_response(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _build_litellm_call_info
# ---------------------------------------------------------------------------


def test_build_litellm_call_info_pulls_from_hidden_params_and_metadata(proxy_logging):
    response = MagicMock()
    response._hidden_params = {
        "custom_llm_provider": "openai",
        "api_base": "https://api.openai.com",
        "model_id": "model-1",
    }
    info = proxy_logging._build_litellm_call_info(
        data={"metadata": {"model_info": {"name": "gpt-4o-mini"}}},
        response=response,
    )
    assert info == {
        "custom_llm_provider": "openai",
        "model_info": {"name": "gpt-4o-mini"},
        "api_base": "https://api.openai.com",
        "model_id": "model-1",
    }


def test_build_litellm_call_info_fallbacks_to_litellm_metadata(proxy_logging):
    response = MagicMock()
    response._hidden_params = {"custom_llm_provider": "azure"}
    info = proxy_logging._build_litellm_call_info(
        data={"litellm_metadata": {"model_info": {"alias": "azure-gpt"}}},
        response=response,
    )
    snapshot = {
        "custom_llm_provider": info["custom_llm_provider"],
        "model_info": info["model_info"],
        "api_base": info["api_base"],
        "model_id": info["model_id"],
    }
    assert snapshot == {
        "custom_llm_provider": "azure",
        "model_info": {"alias": "azure-gpt"},
        "api_base": None,
        "model_id": None,
    }


def test_build_litellm_call_info_invalid_data_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._build_litellm_call_info(data=None, response=MagicMock())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _init_response_taking_too_long_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_response_taking_too_long_task_runs_when_alerting(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.slack_alerting_instance.alerting = ["slack"]
    captured: Dict[str, Any] = {}

    async def fake_resp_too_long(request_data):
        captured["request_data"] = request_data

    proxy_logging.slack_alerting_instance.response_taking_too_long = fake_resp_too_long
    payload = {"req": "y", "litellm_call_id": "c1", "model": "m"}
    proxy_logging._init_response_taking_too_long_task(data=payload)
    await asyncio.sleep(0)
    snapshot = {
        "received_payload": captured["request_data"],
        "fired_once": len(captured) == 1,
        "alerting_was_truthy": bool(proxy_logging.slack_alerting_instance.alerting),
    }
    assert snapshot == {
        "received_payload": payload,
        "fired_once": True,
        "alerting_was_truthy": True,
    }


@pytest.mark.asyncio
async def test_init_response_taking_too_long_task_no_op_when_alerting_off(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.slack_alerting_instance.alerting = None
    proxy_logging.slack_alerting_instance.response_taking_too_long = AsyncMock()
    proxy_logging._init_response_taking_too_long_task(data=None)
    await asyncio.sleep(0)
    proxy_logging.slack_alerting_instance.response_taking_too_long.assert_not_called()


def test_init_response_taking_too_long_task_no_slack_instance_no_error_raises(proxy_logging):
    proxy_logging.slack_alerting_instance = None
    proxy_logging._init_response_taking_too_long_task(data=None)


# ---------------------------------------------------------------------------
# _wrap_streaming_iterator_with_enrichment
# ---------------------------------------------------------------------------


async def _passthrough_hook(*, response: AsyncIterator[object]) -> AsyncGenerator[object, None]:
    async for chunk in response:
        yield chunk


async def _one_chunk() -> AsyncGenerator[object, None]:
    yield "chunk"


@pytest.mark.asyncio
async def test_wrap_streaming_iterator_with_enrichment_passes_through_chunks(proxy_logging):
    async def gen():
        for ch in ("a", "b", "c"):
            yield ch

    cb = MagicMock(guardrail_name="g", event_hook="pre_call")
    wrapped = proxy_logging._wrap_streaming_iterator_with_enrichment(
        callback=cb, response=gen(), hook=_passthrough_hook, request_data={}
    )
    out = [ch async for ch in wrapped]
    snapshot = {
        "chunks": out,
        "count": len(out),
        "first": out[0],
        "last": out[-1],
    }
    assert snapshot == {
        "chunks": ["a", "b", "c"],
        "count": 3,
        "first": "a",
        "last": "c",
    }


@pytest.mark.asyncio
async def test_wrap_streaming_iterator_with_enrichment_enriches_http_exception_raises(proxy_logging):
    detail = {"error": "blocked"}

    async def boom_hook(*, response: AsyncIterator[object]) -> AsyncGenerator[object, None]:
        if False:
            yield  # pragma: no cover
        raise HTTPException(status_code=400, detail=detail)

    cb = MagicMock(guardrail_name="presidio", event_hook="post_call")
    request_data: dict[str, object] = {}
    wrapped = proxy_logging._wrap_streaming_iterator_with_enrichment(
        callback=cb, response=_one_chunk(), hook=boom_hook, request_data=request_data
    )
    with pytest.raises(HTTPException):
        async for _ in wrapped:
            pass
    assert detail["guardrail_name"] == "presidio"
    assert detail["guardrail_mode"] == "post_call"
    assert request_data["metadata"]["applied_guardrails"] == ["presidio"]


@pytest.mark.asyncio
async def test_wrap_streaming_iterator_leaves_upstream_http_exception_unattributed(proxy_logging):
    detail = {"error": "upstream rejected the stream"}

    async def failing_upstream() -> AsyncGenerator[object, None]:
        if False:
            yield  # pragma: no cover
        raise HTTPException(status_code=502, detail=detail)

    cb = MagicMock(guardrail_name="presidio", event_hook="post_call")
    request_data: dict[str, object] = {}
    wrapped = proxy_logging._wrap_streaming_iterator_with_enrichment(
        callback=cb, response=failing_upstream(), hook=_passthrough_hook, request_data=request_data
    )
    with pytest.raises(HTTPException):
        async for _ in wrapped:
            pass
    assert detail == {"error": "upstream rejected the stream"}
    assert request_data == {}


# ---------------------------------------------------------------------------
# async_post_call_streaming_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_post_call_streaming_hook_fast_path_returns_response(proxy_logging, mock_callbacks_disabled, make_user_api_key_auth):
    resp = "chunk-1"
    out = await proxy_logging.async_post_call_streaming_hook(
        data={}, response=resp, user_api_key_dict=make_user_api_key_auth()
    )
    snapshot = {
        "out_is_input": out is resp,
        "out_value": out,
        "type": type(out).__name__,
        "callbacks_empty": len(litellm.callbacks) == 0,
    }
    assert snapshot == {
        "out_is_input": True,
        "out_value": "chunk-1",
        "type": "str",
        "callbacks_empty": True,
    }


@pytest.mark.asyncio
async def test_async_post_call_streaming_hook_invokes_per_chunk_callback(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _Per(CustomLogger):
        async def async_post_call_streaming_hook(self, **kwargs):  # type: ignore[override]
            return "modified-" + str(kwargs.get("response", ""))

    cb = _Per()
    monkeypatch.setattr(litellm, "callbacks", [cb])

    from litellm import ModelResponse

    fake_resp = ModelResponse(
        id="rid",
        choices=[{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )
    out = await proxy_logging.async_post_call_streaming_hook(
        data={},
        response=fake_resp,
        user_api_key_dict=make_user_api_key_auth(),
    )
    assert isinstance(out, str)
    assert out.startswith("modified-")


@pytest.mark.asyncio
async def test_async_post_call_streaming_hook_callback_error_raises(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _Per(CustomLogger):
        async def async_post_call_streaming_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("hook-fail")

    monkeypatch.setattr(litellm, "callbacks", [_Per()])

    from litellm import ModelResponse

    fake_resp = ModelResponse(
        id="rid",
        choices=[{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}],
        created=0,
        model="gpt-4o-mini",
        object="chat.completion.chunk",
    )
    with pytest.raises(RuntimeError):
        await proxy_logging.async_post_call_streaming_hook(
            data={},
            response=fake_resp,
            user_api_key_dict=make_user_api_key_auth(),
        )


# ---------------------------------------------------------------------------
# async_post_call_streaming_iterator_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_post_call_streaming_iterator_hook_no_overrides_passes_through(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    async def gen():
        for ch in ("a", "b"):
            yield ch

    chunks = []
    async for ch in proxy_logging.async_post_call_streaming_iterator_hook(
        response=gen(),
        user_api_key_dict=make_user_api_key_auth(),
        request_data={},
    ):
        chunks.append(ch)
    snapshot = {
        "chunks": chunks,
        "count": len(chunks),
        "passthrough_preserved_order": chunks == ["a", "b"],
    }
    assert snapshot == {
        "chunks": ["a", "b"],
        "count": 2,
        "passthrough_preserved_order": True,
    }


@pytest.mark.asyncio
async def test_async_post_call_streaming_iterator_hook_with_override_chains_callback(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _IterOverride(CustomLogger):
        async def async_post_call_streaming_iterator_hook(self, **kwargs):  # type: ignore[override]
            async for ch in kwargs["response"]:
                yield ch + "*"

    monkeypatch.setattr(litellm, "callbacks", [_IterOverride()])

    async def gen():
        for ch in ("a", "b"):
            yield ch

    out: List[str] = []
    async for ch in proxy_logging.async_post_call_streaming_iterator_hook(
        response=gen(),
        user_api_key_dict=make_user_api_key_auth(),
        request_data={},
    ):
        out.append(ch)
    assert out == ["a*", "b*"]


@pytest.mark.asyncio
async def test_async_post_call_streaming_iterator_hook_presidio_output_masking_keeps_split_pii_buffered(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    Regression test for #41611: output PII split across SSE chunks must be
    evaluated by Presidio's native streaming iterator as a reconstructed
    response, rather than by the unified per-chunk guardrail path.
    """
    guardrail = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        apply_to_output=True,
    )

    analyzed_texts: list[str] = []

    async def mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        analyzed_texts.append(text)
        return text.replace("user@example.com", "<EMAIL_ADDRESS>")

    guardrail.check_pii = mock_check_pii

    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    chunks = [
        ModelResponseStream(
            id="chatcmpl-proxy-split-1",
            choices=[
                Choices(
                    index=0,
                    delta=Message(content="Please contact user@exa"),
                )
            ],
            created=1,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
        ModelResponseStream(
            id="chatcmpl-proxy-split-2",
            choices=[
                Choices(
                    index=0,
                    delta=Message(content="mple.com for support."),
                )
            ],
            created=2,
            model="gpt-4o-mini",
            object="chat.completion.chunk",
        ),
    ]

    async def upstream():
        for chunk in chunks:
            yield chunk

    received = [
        chunk
        async for chunk in proxy_logging.async_post_call_streaming_iterator_hook(
            response=upstream(),
            user_api_key_dict=make_user_api_key_auth(),
            request_data={},
        )
    ]

    reconstructed = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in received
        if isinstance(chunk, ModelResponseStream)
    )

    assert analyzed_texts == ["Please contact user@example.com for support."]
    assert "<EMAIL_ADDRESS>" in reconstructed
    assert "user@example.com" not in reconstructed


@pytest.mark.asyncio
async def test_async_post_call_streaming_iterator_hook_upstream_error_raises(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    async def gen():
        if False:
            yield  # pragma: no cover
        raise RuntimeError("upstream")

    with pytest.raises(RuntimeError):
        async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
            response=gen(),
            user_api_key_dict=make_user_api_key_auth(),
            request_data={},
        ):
            pass


# ---------------------------------------------------------------------------
# deferred native /v1/messages stream logging (LIT-6409)
# ---------------------------------------------------------------------------


_NATIVE_MESSAGES_STREAM_EVENTS = (
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 3, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    {"type": "message_stop"},
)


def _armed_native_messages_stream(test_name: str, request_data: Dict[str, Any], events: List[Any]):
    """The proxy-side setup for a native /v1/messages stream with post_call
    guardrails active: a real BaseAnthropicMessagesStreamingIterator whose
    logging_obj carries the deferred-dispatch callback the proxy arms in
    common_request_processing. The callback records what the guardrail
    metadata contained at the moment the deferred logging was dispatched."""
    logging_obj = LiteLLMLoggingObj(
        model="bedrock/invoke/anthropic.claude-sonnet-4-20250514-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.now(),
        litellm_call_id=test_name,
        function_id=test_name,
    )

    async def _dispatch_deferred_logging(logging_coroutine):
        events.append(
            (
                "logging_dispatched",
                "post_call_entry_visible",
                bool(request_data.get("metadata", {}).get("standard_logging_guardrail_information")),
            )
        )
        logging_coroutine.close()

    logging_obj._on_deferred_stream_complete = _dispatch_deferred_logging
    request_data["litellm_logging_obj"] = logging_obj

    iterator = BaseAnthropicMessagesStreamingIterator(litellm_logging_obj=logging_obj, request_body={})

    async def _upstream():
        for event in _NATIVE_MESSAGES_STREAM_EVENTS:
            yield event

    return logging_obj, iterator.async_sse_wrapper(_upstream())


@pytest.mark.asyncio
async def test_native_messages_stream_logging_fires_after_guardrail_end_of_stream_scan(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    Regression test for LIT-6409: on native /v1/messages streams the
    end-of-stream guardrail scan writes its post_call entry AFTER the
    upstream iterator is exhausted, so success logging dispatched at
    upstream exhaustion never sees it. The deferred dispatch must fire
    only after the guardrail chain fully drains.
    """
    events: List[Any] = []
    request_data: Dict[str, Any] = {"metadata": {}}
    _, native_stream = _armed_native_messages_stream(
        "test_native_stream_deferred_ordering", request_data, events
    )

    class _EndOfStreamScanGuardrail(CustomLogger):
        async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
            async for chunk in response:
                yield chunk
            request_data.setdefault("metadata", {})["standard_logging_guardrail_information"] = [
                {"guardrail_mode": "post_call", "guardrail_status": "success"}
            ]
            events.append("scan_appended")

    monkeypatch.setattr(litellm, "callbacks", [_EndOfStreamScanGuardrail()])

    async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
        response=native_stream,
        user_api_key_dict=make_user_api_key_auth(),
        request_data=request_data,
    ):
        pass
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert events == ["scan_appended", ("logging_dispatched", "post_call_entry_visible", True)]


@pytest.mark.asyncio
async def test_native_messages_stream_logging_fires_when_guardrail_blocks_after_stream_end(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    A guardrail block raised after upstream exhaustion (unified_guardrail
    re-raises HTTPException for blocked content) must still flush the
    parked deferred logging, or the blocked stream loses its spend log.
    """
    events: List[Any] = []
    request_data: Dict[str, Any] = {"metadata": {}}
    logging_obj, native_stream = _armed_native_messages_stream(
        "test_native_stream_deferred_block", request_data, events
    )

    class _BlockingGuardrail(CustomLogger):
        async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
            async for chunk in response:
                yield chunk
            raise HTTPException(status_code=400, detail={"error": "Violated guardrail policy"})

    monkeypatch.setattr(litellm, "callbacks", [_BlockingGuardrail()])

    with pytest.raises(HTTPException):
        async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
            response=native_stream,
            user_api_key_dict=make_user_api_key_auth(),
            request_data=request_data,
        ):
            pass
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert [event[0] for event in events] == ["logging_dispatched"]
    assert logging_obj._deferred_stream_complete_args is None


def _armed_chat_stream(
    test_name: str, request_data: dict[str, object], events: list[str]
) -> tuple[LiteLLMLoggingObj, AsyncIterator[dict[str, object]]]:
    """A /chat/completions stream whose CSW shape parks ``(assembled ModelResponse, cache_hit)``
    at upstream exhaustion, with the deferred dispatch recording into ``events``."""
    logging_obj = LiteLLMLoggingObj(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id=test_name,
        function_id=test_name,
    )
    logging_obj.optional_params = {}
    logging_obj.litellm_params = {}
    logging_obj.standard_built_in_tools_params = None

    async def _dispatch_deferred_logging(*args: object) -> None:
        events.append("success_dispatched")

    logging_obj._on_deferred_stream_complete = _dispatch_deferred_logging
    request_data["litellm_logging_obj"] = logging_obj

    assembled = litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "BANANA"}}],
        usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )

    async def _upstream() -> AsyncIterator[dict[str, object]]:
        yield {"id": "c1", "choices": [{"index": 0, "delta": {"content": "BAN"}}]}
        yield {"id": "c1", "choices": [{"index": 0, "delta": {"content": "ANA"}}]}
        logging_obj._deferred_stream_complete_args = (assembled, False)

    return logging_obj, _upstream()


def _raising_at_end_of_stream(error: Exception) -> CustomLogger:
    class _EndOfStreamRaiser(CustomLogger):
        async def async_post_call_streaming_iterator_hook(
            self, user_api_key_dict: UserAPIKeyAuth, response: AsyncIterator[object], request_data: dict[str, object]
        ) -> AsyncGenerator[object, None]:
            async for chunk in response:
                yield chunk
            raise error

    return _EndOfStreamRaiser()


@pytest.mark.asyncio
async def test_chat_stream_guardrail_block_after_stream_end_logs_failure_not_success(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    A guardrail that raises ``GuardrailRaisedException`` at end of a
    /chat/completions stream must NOT dispatch the parked success logging:
    the request is logged via the failure path instead, with the consumed
    usage carried over so the failure row bills correctly.
    """
    events: list[str] = []
    request_data: dict[str, object] = {"metadata": {}}
    logging_obj, upstream = _armed_chat_stream("test_chat_stream_guardrail_block", request_data, events)
    monkeypatch.setattr(
        litellm,
        "callbacks",
        [_raising_at_end_of_stream(GuardrailRaisedException(guardrail_name="g", message="blocked"))],
    )

    with pytest.raises(GuardrailRaisedException):
        async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
            response=upstream,
            user_api_key_dict=make_user_api_key_auth(),
            request_data=request_data,
        ):
            pass
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    snapshot = {
        "events": events,
        "callback_cleared": logging_obj._on_deferred_stream_complete is None,
        "args_cleared": logging_obj._deferred_stream_complete_args is None,
        "combined_usage_total_tokens": logging_obj.model_call_details["combined_usage_object"].total_tokens,
        "response_cost_positive": logging_obj.model_call_details["response_cost"] > 0,
    }
    assert snapshot == {
        "events": [],
        "callback_cleared": True,
        "args_cleared": True,
        "combined_usage_total_tokens": 8,
        "response_cost_positive": True,
    }


@pytest.mark.asyncio
async def test_chat_stream_generic_callback_error_after_stream_end_still_flushes_success_logging(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    ``post_call_failure_hook`` only routes proxy-level errors (HTTPException,
    ProxyException, GuardrailRaisedException) through failure logging. A
    callback that dies with any other exception after the stream completed
    must keep flushing the parked success dispatch, or the request ends with
    no terminal log at all.
    """
    events: list[str] = []
    request_data: dict[str, object] = {"metadata": {}}
    logging_obj, upstream = _armed_chat_stream("test_chat_stream_generic_callback_error", request_data, events)
    monkeypatch.setattr(litellm, "callbacks", [_raising_at_end_of_stream(RuntimeError("callback crashed"))])

    with pytest.raises(RuntimeError):
        async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
            response=upstream,
            user_api_key_dict=make_user_api_key_auth(),
            request_data=request_data,
        ):
            pass
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    snapshot = {
        "events": events,
        "args_cleared": logging_obj._deferred_stream_complete_args is None,
        "failure_usage_recorded": "combined_usage_object" in logging_obj.model_call_details,
    }
    assert snapshot == {"events": ["success_dispatched"], "args_cleared": True, "failure_usage_recorded": False}


# ---------------------------------------------------------------------------
# _fire_deferred_stream_logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fire_deferred_stream_logging_fires_callback():
    logging_obj = MagicMock()
    captured: Dict[str, Any] = {}

    async def deferred(arg):
        captured["arg"] = arg

    logging_obj._on_deferred_stream_complete = deferred
    logging_obj._deferred_stream_complete_args = ("payload",)

    ProxyLogging._fire_deferred_stream_logging(request_data={"litellm_logging_obj": logging_obj})
    await asyncio.sleep(0)
    snapshot = {
        "arg": captured["arg"],
        "callback_cleared": logging_obj._on_deferred_stream_complete is None,
        "args_cleared": logging_obj._deferred_stream_complete_args is None,
    }
    assert snapshot == {"arg": "payload", "callback_cleared": True, "args_cleared": True}


def test_fire_deferred_stream_logging_no_logging_obj_no_error():
    ProxyLogging._fire_deferred_stream_logging(request_data={})


def test_fire_deferred_stream_logging_missing_obj_raises_on_invalid_dict():
    with pytest.raises(AttributeError):
        ProxyLogging._fire_deferred_stream_logging(request_data=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# post_call_response_headers_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_response_headers_hook_returns_empty_when_no_callbacks(
    proxy_logging, mock_callbacks_disabled, make_user_api_key_auth
):
    out = await proxy_logging.post_call_response_headers_hook(
        data={}, user_api_key_dict=make_user_api_key_auth(), response=MagicMock(_hidden_params={})
    )
    assert out == {}


@pytest.mark.asyncio
async def test_post_call_response_headers_hook_merges_callback_headers(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _Cb(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):  # type: ignore[override]
            return {"X-One": "1", "X-Two": "2", "X-Common": "first"}

    class _Cb2(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):  # type: ignore[override]
            return {"X-Common": "second", "X-Three": "3"}

    monkeypatch.setattr(litellm, "callbacks", [_Cb(), _Cb2()])
    response = MagicMock()
    response._hidden_params = {}
    out = await proxy_logging.post_call_response_headers_hook(
        data={}, user_api_key_dict=make_user_api_key_auth(), response=response
    )
    assert out == {"X-One": "1", "X-Two": "2", "X-Common": "second", "X-Three": "3"}


@pytest.mark.asyncio
async def test_post_call_response_headers_hook_swallows_callback_error(proxy_logging, make_user_api_key_auth, monkeypatch):
    """Errors inside the hook are caught — function returns merged so-far."""

    class _Cb(CustomLogger):
        async def async_post_call_response_headers_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("bad header")

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    response = MagicMock()
    response._hidden_params = {}
    out = await proxy_logging.post_call_response_headers_hook(
        data={}, user_api_key_dict=make_user_api_key_auth(), response=response
    )
    assert out == {}


class _StreamBlocker(CustomGuardrail):
    def __init__(self, guardrail_name: str = "stream-blocker") -> None:
        super().__init__(guardrail_name=guardrail_name, event_hook=GuardrailEventHooks.post_call, default_on=True)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: AsyncIterator[object], request_data: dict[str, object]
    ) -> AsyncGenerator[object, None]:
        async for _ in response:
            raise HTTPException(status_code=400, detail={"error": "blocked"})
            yield  # pragma: no cover


class _StreamPasser(CustomGuardrail):
    def __init__(self, guardrail_name: str = "stream-passer") -> None:
        super().__init__(guardrail_name=guardrail_name, event_hook=GuardrailEventHooks.post_call, default_on=True)

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: AsyncIterator[object], request_data: dict[str, object]
    ) -> AsyncGenerator[object, None]:
        async for chunk in response:
            yield chunk


async def _drain_stream_chain(
    proxy_logging: ProxyLogging,
    user_api_key_dict: UserAPIKeyAuth,
    upstream: AsyncIterator[object],
    request_data: dict[str, object],
) -> None:
    async for _ in proxy_logging.async_post_call_streaming_iterator_hook(
        response=upstream,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
    ):
        pass


async def _failing_provider_stream() -> AsyncGenerator[object, None]:
    yield "chunk"
    raise RuntimeError("provider connection dropped")


@pytest.mark.asyncio
async def test_stream_guardrail_block_names_the_blocking_guardrail_in_applied_guardrails(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_StreamBlocker()])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    request_data: dict[str, object] = {"metadata": {}}
    with pytest.raises(HTTPException):
        await _drain_stream_chain(proxy_logging, make_user_api_key_auth(), _one_chunk(), request_data)
    assert request_data["metadata"]["applied_guardrails"] == ["stream-blocker"]


@pytest.mark.asyncio
async def test_stream_block_by_inner_guardrail_does_not_name_the_outer_layers(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_StreamBlocker(), _StreamPasser("outer-a"), _StreamPasser("outer-b")])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    request_data: dict[str, object] = {"metadata": {}}
    with pytest.raises(HTTPException) as info:
        await _drain_stream_chain(proxy_logging, make_user_api_key_auth(), _one_chunk(), request_data)
    assert info.value.detail["guardrail_name"] == "stream-blocker"
    assert request_data["metadata"]["applied_guardrails"] == ["stream-blocker"]


@pytest.mark.asyncio
async def test_stream_provider_failure_is_not_attributed_to_any_guardrail(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    monkeypatch.setattr(litellm, "callbacks", [_StreamPasser("outer-a"), _StreamPasser("outer-b")])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None, raising=False)

    request_data: dict[str, object] = {"metadata": {}}
    with pytest.raises(RuntimeError, match="provider connection dropped"):
        await _drain_stream_chain(proxy_logging, make_user_api_key_auth(), _failing_provider_stream(), request_data)
    assert request_data["metadata"] == {}
