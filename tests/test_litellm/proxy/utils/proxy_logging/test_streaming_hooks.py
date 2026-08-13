"""Pin ProxyLogging streaming + response-headers helpers.

Covers ``_wrap_streaming_iterator_with_enrichment``,
``async_post_call_streaming_hook``,
``async_post_call_streaming_iterator_hook``, ``_fire_deferred_stream_logging``,
``is_a2a_streaming_response``, ``_init_response_taking_too_long_task``,
``post_call_response_headers_hook``, ``_build_litellm_call_info``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


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


@pytest.mark.asyncio
async def test_wrap_streaming_iterator_with_enrichment_passes_through_chunks(proxy_logging):
    async def gen():
        for ch in ("a", "b", "c"):
            yield ch

    cb = MagicMock(guardrail_name="g", event_hook="pre_call")
    wrapped = proxy_logging._wrap_streaming_iterator_with_enrichment(callback=cb, gen=gen())
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

    async def boom_gen():
        if False:
            yield  # pragma: no cover
        raise HTTPException(status_code=400, detail=detail)

    cb = MagicMock(guardrail_name="presidio", event_hook="post_call")
    wrapped = proxy_logging._wrap_streaming_iterator_with_enrichment(callback=cb, gen=boom_gen())
    with pytest.raises(HTTPException):
        async for _ in wrapped:
            pass
    assert detail["guardrail_name"] == "presidio"
    assert detail["guardrail_mode"] == "post_call"


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
