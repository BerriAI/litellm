import uuid
from typing import Final

import pytest
from integration._support.responses_stream import (
    AZURE_TARGET,
    RATE_LIMIT_MESSAGE,
    function_tools,
    rate_limited_stream,
    serve,
)
from integration._support.wire import Wire, wire_server

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError, RateLimitError
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

_MODEL: Final = "azure/gpt-6"
_GROUP: Final = "bridged-gpt-6"
_API_KEY: Final = "synthetic-azure-key"
_SENTINEL_PREFIX: Final = "litellm.MidStreamFallbackError: "
_TOOLS: Final = function_tools()


def _messages(marker: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": marker}]


def _router(wire: Wire) -> Router:
    return Router(
        model_list=[
            {"model_name": _GROUP, "litellm_params": {"model": _MODEL, "api_base": wire.url, "api_key": _API_KEY}}
        ],
        num_retries=0,
    )


def _assert_one_attempt(wire: Wire, marker: str) -> None:
    received: Final = wire.drain()
    assert len(received) == 1 and marker.encode() in received[0].body, [request.target for request in received]


def _assert_wraps_the_provider_exception_once(raised: MidStreamFallbackError, wire: Wire, marker: str) -> None:
    inner: Final = raised.original_exception
    assert isinstance(inner, RateLimitError), repr(inner)
    assert inner.status_code == 429 and RATE_LIMIT_MESSAGE in str(inner), str(inner)
    assert raised.status_code == 429, raised.status_code
    assert raised.is_pre_first_chunk and raised.generated_content == "", (
        raised.is_pre_first_chunk,
        raised.generated_content,
    )
    assert str(raised).count(_SENTINEL_PREFIX) == 1, str(raised)
    _assert_one_attempt(wire, marker)


def _assert_surfaces_the_provider_exception(raised: RateLimitError, wire: Wire, marker: str) -> None:
    assert type(raised) is RateLimitError, type(raised)
    assert raised.status_code == 429 and RATE_LIMIT_MESSAGE in str(raised), str(raised)
    assert _SENTINEL_PREFIX not in str(raised), str(raised)
    _assert_one_attempt(wire, marker)


def test_sync_completion_stream_in_stream_rate_limit_wraps_the_provider_exception_once() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(f"resp_{marker}"), AZURE_TARGET)) as wire:
        response: Final = litellm.completion(
            model=_MODEL,
            messages=_messages(marker),
            tools=_TOOLS,
            stream=True,
            num_retries=0,
            api_base=wire.url,
            api_key=_API_KEY,
        )
        assert isinstance(response, CustomStreamWrapper), type(response)
        with pytest.raises(MidStreamFallbackError) as raised:
            for _ in response:
                pass
        _assert_wraps_the_provider_exception_once(raised.value, wire, marker)


async def test_async_completion_stream_in_stream_rate_limit_wraps_the_provider_exception_once() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(f"resp_{marker}"), AZURE_TARGET)) as wire:
        response: Final = await litellm.acompletion(
            model=_MODEL,
            messages=_messages(marker),
            tools=_TOOLS,
            stream=True,
            num_retries=0,
            api_base=wire.url,
            api_key=_API_KEY,
        )
        assert isinstance(response, CustomStreamWrapper), type(response)
        with pytest.raises(MidStreamFallbackError) as raised:
            async for _ in response:
                pass
        _assert_wraps_the_provider_exception_once(raised.value, wire, marker)


def test_router_sync_stream_in_stream_rate_limit_with_fallbacks_disabled_surfaces_the_provider_exception() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(f"resp_{marker}"), AZURE_TARGET)) as wire:
        response: Final = _router(wire).completion(
            model=_GROUP, messages=_messages(marker), tools=_TOOLS, stream=True, disable_fallbacks=True
        )
        with pytest.raises(RateLimitError) as raised:
            for _ in response:
                pass
        _assert_surfaces_the_provider_exception(raised.value, wire, marker)


async def test_router_async_stream_in_stream_rate_limit_without_fallbacks_surfaces_the_provider_exception() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(f"resp_{marker}"), AZURE_TARGET)) as wire:
        response: Final = await _router(wire).acompletion(
            model=_GROUP, messages=_messages(marker), tools=_TOOLS, stream=True
        )
        with pytest.raises(RateLimitError) as raised:
            async for _ in response:
                pass
        _assert_surfaces_the_provider_exception(raised.value, wire, marker)


async def test_router_async_stream_in_stream_rate_limit_with_fallbacks_disabled_surfaces_the_provider_exception() -> (
    None
):
    marker: Final = uuid.uuid4().hex
    with wire_server(serve(rate_limited_stream(f"resp_{marker}"), AZURE_TARGET)) as wire:
        response: Final = await _router(wire).acompletion(
            model=_GROUP, messages=_messages(marker), tools=_TOOLS, stream=True, disable_fallbacks=True
        )
        with pytest.raises(RateLimitError) as raised:
            async for _ in response:
                pass
        _assert_surfaces_the_provider_exception(raised.value, wire, marker)
