import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Final

import openai
from integration._support.anthropic_sse import dropping_reply
from integration._support.wire import Reply, Request, wire_server

import litellm
from litellm.exceptions import MidStreamFallbackError

_PARTIAL_TEXT: Final = "partial answer"


def _chunk(content: str) -> bytes:
    body: Final = {
        "id": "chatcmpl-sdk-cut",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _cut_after_text(request: Request) -> Reply:
    assert request.method == "POST" and "/chat/completions" in request.target, request.target
    return dropping_reply((_chunk(_PARTIAL_TEXT), _chunk(" never sent")), abort_after=1)


def _text(item: object) -> str:
    if not isinstance(item, litellm.ModelResponseStream):
        return ""
    return "".join(choice.delta.content or "" for choice in item.choices)


async def _items_or_error(stream: AsyncIterable[object]) -> AsyncIterator[object]:
    try:
        async for item in stream:
            yield item
    except openai.APIError as error:
        yield error


async def test_sdk_mid_stream_failure_debug_info_names_the_api_base_it_called() -> None:
    with wire_server(_cut_after_text) as wire:
        response: Final = await litellm.acompletion(
            model="azure/gpt-4o-mini",
            api_base=wire.url,
            api_key="synthetic-azure-key",
            api_version="2025-04-01-preview",
            messages=[{"role": "user", "content": "stream an answer"}],
            stream=True,
            num_retries=0,
        )
        assert isinstance(response, AsyncIterable), response
        items: Final = [item async for item in _items_or_error(response)]
    errors: Final = [item for item in items if isinstance(item, openai.APIError)]
    assert len(errors) == 1 and items[-1] is errors[0], items
    assert "".join(_text(item) for item in items) == _PARTIAL_TEXT, items
    error: Final = errors[0]
    mapped: Final = (
        error.original_exception
        if isinstance(error, MidStreamFallbackError) and error.original_exception is not None
        else error
    )
    debug_info: Final = getattr(mapped, "litellm_debug_info", None)
    assert isinstance(debug_info, str), mapped
    assert f"\nAPI Base: `{wire.url}`" in debug_info, debug_info
