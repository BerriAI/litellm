import json
from typing import Final

from integration._support.wire import Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    CITIES,
    EPHEMERAL,
    POINTS,
    PROVIDER_KEY,
    SYSTEM,
    TOOL,
    anthropic_labels,
    anthropic_peer,
    ask,
    call_id,
    client_marked,
    final_text,
    new_marker,
)
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ChatCompletionMessageToolCall, Message, ModelResponse

_CLIENT_MARKED: Final = client_marked()


def _marked_call(city: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id(city),
        type="function",
        function={"name": "lookup_weather", "arguments": json.dumps({"city": city})},
        cache_control=EPHEMERAL,
    )


def _pydantic_conversation(marker: str) -> list[JsonValue | Message]:
    return [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        Message(role="assistant", content="", tool_calls=[_marked_call(city) for city in CITIES]),
        *({"role": "tool", "tool_call_id": call_id(city), "content": "sunny."} for city in CITIES),
        {"role": "user", "content": final_text(marker)},
    ]


def _completion_kwargs(wire: Wire, marker: str) -> dict[str, object]:
    return {
        "model": f"anthropic/{ANTHROPIC_MODEL}",
        "messages": _pydantic_conversation(marker),
        "tools": [TOOL],
        "max_tokens": 64,
        "api_base": wire.url,
        "api_key": PROVIDER_KEY,
        "num_retries": 0,
        "cache_control_injection_points": POINTS,
    }


def test_pydantic_tool_call_marks_count_against_the_cap_in_process() -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire:
        response: Final = litellm.completion(**_completion_kwargs(wire, marker))  # pyright: ignore[reportArgumentType]  # Message objects stand in for the dict shapes the signature names
        assert isinstance(response, ModelResponse), type(response)
        assert response.choices[0].message.content == "sunny", response  # pyright: ignore[reportAttributeAccessIssue]  # choices are Choices for a non-stream response
        received: Final = wire.drain()
    assert [anthropic_labels(request) for request in received] == [_CLIENT_MARKED]


async def test_pydantic_tool_call_marks_count_against_the_cap_in_process_async_stream() -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire:
        stream: Final = await litellm.acompletion(**_completion_kwargs(wire, marker), stream=True)  # pyright: ignore[reportArgumentType]  # Message objects stand in for the dict shapes the signature names
        assert isinstance(stream, CustomStreamWrapper), type(stream)
        chunks: Final = [chunk async for chunk in stream]
        assert "".join(str(chunk.choices[0].delta.content or "") for chunk in chunks) == "sunny", chunks
        received: Final = wire.drain()
    assert [anthropic_labels(request) for request in received] == [_CLIENT_MARKED]
