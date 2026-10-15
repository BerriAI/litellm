import json
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.openai_wire import responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    CITIES,
    EPHEMERAL,
    POINTS,
    PROVIDER_KEY,
    SYSTEM,
    TOOL,
    UNMAPPED_BEDROCK_ARN,
    anthropic_labels,
    anthropic_peer,
    ask,
    bedrock_peer,
    call_id,
    client_marked,
    final_text,
    new_marker,
    sent_without_cache_points,
)
from openai.types.responses import FunctionTool
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


_OPENAI_MODEL: Final = "gpt-5.6"


def _openai_responses_peer(request: Request) -> Reply:
    return responses_reply("resp_sdk_typed_tool", _OPENAI_MODEL, "sunny", stream=False)


def _outcome(call: Callable[[], object]) -> str:
    try:
        response: Final = call()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return f"{type(response).__name__}: {getattr(response, 'output_text', response)}"


def test_injection_points_leave_an_sdk_typed_function_tool_request_as_it_is_without_them() -> None:
    tool: Final = FunctionTool(type="function", name="lookup_weather", parameters={"type": "object"}, strict=False)
    with wire_server(_openai_responses_peer) as wire:
        request: Final[dict[str, object]] = {
            "model": f"openai/{_OPENAI_MODEL}",
            "input": final_text(new_marker()),
            "tools": [tool],
            "api_base": wire.url,
            "api_key": PROVIDER_KEY,
            "num_retries": 0,
        }
        without_points: Final = _outcome(lambda: litellm.responses(**request))  # pyright: ignore[reportArgumentType]  # the SDK-typed tool stands in for the dict shape the signature names
        without_points_received: Final = len(wire.drain())
        with_points: Final = _outcome(
            lambda: litellm.responses(**request, cache_control_injection_points=POINTS)  # pyright: ignore[reportArgumentType]  # the SDK-typed tool stands in for the dict shape the signature names
        )
        with_points_received: Final = len(wire.drain())
    assert with_points == without_points
    assert with_points_received == without_points_received


def _shaped_input(shape: str, marker: str) -> object:
    system: Final = {"role": "system", "content": SYSTEM, "cache_control": EPHEMERAL}
    marked: Final = {"type": "message", "role": "user", "content": final_text(marker), "cache_control": EPHEMERAL}
    match shape:
        case "string":
            return final_text(marker)
        case "list":
            return [system, marked]
        case "tuple":
            return (system, marked)
        case _:
            return marked


@pytest.mark.parametrize("shape", ("string", "list", "tuple", "object"))
def test_each_input_shape_reads_the_same_whether_or_not_the_route_keeps_item_marks(shape: str) -> None:
    kept_marker, stripped_marker = new_marker(), new_marker()
    with wire_server(bedrock_peer) as wire:
        request: Final[dict[str, object]] = {
            "model": f"bedrock/converse/{UNMAPPED_BEDROCK_ARN}",
            "instructions": SYSTEM,
            "max_output_tokens": 64,
            "api_base": wire.url,
            "api_key": PROVIDER_KEY,
            "aws_region_name": "us-east-1",
            "aws_bedrock_runtime_endpoint": wire.url,
            "num_retries": 0,
        }
        kept: Final = _outcome(lambda: litellm.responses(input=_shaped_input(shape, kept_marker), **request))  # pyright: ignore[reportArgumentType]  # the shapes a caller can pass, unvalidated
        kept_sent: Final = sent_without_cache_points(wire.drain(), kept_marker)
        stripped: Final = _outcome(
            lambda: litellm.responses(input=_shaped_input(shape, stripped_marker), drop_params=True, **request)  # pyright: ignore[reportArgumentType]  # the shapes a caller can pass, unvalidated
        )
        stripped_sent: Final = sent_without_cache_points(wire.drain(), stripped_marker)
    assert stripped == kept
    assert stripped_sent == kept_sent
