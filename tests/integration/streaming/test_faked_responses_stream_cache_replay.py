import os
import uuid
from dataclasses import dataclass
from typing import Final

import openai
from integration._support.client import Gateway, eventually, string_value
from integration._support.wire import Wire, wire_server
from integration.streaming.faked_responses_case import (
    ANSWER,
    ARGUMENTS,
    RESPONSES_TOOL_PARAMS,
    RESPONSES_TOOLS,
    answering,
    assert_each_value_arrives_once,
    every_item_kind,
    faked_model,
    function_call_item,
    items_by_index,
    message_item,
    stream_events,
)
from openai.lib.streaming.responses import ResponseStreamEvent
from openai.types.responses import ParsedResponse, ResponseOutputMessage
from pydantic import JsonValue
from redis import Redis

_TARGET: Final = "/v1/responses"


@dataclass(frozen=True, slots=True)
class _SdkAnswer:
    events: "tuple[ResponseStreamEvent[None], ...]"
    final: ParsedResponse[None]
    reached_upstream: bool


def _cache_holds(needle: str) -> bool:
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as store:
        return any(needle.encode() in _bytes(store.get(key)) for key in store.scan_iter(count=1000, _type="STRING"))


def _bytes(value: object) -> bytes:
    return value if isinstance(value, bytes) else b""


def _reached_upstream(wire: Wire) -> bool:
    return bool([request for request in wire.drain() if request.method == "POST"])


def _message_id(events: tuple[dict[str, JsonValue], ...], index: int) -> str:
    return string_value(items_by_index(events, "response.output_item.done")[index]["id"])


def test_replayed_stream_opens_every_item_empty_and_each_value_arrives_once(gateway: Gateway) -> None:
    with wire_server(answering(every_item_kind)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        body: Final = {"model": model, "input": f"cached weather {uuid.uuid4().hex}", "tools": RESPONSES_TOOLS}
        _, filled = stream_events(gateway, _TARGET, body)
        message_id: Final = _message_id(filled, 1)
        assert_each_value_arrives_once(filled, every_item_kind(message_id.removeprefix("msg_")))
        assert len([request for request in wire.drain() if request.method == "POST"]) == 1
        eventually(lambda: _cache_holds(message_id), bool, seconds=30)
        _, replayed = stream_events(gateway, _TARGET, body)
        assert not _reached_upstream(wire), "the repeated stream reached the upstream instead of the cache"
        assert _message_id(replayed, 1) == message_id
        added: Final = items_by_index(replayed, "response.output_item.added")
        assert (added[1]["content"], added[2]["arguments"], added[0]["summary"]) == ([], "", []), added
        assert_each_value_arrives_once(replayed, every_item_kind(message_id.removeprefix("msg_")))


async def _sdk_attempt(client: openai.AsyncOpenAI, model: str, prompt: str, wire: Wire) -> _SdkAnswer:
    async with client.responses.stream(model=model, input=prompt, tools=RESPONSES_TOOL_PARAMS) as stream:
        events: Final = tuple([event async for event in stream])
        final: Final = await stream.get_final_response()
    return _SdkAnswer(events, final, _reached_upstream(wire))


def _message_ids(response: ParsedResponse[None]) -> list[str]:
    return [item.id for item in response.output if isinstance(item, ResponseOutputMessage)]


async def test_async_openai_sdk_replay_snapshots_hold_each_value_once(gateway: Gateway) -> None:
    with (
        wire_server(answering(lambda identity: (message_item(identity), function_call_item(identity)))) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = faked_model(scenario, wire)
        prompt: Final = f"cached sdk weather {uuid.uuid4().hex}"
        client: Final = openai.AsyncOpenAI(
            base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0
        )
        first: Final = await _sdk_attempt(client, model, prompt, wire)
        assert first.reached_upstream, first.final.id
        message_ids: Final = _message_ids(first.final)
        assert len(message_ids) == 1, message_ids
        eventually(lambda: _cache_holds(message_ids[0]), bool, seconds=30)
        replay: Final = await _sdk_attempt(client, model, prompt, wire)
        assert not replay.reached_upstream, "the repeated SDK stream reached the upstream instead of the cache"
        assert _message_ids(replay.final) == message_ids, (_message_ids(replay.final), message_ids)
        text: Final = [event.snapshot for event in replay.events if event.type == "response.output_text.delta"]
        arguments: Final = [
            event.snapshot for event in replay.events if event.type == "response.function_call_arguments.delta"
        ]
        assert text and text[-1] == ANSWER, text[-1:]
        assert arguments and arguments[-1] == ARGUMENTS, arguments[-1:]
