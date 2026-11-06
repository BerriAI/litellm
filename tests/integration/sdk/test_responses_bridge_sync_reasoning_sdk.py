import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import LiteLLMCompletionStreamingIterator

_BACKEND: Final = "qwen3-sync-bridge-audit"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_ANSWER: Final = "It is sunny in Paris."
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEM_EVENTS: Final = frozenset({"response.output_item.added", "response.output_item.done"})


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _Item(_Frozen):
    type: str


class _Event(_Frozen):
    type: str
    item: _Item | None = None
    output_index: int | None = None
    delta: JsonValue = None


def _frame(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None = None) -> bytes:
    usage: Final = (
        {"usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42}} if finish_reason else {}
    )
    body: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _BACKEND,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        **usage,
    }
    return b"data: " + json.dumps(body).encode() + b"\n\n"


def _sse(deltas: Sequence[Mapping[str, JsonValue]]) -> Reply:
    identity: Final = f"chatcmpl-sync-bridge-{uuid.uuid4().hex}"
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _frame(identity, {"role": "assistant", "content": ""}),
            *(_frame(identity, delta) for delta in deltas),
            _frame(identity, {}, "stop"),
            b"data: [DONE]\n\n",
        ),
    )


def _streamed_chat_posts(wire: Wire) -> list[JsonValue]:
    posted: Final = [request for request in wire.drain() if request.method == "POST"]
    assert [request.target for request in posted] == ["/v1/chat/completions"]
    return [_JSON_OBJECT.validate_json(request.body).get("stream") for request in posted]


def _events(stream: Iterable[object]) -> tuple[_Event, ...]:
    return tuple(_Event.model_validate(_JSON_OBJECT.validate_python(_dumped(event))) for event in stream)


def _dumped(event: object) -> object:
    assert isinstance(event, BaseModel), type(event)
    return event.model_dump(mode="json")


def _delta_text(events: Sequence[_Event], event_type: str) -> str:
    return "".join(str(event.delta) for event in events if event.type == event_type)


@pytest.mark.parametrize(
    "deltas",
    [
        pytest.param(
            ({"reasoning_content": "REASONING"}, {"content": _ANSWER}),
            id="reasoning-then-answer-chunks",
        ),
        pytest.param(({"reasoning_content": "REASONING", "content": _ANSWER},), id="reasoning-and-answer-in-one-chunk"),
    ],
)
def test_sync_responses_stream_closes_the_reasoning_item_before_the_message_opens(
    deltas: tuple[dict[str, JsonValue], ...],
) -> None:
    marker: Final = uuid.uuid4().hex
    reasoning: Final = f"Sync bridge reasoning {marker}."
    scripted: Final = tuple(
        {key: reasoning if key == "reasoning_content" else value for key, value in delta.items()} for delta in deltas
    )

    def respond(request: Request) -> Reply:
        return Reply(body=b'{"object":"list","data":[]}') if request.method == "GET" else _sse(scripted)

    with wire_server(respond) as wire:
        stream: Final = litellm.responses(
            model=f"hosted_vllm/{_BACKEND}",
            input=f"Weather in Paris? {marker}",
            stream=True,
            api_base=wire.url + "/v1",
            api_key=_API_KEY,
            use_chat_completions_api=True,
            num_retries=0,
        )
        assert isinstance(stream, LiteLLMCompletionStreamingIterator), type(stream)
        events: Final = _events(stream)

        assert _streamed_chat_posts(wire) == [True]
        lifecycle: Final = [
            (event.type, event.item.type, event.output_index)
            for event in events
            if event.type in _ITEM_EVENTS and event.item is not None
        ]
        assert lifecycle == [
            ("response.output_item.added", "reasoning", 0),
            ("response.output_item.done", "reasoning", 0),
            ("response.output_item.added", "message", 1),
            ("response.output_item.done", "message", 1),
        ], [event.type for event in events]
        assert _delta_text(events, "response.reasoning_summary_text.delta") == reasoning
        assert _delta_text(events, "response.output_text.delta") == _ANSWER
