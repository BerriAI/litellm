"""A chat stream served from the response cache carries the reasoning, citations, audio and tool calls it first had.

Every cell scripts one chat stream for its own deployment, streams it once through the OpenAI SDK so the proxy
assembles and caches it, then streams the identical request again until an answer arrives without an upstream call
and reads the replayed deltas the way a client would.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest
from openai import OpenAI
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration.caching.response_cache_case import (
    CHAT_CHUNK,
    Upstream,
    await_entry,
    prompt,
    redis_store,
    scripted,
)
from tests.integration.cost_calculation.cost_tracking_case import SseResponse

REPLAY_ATTEMPTS: Final = 8
CITATION: Final = (
    '{"type":"url_citation","url_citation":{"url":"https://example.com/apples","title":"Apples",'
    '"start_index":0,"end_index":19}}'
)
STOP: Final = CHAT_CHUNK + '[{"index":0,"delta":{},"finish_reason":"stop"}]}'
REASONING_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant","reasoning_content":"Count the apples "},'
        '"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"reasoning_content":"in each basket"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"There are "},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"14 apples"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"annotations":[' + CITATION + ']},"finish_reason":null}]}',
        STOP,
        "data: [DONE]",
    ),
)
AUDIO_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant","audio":{"id":"audio_$UNIQUE_ID","transcript":"Hello"}},'
        '"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"audio":{"data":"UklGRg=="}},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"audio":{"expires_at":1700000600}},"finish_reason":null}]}',
        STOP,
        "data: [DONE]",
    ),
)

TOOL_CALL: Final = (
    '{"index":0,"id":"call_$UNIQUE_ID","type":"function","function":{"name":"count_apples","arguments":"{}"}}'
)
TOOL_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant","reasoning_content":"I need the basket totals"},'
        '"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"Let me count "},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"the apples"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"tool_calls":[' + TOOL_CALL + ']},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ),
)
TOOL_ONLY_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant","reasoning_content":"I need the basket totals"},'
        '"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"tool_calls":[' + TOOL_CALL + ']},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ),
)

WHITESPACE_TOOL_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        CHAT_CHUNK + '[{"index":0,"delta":{"role":"assistant","reasoning_content":"I need the basket totals"},'
        '"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"content":"\\n\\n"},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{"tool_calls":[' + TOOL_CALL + ']},"finish_reason":null}]}',
        CHAT_CHUNK + '[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ),
)

_Stream = tuple[frozenset[str], tuple[dict[str, JsonValue], ...]]


def _stream(gateway: Gateway, model: str, text: str) -> _Stream:
    """The chunk ids and every choice's delta with its finish reason, after the whole stream was read."""
    with (
        OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
        ) as client,
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": text}], stream=True
        ) as stream,
    ):
        chunks: Final = tuple(stream)
    deltas: Final = tuple(
        {**choice.delta.model_dump(exclude_none=True), "finish_reason": choice.finish_reason}
        for chunk in chunks
        for choice in chunk.choices
    )  # comprehension-ok: flattens chunk choices
    return frozenset(chunk.id for chunk in chunks), deltas


def _attempt(gateway: Gateway, upstream: Upstream, identity: str, model: str, text: str) -> _Stream | None:
    before: Final = upstream.calls(identity)
    answer: Final = _stream(gateway, model, text)
    return answer if upstream.calls(identity) == before else None


def _attempts(gateway: Gateway, upstream: Upstream, identity: str, model: str, text: str) -> Iterator[_Stream | None]:
    return (_attempt(gateway, upstream, identity, model, text) for _ in range(REPLAY_ATTEMPTS))


def _replayed(gateway: Gateway, script: SseResponse) -> tuple[dict[str, JsonValue], ...]:
    """The deltas of the first identical stream served without an upstream call, after one stream filled the cache.

    A streamed request can reach the upstream once per proxy worker before a replay is served, since a worker keys a
    stream only once its router resolves the model the other worker registered.
    """
    upstream: Final = Upstream(gateway.upstream_url)
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, script)
        model: Final = scenario.model(api_base=handle.api_base())
        first_ids, _ = _stream(gateway, model, text)
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(redis_store(), next(iter(first_ids)))
        replay: Final = next(
            (answer for answer in _attempts(gateway, upstream, handle.scenario_id, model, text) if answer), None
        )
    if replay is None:
        pytest.fail(f"no stream was replayed from the cache in {REPLAY_ATTEMPTS} attempts")
    return replay[1]


def _joined(deltas: tuple[dict[str, JsonValue], ...], field: str) -> str:
    return "".join(str(delta[field]) for delta in deltas if delta.get(field))


def test_a_replayed_chat_stream_sends_the_reasoning_before_the_text_and_the_citation_after_it(
    gateway: Gateway,
) -> None:
    deltas: Final = _replayed(gateway, REASONING_STREAM)
    reasoning: Final = [index for index, delta in enumerate(deltas) if delta.get("reasoning_content")]
    text: Final = [index for index, delta in enumerate(deltas) if delta.get("content")]
    cited: Final = [index for index, delta in enumerate(deltas) if delta.get("annotations")]
    assert _joined(deltas, "reasoning_content") == "Count the apples in each basket", deltas
    assert _joined(deltas, "content") == "There are 14 apples", deltas
    assert reasoning and text and max(reasoning) < min(text), deltas
    assert cited == [max(text)] and deltas[max(text)]["annotations"] == [json.loads(CITATION)], deltas
    assert deltas[-1]["finish_reason"] == "stop", deltas


def test_a_replayed_chat_stream_without_text_sends_its_role_and_audio(gateway: Gateway) -> None:
    deltas: Final = _replayed(gateway, AUDIO_STREAM)
    audio: Final = [delta["audio"] for delta in deltas if delta.get("audio")]
    assert deltas[0].get("role") == "assistant", deltas
    assert len(audio) == 1, deltas
    assert isinstance(audio[0], dict) and audio[0]["data"] == "UklGRg==" and audio[0]["transcript"] == "Hello", deltas
    assert deltas[-1]["finish_reason"] == "stop", deltas


def test_a_replayed_chat_stream_sends_the_tool_call_after_the_reasoning_and_the_text(gateway: Gateway) -> None:
    deltas: Final = _replayed(gateway, TOOL_STREAM)
    reasoning: Final = [index for index, delta in enumerate(deltas) if delta.get("reasoning_content")]
    text: Final = [index for index, delta in enumerate(deltas) if delta.get("content")]
    called: Final = [index for index, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert _joined(deltas, "content") == "Let me count the apples", deltas
    assert reasoning and text and max(reasoning) < min(text), deltas
    assert called == [max(text) + 1], deltas
    assert deltas[-1]["finish_reason"] == "tool_calls", deltas


def test_a_replayed_chat_stream_without_text_sends_the_reasoning_before_the_tool_call(gateway: Gateway) -> None:
    deltas: Final = _replayed(gateway, TOOL_ONLY_STREAM)
    reasoning: Final = [index for index, delta in enumerate(deltas) if delta.get("reasoning_content")]
    called: Final = [index for index, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert _joined(deltas, "reasoning_content") == "I need the basket totals", deltas
    assert reasoning and called and max(reasoning) < min(called), deltas
    assert deltas[-1]["finish_reason"] == "tool_calls", deltas


def test_a_replayed_chat_stream_keeps_whitespace_only_text_between_the_reasoning_and_the_tool_call(
    gateway: Gateway,
) -> None:
    deltas: Final = _replayed(gateway, WHITESPACE_TOOL_STREAM)
    reasoning: Final = [index for index, delta in enumerate(deltas) if delta.get("reasoning_content")]
    text: Final = [index for index, delta in enumerate(deltas) if delta.get("content")]
    called: Final = [index for index, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert _joined(deltas, "content") == "\n\n", deltas
    assert reasoning and text and called and max(reasoning) < min(text) and max(text) < min(called), deltas
    assert deltas[-1]["finish_reason"] == "tool_calls", deltas
