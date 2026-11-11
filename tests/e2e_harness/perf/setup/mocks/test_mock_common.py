from __future__ import annotations

import asyncio
import time
from typing import Final

from fastapi.responses import Response, StreamingResponse
from mock_common import Decision, Latency, RequestLog, json_reply, stream_reply

DELAY_S: Final = 0.05
TOKEN_INTERVAL_S: Final = 0.0005


def _decision(*, stream: bool, latency: Latency) -> tuple[RequestLog, Decision]:
    log: Final = RequestLog("anthropic")
    _ = log.set_latency(latency)
    body: Final = {
        "model": "claude-sonnet-5",
        "max_tokens": 2048,
        "stream": stream,
        "messages": [{"role": "user", "content": "which files swallow exceptions? perfreq-000001-n"}],
    }
    return log, log.decide(route="messages", stream=stream, body=body, model="claude-sonnet-5")


async def _chunk_arrivals(response: Response) -> tuple[float, ...]:
    assert isinstance(response, StreamingResponse)
    started: Final = time.monotonic()
    return tuple([time.monotonic() - started async for _ in response.body_iterator])


def test_stream_spreads_generation_time_over_the_gaps_between_chunks() -> None:
    log, decision = _decision(stream=True, latency=Latency(seconds=DELAY_S, token_interval_seconds=TOKEN_INTERVAL_S))
    arrivals: Final = asyncio.run(_chunk_arrivals(stream_reply(decision, "text/event-stream")))
    gaps: Final = tuple(later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False))
    expected_gap_s: Final = decision.generation_s / len(gaps)

    assert decision.generation_s == decision.reply.output_tokens * TOKEN_INTERVAL_S
    assert len(gaps) >= 3
    assert DELAY_S <= arrivals[0] < DELAY_S + 0.05
    assert decision.generation_s <= arrivals[-1] - arrivals[0] < decision.generation_s * 1.3
    assert min(gaps) >= expected_gap_s * 0.8
    assert log.snapshot()[0].generation_s == decision.generation_s


def test_stream_without_token_interval_sends_chunks_back_to_back() -> None:
    _, decision = _decision(stream=True, latency=Latency(seconds=DELAY_S))
    arrivals: Final = asyncio.run(_chunk_arrivals(stream_reply(decision, "text/event-stream")))

    assert decision.generation_s == 0.0
    assert arrivals[-1] - arrivals[0] < 0.02


def test_json_reply_waits_for_delay_plus_generation_time() -> None:
    _, decision = _decision(stream=False, latency=Latency(seconds=DELAY_S, token_interval_seconds=TOKEN_INTERVAL_S))
    started: Final = time.monotonic()
    _ = asyncio.run(json_reply(decision))
    elapsed: Final = time.monotonic() - started

    assert decision.generation_s > 0.2
    assert DELAY_S + decision.generation_s <= elapsed < DELAY_S + decision.generation_s + 0.1
