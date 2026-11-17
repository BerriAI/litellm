import json
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.streaming.test_stream_contracts import text_stream

KEEPALIVE_SECONDS: Final = 1


def _reply_after_first_ping(identity: str, first_ping_seen: threading.Event) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        first_ping_seen.wait(timeout=10)
        return Reply(content_type="text/event-stream", chunks=text_stream(identity))

    return respond


def _frames_setting(first_ping_seen: threading.Event, lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        if line == ": ping":
            first_ping_seen.set()
        yield line


@pytest.mark.covers("streaming.keepalive.sse_pings_fill_silent_time_to_first_token")
def test_stream_emits_sse_ping_comments_before_the_first_data_frame_while_upstream_is_silent(
    gateway: Gateway,
) -> None:
    identity: Final = "stream-ttft-keepalive-" + uuid.uuid4().hex
    first_ping_seen: Final = threading.Event()
    with gateway.scenario() as scenario:
        with wire_server(_reply_after_first_ping(identity, first_ping_seen)) as wire:
            model: Final = scenario.model(api_base=wire.url + "/v1", keepalive_seconds=KEEPALIVE_SECONDS)
            with gateway.client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": identity}], "stream": True},
                headers={"Authorization": f"Bearer {gateway.key}"},
            ) as response:
                assert response.status_code == 200, response.read().decode()
                frames: Final = tuple(
                    _frames_setting(first_ping_seen, (line for line in response.iter_lines() if line))
                )
            first_data: Final = next(index for index, line in enumerate(frames) if line.startswith("data:"))
            assert first_data >= 1, f"No keepalive reached the client before the first data frame: {frames}"
            assert frames[:first_data] == (": ping",) * first_data, frames
            assert frames[-1] == "data: [DONE]", frames
            deltas: Final = tuple(json.loads(line.removeprefix("data: ")) for line in frames[first_data:-1])
            assert (
                "".join(choice["delta"].get("content", "") for chunk in deltas for choice in chunk["choices"])
                == "Hello 雪 café"
            ), frames
            requests: Final = wire.drain()
            assert len(requests) == 1
            outbound: Final = json.loads(requests[0].body)
            assert outbound["model"] == "gpt-4o-mini" and outbound["stream"] is True, outbound
            assert "keepalive_seconds" not in outbound, outbound
