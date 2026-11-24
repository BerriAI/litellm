import asyncio
from collections.abc import AsyncIterator, Iterator
from types import MappingProxyType
from typing import Final

import anyio
import pytest
import pytest_asyncio

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_config import build_router_config
from litellm.router_python_host import python_host
from litellm.router_rust.core import RustCore
from litellm.router_rust.invoke.streaming import StreamAttempt, _until_first_output, wrap_anthropic_stream


class _ManualClock:
    __slots__ = ("now",)

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    async def elapse(self, seconds: float) -> None:
        steps: Final = max(1, round(seconds / 0.01))
        for _ in range(steps):
            self.now += seconds / steps
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)


@pytest_asyncio.fixture(loop_scope="function")
async def stream_clock(monkeypatch: pytest.MonkeyPatch) -> _ManualClock:
    loop: Final = asyncio.get_running_loop()
    clock: Final = _ManualClock(loop.time())
    monkeypatch.setattr(loop, "time", clock)
    return clock


def _expired() -> MidStreamFallbackError:
    return MidStreamFallbackError(
        message="no output",
        model="test",
        llm_provider="openai",
        is_pre_first_chunk=True,
        original_exception=litellm.Timeout(message="no output", model="test", llm_provider="openai"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("already_expired", (False, True), ids=("H1-wait", "H5-expired"))
async def test_first_output_deadline_raises_provider_timeout(stream_clock: _ManualClock, already_expired: bool) -> None:
    async def source() -> AsyncIterator[str]:
        await stream_clock.elapse(0.2)
        yield "output"

    started: Final = anyio.current_time()
    deadline: Final = started + (-0.1 if already_expired else 0.1)
    with pytest.raises(MidStreamFallbackError) as raised:
        _ = tuple([item async for item in _until_first_output(source(), deadline, bool, _expired)])
    assert raised.value.is_pre_first_chunk is True
    assert isinstance(raised.value.original_exception, litellm.Timeout)
    assert stream_clock() - started == (0.0 if already_expired else pytest.approx(0.1, abs=0.02))


@pytest.mark.asyncio
async def test_lifecycle_items_do_not_reset_first_output_deadline(stream_clock: _ManualClock) -> None:
    async def source() -> AsyncIterator[str]:
        for _ in range(5):
            await stream_clock.elapse(0.04)
            yield "lifecycle"

    items: Final = _until_first_output(source(), anyio.current_time() + 0.1, lambda item: item == "output", _expired)
    assert await anext(items) == "lifecycle"
    assert await anext(items) == "lifecycle"
    with pytest.raises(MidStreamFallbackError):
        await anext(items)


@pytest.mark.asyncio
async def test_first_output_disables_deadline_for_remaining_items(stream_clock: _ManualClock) -> None:
    async def source() -> AsyncIterator[str]:
        yield "output"
        await stream_clock.elapse(0.3)
        yield "later"

    assert tuple(
        [item async for item in _until_first_output(source(), anyio.current_time() + 0.1, bool, _expired)]
    ) == (
        "output",
        "later",
    )


@pytest.mark.asyncio
async def test_empty_source_ends_without_timeout(stream_clock: _ManualClock) -> None:
    async def source() -> AsyncIterator[str]:
        for item in ():
            yield item

    assert (
        tuple([item async for item in _until_first_output(source(), anyio.current_time() + 0.1, bool, _expired)]) == ()
    )


@pytest.fixture
def stream_attempt() -> Iterator[StreamAttempt]:
    core: Final = RustCore.build(build_router_config(MappingProxyType({"model_list": []})), python_host)
    yield StreamAttempt(
        core=core,
        op="anthropic_messages",
        request=MappingProxyType({"model": "claude"}),
        kwargs=MappingProxyType({"stream": True}),
        group="claude",
        order=None,
        depth=0,
        level=0,
        deployment_id=None,
        attempted_retries=0,
    )
    core.retire()


@pytest.mark.asyncio
async def test_a_deeply_nested_line_with_a_usage_key_does_not_end_the_messages_stream(stream_attempt: StreamAttempt):
    frames: Final = (
        b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","usage":' + b"[" * 100_000 + b"]" * 100_000 + b"}\n\n",
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":5}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )

    async def upstream() -> AsyncIterator[bytes]:
        for frame in frames:
            yield frame

    relayed: Final = b"".join([chunk async for chunk in wrap_anthropic_stream(stream_attempt, upstream())])

    assert relayed == b"".join(frames)
