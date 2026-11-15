from collections.abc import AsyncIterator
from typing import Final

import anyio
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.router_rust.invoke.streaming import _until_first_output
from tests.unit.conftest import StreamClock


def _expired() -> MidStreamFallbackError:
    return MidStreamFallbackError(
        message="no output", model="test", llm_provider="openai", is_pre_first_chunk=True,
        original_exception=litellm.Timeout(message="no output", model="test", llm_provider="openai"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("already_expired", (False, True), ids=("H1-wait", "H5-expired"))
async def test_first_output_deadline_raises_provider_timeout(stream_clock: StreamClock, already_expired: bool) -> None:
    async def source() -> AsyncIterator[str]:
        await stream_clock.elapse(0.2)
        yield "output"

    deadline: Final = anyio.current_time() + (-0.1 if already_expired else 0.1)
    with pytest.raises(MidStreamFallbackError) as raised:
        _ = tuple([item async for item in _until_first_output(source(), deadline, bool, _expired)])
    assert raised.value.is_pre_first_chunk is True
    assert isinstance(raised.value.original_exception, litellm.Timeout)


@pytest.mark.asyncio
async def test_lifecycle_items_do_not_reset_first_output_deadline(stream_clock: StreamClock) -> None:
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
async def test_first_output_disables_deadline_for_remaining_items(stream_clock: StreamClock) -> None:
    async def source() -> AsyncIterator[str]:
        yield "output"
        await stream_clock.elapse(0.3)
        yield "later"

    assert tuple([item async for item in _until_first_output(source(), anyio.current_time() + 0.1, bool, _expired)]) == (
        "output", "later",
    )


@pytest.mark.asyncio
async def test_empty_source_ends_without_timeout(stream_clock: StreamClock) -> None:
    async def source() -> AsyncIterator[str]:
        for item in ():
            yield item

    assert tuple([item async for item in _until_first_output(source(), anyio.current_time() + 0.1, bool, _expired)]) == ()
