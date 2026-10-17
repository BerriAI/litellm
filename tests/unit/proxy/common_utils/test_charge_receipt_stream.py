import asyncio
from typing import Final

import pytest

from litellm.proxy.common_utils.charge_receipt_stream import ChargeReceiptStream


class HeldProvider:
    def __init__(self) -> None:
        self.entered: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.closed = False
        self.reads = 0

    def __aiter__(self) -> "HeldProvider":
        return self

    async def __anext__(self) -> object:
        self.entered.set()
        await self.release.wait()
        if self.reads:
            raise StopAsyncIteration
        self.reads += 1
        return "final usage"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_disconnect_preserves_pending_provider_read_for_final_usage() -> None:
    provider: Final = HeldProvider()
    stream: Final = ChargeReceiptStream(provider, provider.aclose)
    consumer: Final = asyncio.create_task(anext(stream))
    await provider.entered.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert provider.reads == 0 and not provider.closed
    provider.release.set()
    assert await anext(stream) == "final usage"
    await stream.drain()
    await stream.aclose()
    assert provider.reads == 1 and provider.closed


@pytest.mark.asyncio
async def test_expired_drain_closes_provider_with_an_unfinished_read() -> None:
    provider: Final = HeldProvider()
    stream: Final = ChargeReceiptStream(provider, provider.aclose)
    consumer: Final = asyncio.create_task(anext(stream))
    await provider.entered.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await stream.aclose()
    assert provider.closed and provider.reads == 0
