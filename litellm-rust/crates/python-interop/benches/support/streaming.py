import asyncio
from collections.abc import AsyncIterator
from typing import Final

loop: Final = asyncio.new_event_loop()


async def consume(source: AsyncIterator[bytes]) -> tuple[int, int]:
    count = 0  # rebind-ok: benchmark accumulator avoids retaining payloads
    total = 0  # rebind-ok: benchmark accumulator avoids retaining payloads
    async for chunk in source:
        count += 1
        total += len(chunk)
    return count, total
