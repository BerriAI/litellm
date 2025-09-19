from __future__ import annotations
import asyncio
from typing import AsyncIterator, Dict, Iterable, Tuple


async def acompletion_as_completed(router, requests: Iterable[Dict], *, concurrency: int = 5) -> AsyncIterator[Tuple[int, Dict]]:
    """Schedule router.acompletion over a list and yield (index, response) in completion order."""
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()

    async def _one(i: int, req: Dict):
        async with sem:
            return i, await router.acompletion(**req)

    tasks = [loop.create_task(_one(i, r)) for i, r in enumerate(requests)]
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            yield await d
