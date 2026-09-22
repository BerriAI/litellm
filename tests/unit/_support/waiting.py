import asyncio
from collections.abc import Callable

_DEFAULT_TICKS = 200_000


async def until(condition: Callable[[], bool], *, ticks: int = _DEFAULT_TICKS) -> None:
    """Yield to the event loop until `condition` holds; no clock, bounded by loop iterations."""
    for _ in range(ticks):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"condition never became true within {ticks} loop iterations")
