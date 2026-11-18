import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.models.performance_alert import (
    Claim,
    PerformanceAlertStore,
    ResourceUnreadableError,
    SweepOwnershipLostError,
    SweepState,
    new_sweep,
    target_day,
)
from litellm.repositories.performance_alert_repository import KEY_DATE_INDEX

RUN_INTERVAL: Final = timedelta(minutes=5)
RUN_BUDGET_SECONDS: Final = 10.0
PAGE_SIZE: Final = 25
PAUSE_SECONDS: Final = 0.1


async def _step(store: PerformanceAlertStore, owner: str, state: SweepState) -> SweepState | None:
    """One page, falling back to a single resource and then stepping over it, so one bad resource
    cannot hold up the rest of the sweep."""
    try:
        return await store.evaluate_page(owner, state, PAGE_SIZE)
    except ResourceUnreadableError:
        pass
    try:
        return await store.evaluate_page(owner, state, 1)
    except ResourceUnreadableError:
        verbose_proxy_logger.warning(
            "Performance alerts skipped one %s it could not read within its limits", state.scope
        )
    return await store.skip_next(owner, state)


def _start(claim: Claim, now: datetime) -> SweepState:
    day: Final = target_day(now)
    if claim.state is not None and claim.state.day == day:
        return claim.state
    if claim.state is not None and not claim.state.completed:
        verbose_proxy_logger.info(
            "Performance alerts did not finish %s: stopped in %s after %d resources, %d skipped",
            claim.state.day,
            claim.state.scope,
            claim.state.checked,
            claim.state.skipped,
        )
    return new_sweep(day, claim.state)


async def run_performance_alerts(
    store: PerformanceAlertStore,
    interval: timedelta = RUN_INTERVAL,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    from prisma.errors import PrismaError
    from pydantic import ValidationError

    try:
        claim: Final = await store.claim(interval)
        if claim is None:
            return
        started: Final = _start(claim, now())
        if started.completed:
            return
        deadline: Final = clock() + RUN_BUDGET_SECONDS
        keys_ready: Final = await store.key_index_ready()
        state = started  # rebind-ok: each page returns the next saved position
        while not state.completed and clock() < deadline:
            if state.scope == "key" and not keys_ready:
                verbose_proxy_logger.info("Performance alerts waiting for %s before checking keys", KEY_DATE_INDEX)
                break
            state = await _step(store, claim.owner, state)
            if state is None:
                return
            await sleep(PAUSE_SECONDS)
        if state.completed:
            verbose_proxy_logger.info(
                "Performance alerts checked %d resources for %s, %d skipped", state.checked, state.day, state.skipped
            )
    except SweepOwnershipLostError:
        verbose_proxy_logger.info("Performance alerts run ended, another worker took over")
    except (PrismaError, ValidationError, TimeoutError) as exc:
        verbose_proxy_logger.warning("Performance alerts run ended early (%s)", type(exc).__name__)
