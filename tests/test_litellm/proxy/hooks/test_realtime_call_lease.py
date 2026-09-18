import asyncio
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.hooks.realtime_call_lease import RealtimeCallLease


@pytest.mark.asyncio
async def test_failed_renewal_signals_owner_and_close_releases_once():
    renew = AsyncMock(side_effect=[True, False, True])
    release = AsyncMock()
    lease = RealtimeCallLease(renew=renew, release=release, interval=0.001)
    lease.start()
    await asyncio.wait_for(lease.wait_failed(), timeout=1)
    assert renew.await_count == 2
    assert not await lease.renew()
    assert renew.await_count == 2
    await asyncio.gather(lease.close(), lease.close())
    assert release.await_count == 1


@pytest.mark.asyncio
async def test_renewal_exception_and_close_before_start():
    release = AsyncMock()
    lease = RealtimeCallLease(renew=AsyncMock(side_effect=RuntimeError("backend")), release=release, interval=0.001)
    lease.start()
    await asyncio.wait_for(lease.wait_failed(), timeout=1)
    await lease.close()
    assert release.await_count == 1
    unused = RealtimeCallLease(renew=AsyncMock(), release=release)
    await unused.close()
    assert release.await_count == 2


@pytest.mark.asyncio
async def test_renewal_timeout_signals_failure_without_start():
    lease = RealtimeCallLease(renew=asyncio.Event().wait, release=AsyncMock(), renewal_timeout=0.001)
    assert not await lease.renew()
    await asyncio.wait_for(lease.wait_failed(), timeout=1)
    await lease.close()


@pytest.mark.asyncio
async def test_cancelled_close_still_releases_exactly_once():
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def release():
        entered.set()
        await finish.wait()

    cleanup = AsyncMock(side_effect=release)
    lease = RealtimeCallLease(renew=AsyncMock(return_value=True), release=cleanup)
    lease.start()
    closing = asyncio.create_task(lease.close())
    await asyncio.wait_for(entered.wait(), timeout=1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    finish.set()
    await lease.close()
    assert cleanup.await_count == 1


@pytest.mark.asyncio
async def test_concurrent_renewal_cannot_restore_a_failed_lease():
    pending = asyncio.Event()
    entered = asyncio.Event()

    async def delayed_success():
        entered.set()
        await pending.wait()
        return True

    renew = AsyncMock(side_effect=delayed_success)
    lease = RealtimeCallLease(renew=renew, release=AsyncMock())
    first = asyncio.create_task(lease.renew())
    await asyncio.wait_for(entered.wait(), timeout=1)
    renew.side_effect = None
    renew.return_value = False
    assert not await lease.renew()
    pending.set()
    assert not await first
    await lease.close()
