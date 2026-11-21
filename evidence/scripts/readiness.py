# Completion signals for the issue 49929 fixtures. No elapsed-time waits are used as evidence:
# async cache writes are awaited through litellm's own pending-write task set, and a cache entry is
# only treated as written once it is observable in the store (bounded polling, failure on timeout).
import asyncio, time
import litellm

def cache_entries_ready(loop, before, min_entries=1, timeout=15.0):
    from litellm.caching import caching_handler
    pending = list(getattr(caching_handler, "_PENDING_CACHE_WRITES", ()))
    if pending and loop is not None:
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    store = litellm.cache.cache.cache_dict
    deadline = time.monotonic() + timeout
    while True:
        new = [k for k in list(store) if k not in before]
        if len(new) >= min_entries:
            return new
        if time.monotonic() > deadline:
            raise AssertionError("cache entry not observable: %d of %d new entries" % (len(new), min_entries))
        if loop is not None:
            loop.run_until_complete(asyncio.sleep(0))
        time.sleep(0.01)
