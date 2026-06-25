"""Black-box regression guard for the unbounded managed-object poll OOM (#23472).

CheckBatchCost pages its managed-object query with take=MAX_OBJECTS_PER_POLL_CYCLE.
Before the fix the query was unbounded, so each poll cycle pulled the entire table
into pod memory and OOM'd. This guard seeds one page + 10 selectable rows into the
real Postgres, lets the live proxy run a poll cycle, and asserts the cycle touched
exactly the oldest page (the cap) and never the 10 newest rows. Drop the take (the
regression) and the cycle selects every seeded row, so the newest rows show up in
the logs and the assertion fails.

Pure black-box: rows are seeded via the generated prisma client and the cap is read
from the proxy's logs - no imports from the litellm codebase, no calls into its
internals. The proxy's batch poll runs every ~15-45s here (proxy_batch_polling_interval
is shortened in the e2e config so a cycle lands inside the test).

Skips unless a proxy answers (shared conftest) and the poll actually runs within the
deadline; once a cycle is observed, behavior is asserted. Seeded rows carry a unique
prefix and are deleted in a finally block.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from poll_cap import (
    CONTAINER,
    PROXY_DB_URL,
    SEED_PREFIX,
    ManagedObjectSeeder,
    ProxyLog,
    proxy_poll_cap,
)

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]

EXTRA_ROWS = 10
DEADLINE_SECONDS = 180
STABLE_SECONDS = 50


async def test_managed_object_poll_is_capped_per_cycle() -> None:
    cap = proxy_poll_cap()
    prefix = f"{SEED_PREFIX}-{uuid.uuid4().hex[:8]}"
    seeder = ManagedObjectSeeder(db_url=PROXY_DB_URL, prefix=prefix)
    log = ProxyLog(container=CONTAINER)
    oldest_page = frozenset(range(cap))
    newest_rows = frozenset(range(cap, cap + EXTRA_ROWS))

    start = time.monotonic()
    await seeder.reset()
    await seeder.seed(cap + EXTRA_ROWS)
    try:
        skipped: frozenset[int] = frozenset()
        first_full_at: float | None = None
        while time.monotonic() - start < DEADLINE_SECONDS:
            since = int(time.monotonic() - start) + 5
            skipped = log.skipped_indices(prefix, since_seconds=since)
            if skipped & newest_rows:
                break
            if len(skipped) >= cap and first_full_at is None:
                first_full_at = time.monotonic()
            if (
                first_full_at is not None
                and time.monotonic() - first_full_at >= STABLE_SECONDS
            ):
                break
            await asyncio.sleep(5)

        if not skipped:
            pytest.skip("managed-object batch poll did not run within the deadline")

        assert skipped == oldest_page, (
            f"poll cycle touched {len(skipped)} of {cap + EXTRA_ROWS} seeded rows "
            f"(expected exactly the oldest {cap}); newest rows it should never reach "
            f"= {sorted(skipped & newest_rows)}. The per-cycle cap "
            f"(MAX_OBJECTS_PER_POLL_CYCLE={cap}) must bound the query (#23472 OOM regression)"
        )
    finally:
        await seeder.delete()
