"""Helpers for the managed-object poll-cap regression (#23472), black-box.

The bug: CheckBatchCost paged its managed-object query unbounded, so each poll
cycle pulled the whole table into pod memory and OOM'd. The fix caps the query at
MAX_OBJECTS_PER_POLL_CYCLE rows per cycle. There is no API that exposes the query,
but the cap is observable in the proxy's logs: rows the cycle selects but can't
decode are logged as "Skipping job <unified_object_id> ...". So seed more than one
page of selectable rows and watch which the cycle touches - a capped cycle touches
exactly the oldest page, an unbounded one touches them all.

No imports from the litellm codebase: rows are seeded directly in Postgres via the
generated prisma client, and the proxy's logs are read via a swappable command
(docker locally, a pod-log command on EKS).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from prisma import Prisma

PROXY_DB_URL = os.getenv(
    "E2E_MANAGED_DB_URL",
    "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm",
)
CONTAINER = os.getenv("LITELLM_CONTAINER", "e2e-litellm-1")

SEED_PREFIX = "pollcap"


def proxy_poll_cap() -> int:
    """The proxy's per-cycle row cap (MAX_OBJECTS_PER_POLL_CYCLE). Read from the
    proxy container's env so the test matches the running proxy; defaults to the
    proxy's own default of 50 when unset. Override with E2E_POLL_CAP."""
    override = os.getenv("E2E_POLL_CAP")
    if override is not None:
        return int(override)
    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER, "printenv", "MAX_OBJECTS_PER_POLL_CYCLE"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 50
    value = result.stdout.strip()
    return int(value) if value.isdigit() else 50


def parse_skipped_indices(log_text: str, prefix: str) -> frozenset[int]:
    """The seeded-row indices the proxy logged as skipped. Each seeded row's
    unified_object_id is ``<prefix>-NNN``; the poll logs it when it selects but
    cannot decode the row."""
    pattern = re.compile(re.escape(prefix) + r"-(\d{3})")
    return frozenset(int(match) for match in pattern.findall(log_text))


@dataclass(frozen=True, slots=True)
class ProxyLog:
    container: str

    def skipped_indices(self, prefix: str, *, since_seconds: int) -> frozenset[int]:
        result = subprocess.run(
            ["docker", "logs", "--since", f"{since_seconds}s", self.container],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return parse_skipped_indices(result.stdout + result.stderr, prefix)


@dataclass(frozen=True, slots=True)
class ManagedObjectSeeder:
    """Seeds invalid (undecodable) batch managed-objects so the poll selects them,
    logs a skip, and leaves them in place to be re-selected next cycle - never
    processed, never deleted by the proxy. created_at is recent (well inside the
    staleness cutoff) and strictly increasing, so the oldest `count` rows are a
    deterministic page."""

    db_url: str
    prefix: str

    async def _client(self) -> Prisma:
        client = Prisma(datasource={"url": self.db_url})
        await client.connect()
        return client

    async def reset(self) -> None:
        client = await self._client()
        try:
            await client.litellm_managedobjecttable.delete_many(
                where={"unified_object_id": {"startswith": f"{SEED_PREFIX}-"}}
            )
        finally:
            await client.disconnect()

    async def seed(self, count: int) -> None:
        client = await self._client()
        try:
            base = datetime.now(timezone.utc) - timedelta(seconds=count + 5)
            for i in range(count):
                unified_object_id = f"{self.prefix}-{i:03d}"
                await client.litellm_managedobjecttable.create(
                    data={
                        "unified_object_id": unified_object_id,
                        "model_object_id": f"{self.prefix}-mob-{i}",
                        "file_object": json.dumps(
                            {"id": unified_object_id, "object": "batch"}
                        ),
                        "file_purpose": "batch",
                        "status": "validating",
                        "batch_processed": False,
                        "team_id": self.prefix,
                        "created_at": base + timedelta(seconds=i),
                    }
                )
        finally:
            await client.disconnect()

    async def delete(self) -> None:
        client = await self._client()
        try:
            await client.litellm_managedobjecttable.delete_many(
                where={"unified_object_id": {"startswith": f"{self.prefix}-"}}
            )
        finally:
            await client.disconnect()
