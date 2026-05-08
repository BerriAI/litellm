#!/usr/bin/env python3
"""
Bench LiteLLM_HealthCheckTable + PrismaClient 
    - set DATABASE_URL to your Postgres
    - Run ```prisma generate``` to install prisma client before running test )
    - This test writes to the default "public" database. Make sure to run cleanup after testing

"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from typing import Any, List

SEED_MARKER = (
    "benchmark_get_all_latest_health_checks.py"  # Utility Marker for cleanup process.
)


def _rss_kb_linux() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _fmt_kb(kb: int) -> str:
    if kb <= 0:
        return "n/a"
    return f"{kb} KiB (~{kb / 1024.0:.1f} MiB)"


def _build_batch(
    *,
    batch_index: int,
    batch_size: int,
    num_models: int,
    base_time: datetime,
) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for i in range(batch_size):
        global_i = batch_index * batch_size + i
        model_idx = global_i % max(num_models, 1)
        model_name = f"bench-model-{model_idx}"
        model_id = f"bench-mid-{model_idx}" if model_idx % 2 == 0 else None
        checked_at = base_time - timedelta(seconds=global_i)
        rows.append(
            {
                "model_name": model_name,
                "model_id": model_id,
                "status": "healthy" if global_i % 3 else "unhealthy",
                "healthy_count": 1,
                "unhealthy_count": 0,
                "checked_by": SEED_MARKER,
                "checked_at": checked_at,
            }
        )
    return rows


async def _seed(
    prisma: Any,
    *,
    total_rows: int,
    batch_size: int,
    num_models: int,
) -> None:
    db = prisma.db
    base_time = datetime.now(timezone.utc)
    inserted = 0
    batch_idx = 0
    while inserted < total_rows:
        n = min(batch_size, total_rows - inserted)
        await db.litellm_healthchecktable.create_many(
            data=_build_batch(
                batch_index=batch_idx,
                batch_size=n,
                num_models=num_models,
                base_time=base_time,
            )
        )
        inserted += n
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"  {inserted}/{total_rows}", flush=True)
    print(f"Seeded {inserted} rows ({SEED_MARKER}).")


async def _cleanup(prisma: Any) -> None:
    result = await prisma.db.litellm_healthchecktable.delete_many(
        where={"checked_by": SEED_MARKER},
    )
    n = getattr(result, "count", result)
    print(f"Deleted {n} rows.")


async def _bench(prisma: Any) -> None:
    gc.collect()
    rss0 = _rss_kb_linux()
    print(f"RSS (after gc): {_fmt_kb(rss0)}")

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        rows = await prisma.get_all_latest_health_checks()
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    gc.collect()
    rss1 = _rss_kb_linux()
    print(f"get_all_latest_health_checks: {len(rows)} rows in {elapsed:.2f}s")
    print(f"tracemalloc peak: {peak / 1e6:.2f} MiB")
    print(f"RSS after: {_fmt_kb(rss1)}")


async def _amain() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("action", choices=("seed", "bench", "cleanup"))
    p.add_argument("--rows", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--num-models", type=int, default=50)
    args = p.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Set DATABASE_URL.", file=sys.stderr)
        return 1

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from litellm.caching.caching import DualCache
    from litellm.proxy.proxy_cli import append_query_params
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    db_url = append_query_params(
        database_url, {"connection_limit": 100, "pool_timeout": 60}
    )
    prisma = PrismaClient(
        database_url=db_url,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
    )
    try:
        await prisma.connect()
    except Exception as e:
        print(f"Connect failed: {e}", file=sys.stderr)
        return 1

    try:
        if args.action == "seed":
            await _seed(
                prisma,
                total_rows=args.rows,
                batch_size=args.batch_size,
                num_models=args.num_models,
            )
        elif args.action == "bench":
            await _bench(prisma)
        else:
            await _cleanup(prisma)
    finally:
        try:
            await prisma.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
