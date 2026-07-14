"""Re-run the PTU reservation daily rollup for a specific UTC date.

Idempotent under the LiteLLM_DailyTeamSpend unique constraint. Reads
DATABASE_URL from the environment, connects Prisma directly, and bypasses
the ``enable_ptu_cost_attribution`` config flag so operators can backfill
without turning the feature on for live traffic.

Usage:
    python scripts/ptu_reservation_backfill.py --date 2026-07-12
    python scripts/ptu_reservation_backfill.py --date-range 2026-07-01:2026-07-12
"""

import argparse
import asyncio
import os
import sys
from datetime import date, datetime, timedelta


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", type=_parse_date, help="Single UTC date (YYYY-MM-DD)")
    group.add_argument(
        "--date-range",
        type=str,
        help="Inclusive UTC range as YYYY-MM-DD:YYYY-MM-DD",
    )
    return parser.parse_args()


def _dates_from_args(args: argparse.Namespace) -> list[date]:
    if args.date is not None:
        return [args.date]
    start_str, _, end_str = args.date_range.partition(":")
    start = _parse_date(start_str)
    end = _parse_date(end_str)
    if end < start:
        raise ValueError(f"end date {end} is before start date {start}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


async def _run(dates: list[date]) -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        return 2

    from litellm._logging import verbose_proxy_logger
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.spend_tracking.ptu_reservation_rollup import (
        run_ptu_reservation_rollup,
    )
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    prisma_client = PrismaClient(database_url=database_url, proxy_logging_obj=proxy_logging_obj)
    await prisma_client.connect()
    try:
        total_rows = 0
        for target in dates:
            result = await run_ptu_reservation_rollup(prisma_client, target_date=target, force=True)
            print(
                f"[{result.day.isoformat()}] "
                f"reservations={result.reservations_processed} rows_written={result.rows_written}"
            )
            total_rows += result.rows_written
        print(f"total rows written: {total_rows}")
        return 0
    finally:
        try:
            await prisma_client.db.disconnect()
        except Exception as exc:
            verbose_proxy_logger.debug("prisma disconnect failed: %s", exc)


def main() -> int:
    args = _parse_args()
    dates = _dates_from_args(args)
    if "PYTHONPATH" not in os.environ:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return asyncio.run(_run(dates))


if __name__ == "__main__":
    sys.exit(main())
