"""MavvrikOrchestrator — export pipeline: register → date loop → upload → advance.

Owns all business orchestration for the Mavvrik export:
  - Retrieving the current marker from the Mavvrik API via client.register()
  - Resolving the first-run start date (env var / earliest DB date / yesterday)
  - Iterating over complete calendar days since the last exported marker
  - Exporting each day via MavvrikExporter.export_usage_data()
  - Advancing the remote marker after each successful upload
  - Reporting errors to Mavvrik for visibility

Infrastructure concerns (pod lock, APScheduler registration) live in
MavvrikScheduler, which delegates to this class.
"""

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_LOOKBACK_START_DATE,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)

if TYPE_CHECKING:
    from litellm.integrations.mavvrik.exporter import MavvrikExporter
else:
    MavvrikExporter = Any


class MavvrikOrchestrator:
    """Pipeline orchestrator that drives the incremental Mavvrik export loop."""

    def __init__(self, exporter: "MavvrikExporter") -> None:
        self._exporter = exporter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today() -> date:
        """Return today's date in UTC.  Extracted so tests can patch it."""
        return datetime.now(timezone.utc).date()

    @property
    def _yesterday(self) -> date:
        return self._utc_today() - timedelta(days=1)

    # ------------------------------------------------------------------
    # Main entry point (called by MavvrikScheduler after acquiring lock)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Export every complete calendar day since the Mavvrik-side marker.

        The marker (cursor) is owned by the Mavvrik API.  We call register()
        at the start of every run to retrieve it — no local marker needed.
        """
        from litellm.integrations.mavvrik.client import MavvrikClient
        from litellm.integrations.mavvrik.database import MavvrikDatabase

        db = MavvrikDatabase()
        yesterday = self._yesterday

        # Retrieve the current export marker from Mavvrik (source of truth).
        # Falls back to _resolve_first_run_start_date if the call fails or
        # returns None (brand-new connection, no marker yet).
        marker_str: Optional[str] = None
        try:
            client = MavvrikClient(
                api_key=self._exporter.api_key or "",
                api_endpoint=self._exporter.api_endpoint or "",
                connection_id=self._exporter.connection_id or "",
            )
            marker_str = await client.register()
            verbose_logger.debug("MavvrikOrchestrator: Mavvrik marker = %s", marker_str)
        except Exception as exc:
            verbose_logger.warning(
                "MavvrikOrchestrator: register() failed (non-fatal), "
                "falling back to first-run start date: %s",
                exc,
            )

        # Determine start date from the Mavvrik marker or first-run logic.
        if marker_str:
            try:
                last_exported = date.fromisoformat(marker_str[:10])
                start_date = last_exported + timedelta(days=1)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikOrchestrator: invalid marker '%s', starting from yesterday",
                    marker_str,
                )
                start_date = yesterday
        else:
            start_date = await self._resolve_first_run_start_date(db, yesterday)

        if start_date > yesterday:
            verbose_logger.debug(
                "MavvrikOrchestrator: marker=%s is up to date, nothing to export",
                marker_str,
            )
            return

        export_date = start_date
        while export_date <= yesterday:
            date_str = export_date.isoformat()
            verbose_logger.info("MavvrikOrchestrator: exporting date %s", date_str)

            try:
                await self._exporter.export_usage_data(
                    date_str=date_str,
                    limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
                )
            except ValueError as exc:
                # Config error (missing credentials) — stop the entire loop;
                # no point trying remaining dates with a broken config.
                verbose_logger.error(
                    "MavvrikOrchestrator: export stopped for %s — config error: %s",
                    date_str,
                    exc,
                )
                await client.report_error(f"Config error for date {date_str}: {exc}")
                return
            except Exception as exc:
                # Transient error (network, upload failure) — log and skip
                # this date so remaining days still get exported.
                verbose_logger.warning(
                    "MavvrikOrchestrator: export failed for %s "
                    "(skipping, will retry next run): %s",
                    date_str,
                    exc,
                )
                await client.report_error(f"Export failed for date {date_str}: {exc}")
                export_date += timedelta(days=1)
                continue

            # Advance the marker on the Mavvrik side (source of truth).
            # No local marker is stored — the next run retrieves the
            # updated marker from Mavvrik via register().
            try:
                export_epoch = int(
                    datetime(
                        export_date.year,
                        export_date.month,
                        export_date.day,
                        tzinfo=timezone.utc,
                    ).timestamp()
                )
                await client.advance_marker(export_epoch)
            except Exception as exc:
                verbose_logger.warning(
                    "MavvrikOrchestrator: advance_marker PATCH failed (non-fatal): %s",
                    exc,
                )

            export_date += timedelta(days=1)

    # ------------------------------------------------------------------
    # First-run start date resolution
    # ------------------------------------------------------------------

    async def _resolve_first_run_start_date(
        self,
        db: Any,  # MavvrikDatabase — Any to avoid circular import at class level
        yesterday: date,
    ) -> date:
        """Determine the export start date on the very first run (no marker yet).

        Priority:
          1. MAVVRIK_LOOKBACK_START_DATE env var (clamped to MIN(date) in DB)
          2. MIN(date) in LiteLLM_DailyUserSpend
          3. Yesterday as a last-resort fallback
        """
        requested_start: Optional[date] = None
        if MAVVRIK_LOOKBACK_START_DATE is not None:
            try:
                requested_start = date.fromisoformat(MAVVRIK_LOOKBACK_START_DATE)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikOrchestrator: invalid MAVVRIK_LOOKBACK_START_DATE '%s' "
                    "(expected YYYY-MM-DD), falling back to earliest DB date",
                    MAVVRIK_LOOKBACK_START_DATE,
                )

        earliest_str = await db.get_earliest_date()
        earliest_db: Optional[date] = None
        if earliest_str:
            try:
                earliest_db = date.fromisoformat(earliest_str)
            except ValueError:
                pass

        if requested_start is not None and earliest_db is not None:
            start_date = max(requested_start, earliest_db)
            if start_date != requested_start:
                verbose_logger.info(
                    "MavvrikOrchestrator: MAVVRIK_LOOKBACK_START_DATE=%s is before "
                    "earliest DB date %s — starting from %s",
                    requested_start,
                    earliest_db,
                    start_date,
                )
            else:
                verbose_logger.info(
                    "MavvrikOrchestrator: no marker found, starting from "
                    "MAVVRIK_LOOKBACK_START_DATE %s",
                    start_date,
                )
        elif requested_start is not None:
            start_date = requested_start
            verbose_logger.info(
                "MavvrikOrchestrator: no marker found, no DB data yet, "
                "starting from MAVVRIK_LOOKBACK_START_DATE %s",
                start_date,
            )
        elif earliest_db is not None:
            start_date = earliest_db
            verbose_logger.info(
                "MavvrikOrchestrator: no marker found, starting from earliest DB date %s",
                start_date,
            )
        else:
            start_date = yesterday

        return start_date
