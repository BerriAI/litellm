"""MavvrikOrchestrator — export pipeline: register → date loop → upload → advance.

Owns all business orchestration for the Mavvrik export:
  - Retrieving the current marker from the Mavvrik API via client.register()
  - Resolving the first-run start date (env var / earliest DB date / yesterday)
  - Iterating over complete calendar days from the marker to yesterday
  - Exporting each day via MavvrikExporter.export_usage_data()
  - Advancing the remote marker to the next day after each successful upload
  - Reporting errors to Mavvrik for visibility

Infrastructure concerns (pod lock, APScheduler registration) live in
MavvrikScheduler, which delegates to this class.

Marker semantics:
  metricsMarker returned by register() is the START of the window to export,
  not the last exported date. After exporting a date, advance_marker() is
  called with (export_date + 1 day) so the next run starts from there.
"""

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_LOOKBACK_START_DATE,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
from litellm.integrations.mavvrik.client import MavvrikClient
from litellm.integrations.mavvrik.database import MavvrikDatabase

if TYPE_CHECKING:
    from litellm.integrations.mavvrik.exporter import MavvrikExporter
else:
    MavvrikExporter = Any


class MavvrikOrchestrator:
    """Pipeline orchestrator that drives the incremental Mavvrik export loop."""

    def __init__(self, exporter: "MavvrikExporter") -> None:
        self._exporter = exporter
        self._client = MavvrikClient(
            api_key=exporter.api_key or "",
            api_endpoint=exporter.api_endpoint or "",
            connection_id=exporter.connection_id or "",
        )
        self._db = MavvrikDatabase()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today() -> date:
        """Return today's date in UTC. Extracted so tests can patch it."""
        return datetime.now(timezone.utc).date()

    @property
    def _yesterday(self) -> date:
        return self._utc_today() - timedelta(days=1)

    # ------------------------------------------------------------------
    # Main entry point (called by MavvrikScheduler after acquiring lock)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Export every complete calendar day starting from the Mavvrik marker.

        Single outer try/catch — any failure is reported to Mavvrik and logged.
        """
        try:
            start_date = await self._get_start_date()
            if start_date > self._yesterday:
                verbose_logger.debug(
                    "MavvrikOrchestrator: up to date, nothing to export"
                )
                return
            await self._export_date_range(start_date)
        except Exception as exc:
            verbose_logger.error("MavvrikOrchestrator: run failed: %s", exc)
            await self._client.report_error(str(exc)[:500])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _get_start_date(self) -> date:
        """Return the start date for this export run.

        Calls register() to get the metricsMarker from Mavvrik (source of truth).
        Falls back to first-run resolution if register() fails or returns None.
        """
        marker_str: Optional[str] = None
        try:
            marker_str = await self._client.register()
            verbose_logger.debug("MavvrikOrchestrator: Mavvrik marker = %s", marker_str)
        except Exception as exc:
            verbose_logger.warning(
                "MavvrikOrchestrator: register() failed, falling back to first-run start date: %s",
                exc,
            )

        if marker_str:
            return date.fromisoformat(marker_str[:10])

        return await self._resolve_first_run_start_date()

    async def _export_date_range(self, start_date: date) -> None:
        """Export each calendar day from start_date to yesterday (inclusive).

        After each successful upload, advances the Mavvrik marker to
        (export_date + 1 day) so the next run picks up from there.
        """
        export_date = start_date
        while export_date <= self._yesterday:
            date_str = export_date.isoformat()
            verbose_logger.info("MavvrikOrchestrator: exporting date %s", date_str)

            await self._exporter.export_usage_data(
                date_str=date_str,
                limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
            )

            # metricsMarker is the start of the next export window.
            next_date = export_date + timedelta(days=1)
            next_epoch = int(
                datetime(
                    next_date.year,
                    next_date.month,
                    next_date.day,
                    tzinfo=timezone.utc,
                ).timestamp()
            )
            await self._client.advance_marker(next_epoch)
            export_date = next_date

    async def _resolve_first_run_start_date(self) -> date:
        """Determine the export start date on the very first run (no Mavvrik marker yet).

        Priority:
          1. MAVVRIK_LOOKBACK_START_DATE env var (clamped to MIN(date) in DB)
          2. MIN(date) in LiteLLM_DailyUserSpend
          3. Yesterday as a last-resort fallback
        """
        yesterday = self._yesterday

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

        earliest_str = await self._db.get_earliest_date()
        earliest_db: Optional[date] = None
        if earliest_str:
            try:
                earliest_db = date.fromisoformat(earliest_str)
            except ValueError:
                pass

        if requested_start is not None and earliest_db is not None:
            start_date = max(requested_start, earliest_db)
            verbose_logger.info(
                "MavvrikOrchestrator: no marker found, starting from %s", start_date
            )
        elif requested_start is not None:
            start_date = requested_start
            verbose_logger.info(
                "MavvrikOrchestrator: no marker found, starting from "
                "MAVVRIK_LOOKBACK_START_DATE %s",
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
