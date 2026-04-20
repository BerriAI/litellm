"""Orchestrator — export pipeline: register → fetch → upload → advance.

Owns all business orchestration for the Mavvrik export:
  - Acquiring/releasing the pod lock (Redis-backed, multi-pod safe)
  - Registering with the Mavvrik API and resolving the export start date
  - Fetching usage data from the local database
  - Uploading the data to Mavvrik via signed URL
  - Advancing the remote marker after each successful upload
  - Reporting errors to Mavvrik for visibility

Marker semantics:
  metricsMarker returned by register() is the START of the window to export,
  not the last exported date. After uploading a date, advance_marker() is
  called with (export_date + 1 day) so the next run starts from there.
"""

from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Iterator, Optional

from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
    MAVVRIK_LOOKBACK_START_DATE,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter

if TYPE_CHECKING:
    from litellm.integrations.mavvrik.uploader import Uploader
else:
    Uploader = Any


class Orchestrator:
    """Pipeline orchestrator that drives the incremental Mavvrik export loop.

    The run() method reads as a high-level pipeline:

        pod lock  →  register  →  for each date:  upload  →  advance marker
    """

    def __init__(self, uploader: "Uploader") -> None:
        self._uploader = uploader
        self._client = Client(
            api_key=uploader.api_key or "",
            api_endpoint=uploader.api_endpoint or "",
            connection_id=uploader.connection_id or "",
        )
        self._exporter = Exporter()

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

    def _date_range(self, start: date) -> Iterator[date]:
        """Yield each date from start to yesterday (inclusive)."""
        current = start
        yesterday = self._yesterday
        while current <= yesterday:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _to_epoch(d: date) -> int:
        """Convert a date to a UTC epoch timestamp (midnight)."""
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    # ------------------------------------------------------------------
    # Main entry point (called by APScheduler)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Acquire the pod lock, then execute the full export pipeline.

        Pipeline:
          1. Register with Mavvrik → resolve start date
          2. For each date from start to yesterday:
             a. Upload usage data (fetch → transform → upload)
             b. Advance the remote marker
        """
        pod_lock_manager = self._get_pod_lock_manager()

        # No Redis — run directly (single-node deployments don't need a pod lock).
        if not pod_lock_manager or not pod_lock_manager.redis_cache:
            await self._run_pipeline()
            return

        if not await pod_lock_manager.acquire_lock(
            cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
        ):
            verbose_logger.debug(
                "Orchestrator: pod lock not acquired — another pod is running"
            )
            return

        try:
            await self._run_pipeline()
        finally:
            await pod_lock_manager.release_lock(
                cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
            )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def _run_pipeline(self) -> None:
        """Execute the export pipeline (called after pod lock is acquired)."""
        try:
            start_date = await self._register()

            if start_date > self._yesterday:
                verbose_logger.debug("Orchestrator: up to date, nothing to export")
                return

            for export_date in self._date_range(start_date):
                await self._upload_date(export_date)
                await self._advance_marker(export_date)

        except Exception as exc:
            verbose_logger.error("Orchestrator: run failed: %s", exc)
            await self._client.report_error(str(exc)[:500])

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _register(self) -> date:
        """Step 1: Register with Mavvrik and return the export start date.

        Calls client.register() to get the metricsMarker (source of truth).
        Falls back to first-run resolution when register() returns None
        (brand-new connection, no marker yet).
        """
        marker_str = await self._client.register()
        verbose_logger.debug("Orchestrator: marker = %s", marker_str)

        if marker_str:
            return date.fromisoformat(marker_str[:10])

        return await self._resolve_first_run_start_date()

    async def _upload_date(self, export_date: date) -> None:
        """Step 2: Fetch usage data, transform to CSV, and upload to Mavvrik."""
        date_str = export_date.isoformat()
        verbose_logger.info("Orchestrator: uploading date %s", date_str)

        await self._uploader.upload_usage_data(
            date_str=date_str,
            limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
        )

    async def _advance_marker(self, exported_date: date) -> None:
        """Step 3: Advance the Mavvrik marker to the next day.

        metricsMarker is the start of the next export window, so we set it
        to (exported_date + 1 day).
        """
        next_date = exported_date + timedelta(days=1)
        await self._client.advance_marker(self._to_epoch(next_date))

    # ------------------------------------------------------------------
    # First-run start date resolution
    # ------------------------------------------------------------------

    async def _resolve_first_run_start_date(self) -> date:
        """Determine the export start date on the very first run (no Mavvrik marker yet).

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
                    "Orchestrator: invalid MAVVRIK_LOOKBACK_START_DATE '%s' "
                    "(expected YYYY-MM-DD), falling back to earliest DB date",
                    MAVVRIK_LOOKBACK_START_DATE,
                )

        earliest_str = await self._exporter.get_earliest_date()
        earliest_db: Optional[date] = None
        if earliest_str:
            try:
                earliest_db = date.fromisoformat(earliest_str)
            except ValueError:
                pass

        if requested_start is not None and earliest_db is not None:
            start_date = max(requested_start, earliest_db)
            verbose_logger.info(
                "Orchestrator: no marker found, starting from %s", start_date
            )
            return start_date

        if requested_start is not None:
            verbose_logger.info(
                "Orchestrator: no marker found, starting from "
                "MAVVRIK_LOOKBACK_START_DATE %s",
                requested_start,
            )
            return requested_start

        if earliest_db is not None:
            verbose_logger.info(
                "Orchestrator: no marker found, starting from earliest DB date %s",
                earliest_db,
            )
            return earliest_db

        return self._yesterday

    # ------------------------------------------------------------------
    # Infrastructure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pod_lock_manager():
        """Return the pod lock manager from the proxy, or None if unavailable."""
        from litellm.proxy.proxy_server import proxy_logging_obj

        if proxy_logging_obj is None:
            return None
        writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
        if writer is None:
            return None
        return getattr(writer, "pod_lock_manager", None)
