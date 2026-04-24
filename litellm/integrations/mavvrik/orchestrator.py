"""Orchestrator — pipeline sequencing: register → export → upload → advance.

Responsibility: sequence the export steps and own the pod lock. Nothing else.

Pipeline in _run_pipeline (one line per step):
  start, end = await self._register(), self._export_end_date()
  for export_date in self._date_range(start, end):
      await self._export(export_date)   # streams DB → GCS via _stream_pages/_stream_upload
      await self._advance(export_date)

One try/except in _run_pipeline. No nested exception handling anywhere else.

Marker semantics:
  metricsMarker from register() is the START of the export window.
  After each date is uploaded, advance_marker() is called with
  (export_date + 1 day) so the next run starts from there.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Iterator

from litellm._logging import verbose_logger
from litellm.constants import MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter
from litellm.integrations.mavvrik.uploader import Uploader


class Orchestrator:
    """Sequences the incremental Mavvrik export pipeline."""

    def __init__(self, client: Client, uploader: Uploader) -> None:
        self._client = client
        self._uploader = uploader
        self._exporter = Exporter()

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_today() -> date:
        return datetime.now(timezone.utc).date()

    def _export_end_date(self) -> date:
        """Last date eligible for export (yesterday UTC — today's data is incomplete)."""
        return self._utc_today() - timedelta(days=1)

    def _date_range(self, start: date, end: date) -> Iterator[date]:
        """Yield each date from start to end (inclusive)."""
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)

    @staticmethod
    def _to_epoch(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())

    # ------------------------------------------------------------------
    # Entry point (called by APScheduler)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Acquire pod lock then run the export pipeline."""
        pod_lock = self._get_pod_lock_manager()

        if not pod_lock or not pod_lock.redis_cache:
            await self._run_pipeline()
            return

        if not await pod_lock.acquire_lock(
            cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
        ):
            verbose_logger.debug(
                "Orchestrator: pod lock not acquired — another pod is running"
            )
            return

        try:
            await self._run_pipeline()
        finally:
            await pod_lock.release_lock(cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME)

    # ------------------------------------------------------------------
    # Pipeline — one try/except, one line per step
    # ------------------------------------------------------------------

    async def _run_pipeline(self) -> None:
        try:
            start = await self._register()
            end = self._export_end_date()

            if start > end:
                verbose_logger.warning(
                    "Orchestrator: up to date (start=%s, end=%s), nothing to export",
                    start,
                    end,
                )
                return

            verbose_logger.warning("Orchestrator: exporting %s → %s", start, end)

            for export_date in self._date_range(start, end):
                total_bytes = await self._export(export_date)
                if total_bytes > 0:
                    await self._advance(export_date)
                else:
                    # DB unavailable or no data — raise so the marker does NOT advance.
                    # Next run will retry this date from the same marker position.
                    raise RuntimeError(
                        f"No data streamed for {export_date.isoformat()} "
                        f"— DB may be unavailable; marker not advanced"
                    )

            verbose_logger.warning("Orchestrator: export complete, last date=%s", end)

        except Exception as exc:
            verbose_logger.error(
                "Orchestrator: pipeline failed: %s", exc, exc_info=True
            )
            await self._client.report_error(str(exc)[:500])

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    async def _register(self) -> date:
        marker_str = await self._client.register()
        verbose_logger.warning("Orchestrator: marker from Mavvrik API = %s", marker_str)

        if marker_str:
            return date.fromisoformat(marker_str[:10])

        return await self._resolve_first_run_start_date()

    async def _export(self, export_date: date) -> int:
        """Stream spend data from DB to GCS for one date.

        Uses Exporter._stream_pages() → Uploader._stream_upload() so only
        one page of rows is in memory at a time. No row limit or overflow check.

        Returns total compressed bytes uploaded (0 when no data for the date).
        """
        date_str = export_date.isoformat()
        pages = self._exporter._stream_pages(
            date_str=date_str,
            connection_id=self._client.connection_id,
        )
        total_bytes = await self._uploader._stream_upload(pages, date_str=date_str)
        if total_bytes > 0:
            verbose_logger.warning(
                "Orchestrator: %s → streamed %d bytes to GCS ✓", date_str, total_bytes
            )
        else:
            verbose_logger.warning("Orchestrator: %s → no data, skipped", date_str)
        return total_bytes

    async def _advance(self, export_date: date) -> None:
        next_date = export_date + timedelta(days=1)
        await self._client.advance_marker(self._to_epoch(next_date))

    # ------------------------------------------------------------------
    # First-run start date
    # ------------------------------------------------------------------

    async def _resolve_first_run_start_date(self) -> date:
        earliest_str = await self._exporter.get_earliest_date()
        if earliest_str:
            try:
                earliest = date.fromisoformat(earliest_str)
                verbose_logger.warning(
                    "Orchestrator: no marker, starting from earliest DB date %s",
                    earliest,
                )
                return earliest
            except ValueError:
                pass
        return self._export_end_date()

    # ------------------------------------------------------------------
    # Infrastructure
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pod_lock_manager():
        from litellm.proxy.proxy_server import proxy_logging_obj

        if proxy_logging_obj is None:
            return None
        writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
        if writer is None:
            return None
        return getattr(writer, "pod_lock_manager", None)
