"""Orchestrator — pipeline sequencing: register → fetch → transform → upload → advance.

Responsibility: sequence the export steps and own the pod lock. Nothing else.

Each step is one method, one line in _run_pipeline:
  _register()  → start_date
  _fetch()     → DataFrame
  _transform() → CSV string
  _upload()    → None
  _advance()   → None

One try/except in _run_pipeline. No nested exception handling anywhere else.

Marker semantics:
  metricsMarker from register() is the START of the export window.
  After each date is uploaded, advance_marker() is called with
  (export_date + 1 day) so the next run starts from there.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Iterator

import polars as pl

from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
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

    @property
    def _yesterday(self) -> date:
        return self._utc_today() - timedelta(days=1)

    def _date_range(self, start: date) -> Iterator[date]:
        current = start
        while current <= self._yesterday:
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
            start_date = await self._register()

            if start_date > self._yesterday:
                verbose_logger.warning(
                    "Orchestrator: up to date (start=%s, yesterday=%s), nothing to export",
                    start_date,
                    self._yesterday,
                )
                return

            verbose_logger.warning(
                "Orchestrator: exporting %s → %s", start_date, self._yesterday
            )

            for export_date in self._date_range(start_date):
                df = await self._fetch(export_date)
                csv = self._transform(df, export_date)
                await self._upload(csv, export_date)
                await self._advance(export_date)

            verbose_logger.warning(
                "Orchestrator: export complete, last date=%s", self._yesterday
            )

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

    async def _fetch(self, export_date: date) -> pl.DataFrame:
        # Fetch one extra row to detect overflow without a separate COUNT query.
        df = await self._exporter.get_usage_data(
            date_str=export_date.isoformat(),
            limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS + 1,
        )
        if len(df) > MAVVRIK_MAX_FETCHED_DATA_RECORDS:
            raise RuntimeError(
                f"Date {export_date.isoformat()} has more than "
                f"{MAVVRIK_MAX_FETCHED_DATA_RECORDS} rows. "
                f"Increase MAVVRIK_MAX_FETCHED_DATA_RECORDS or implement "
                f"chunked streaming upload."
            )
        return df

    def _transform(self, df: pl.DataFrame, export_date: date) -> str:
        filtered = self._exporter.filter(df)
        return self._exporter.to_csv(filtered, connection_id=self._client.connection_id)

    async def _upload(self, csv: str, export_date: date) -> None:
        date_str = export_date.isoformat()
        await self._uploader.upload(csv, date_str=date_str)
        verbose_logger.warning("Orchestrator: %s → upload done", date_str)

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
        return self._yesterday

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
