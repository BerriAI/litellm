"""Focus export logger orchestrating DB pull/transform/upload."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import polars as pl

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

from .database import FocusLiteLLMDatabase
from .destinations import (
    FocusDestination,
    FocusDestinationFactory,
    FocusTimeWindow,
)
from .serializers import FocusParquetSerializer, FocusSerializer
from .transformer import FocusTransformer

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any

FOCUS_USAGE_DATA_JOB_NAME = "focus_export_usage_data"
DEFAULT_DRY_RUN_LIMIT = 500


class FocusLogger(CustomLogger):
    """Coordinates Focus export jobs across transformer/serializer/destination layers."""

    def __init__(
        self,
        *,
        provider: Optional[str] = None,
        export_format: Optional[str] = None,
        frequency: Optional[str] = None,
        cron_offset_minute: Optional[int] = None,
        interval_seconds: Optional[int] = None,
        prefix: Optional[str] = None,
        destination_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = (provider or os.getenv("FOCUS_PROVIDER") or "s3").lower()
        self.export_format = (
            export_format or os.getenv("FOCUS_FORMAT") or "parquet"
        ).lower()
        self.frequency = (
            frequency or os.getenv("FOCUS_FREQUENCY") or "hourly"
        ).lower()
        self.cron_offset_minute = (
            cron_offset_minute
            if cron_offset_minute is not None
            else int(os.getenv("FOCUS_CRON_OFFSET", "5"))
        )
        raw_interval = (
            interval_seconds
            if interval_seconds is not None
            else os.getenv("FOCUS_INTERVAL_SECONDS")
        )
        self.interval_seconds = int(raw_interval) if raw_interval is not None else None
        self.prefix = prefix or os.getenv("FOCUS_PREFIX", "focus_exports")

        self._destination = FocusDestinationFactory.create(
            provider=self.provider,
            prefix=self.prefix,
            config=destination_config,
        )
        self._serializer = self._init_serializer()
        self._transformer = FocusTransformer()
        self._database = FocusLiteLLMDatabase()

    def _init_serializer(self) -> FocusSerializer:
        """Return serializer implementation for requested format."""
        if self.export_format != "parquet":
            raise NotImplementedError("Only parquet export supported currently")
        return FocusParquetSerializer()

    async def export_usage_data(
        self,
        *,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> None:
        """Public hook to trigger export immediately."""
        if bool(start_time_utc) ^ bool(end_time_utc):
            raise ValueError("start_time_utc and end_time_utc must be provided together")

        if start_time_utc and end_time_utc:
            window = FocusTimeWindow(
                start_time=start_time_utc,
                end_time=end_time_utc,
                frequency=self.frequency,
            )
        else:
            window = self._compute_time_window(datetime.now(timezone.utc))
        await self._export_window(window=window, limit=limit)

    async def dry_run_export_usage_data(
        self, limit: Optional[int] = DEFAULT_DRY_RUN_LIMIT
    ) -> dict[str, Any]:
        """Return transformed data without uploading."""
        data = await self._database.get_usage_data(limit=limit)
        normalized = self._transformer.transform(data)

        usage_sample = data.head(min(50, len(data))).to_dicts()
        normalized_sample = normalized.head(min(50, len(normalized))).to_dicts()

        summary = {
            "total_records": len(normalized),
            "total_spend": self._sum_column(normalized, "spend"),
            "total_tokens": self._sum_column(normalized, "total_tokens"),
            "unique_teams": self._count_unique(normalized, "team_id"),
            "unique_models": self._count_unique(normalized, "model"),
        }

        return {
            "usage_data": usage_sample,
            "normalized_data": normalized_sample,
            "summary": summary,
        }

    async def initialize_focus_export_job(self) -> None:
        """Entry point for scheduler jobs to run export cycle with locking."""
        try:
            from litellm.proxy.proxy_server import proxy_logging_obj
        except ImportError:
            proxy_logging_obj = None

        pod_lock_manager = None
        if proxy_logging_obj is not None:
            writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
            if writer is not None:
                pod_lock_manager = getattr(writer, "pod_lock_manager", None)

        if pod_lock_manager and pod_lock_manager.redis_cache:
            acquired = await pod_lock_manager.acquire_lock(
                cronjob_id=FOCUS_USAGE_DATA_JOB_NAME
            )
            if not acquired:
                verbose_logger.debug("Focus export: unable to acquire pod lock")
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(
                    cronjob_id=FOCUS_USAGE_DATA_JOB_NAME
                )
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_focus_export_background_job(
        scheduler: AsyncIOScheduler,
    ) -> None:
        """Register the export cron/interval job with the provided scheduler."""
        from litellm.integrations.custom_logger import CustomLogger

        focus_loggers: List[
            CustomLogger
        ] = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=FocusLogger
        )
        if not focus_loggers:
            verbose_logger.debug("No Focus export logger registered; skipping scheduler")
            return

        focus_logger = cast(FocusLogger, focus_loggers[0])
        trigger_kwargs = focus_logger._build_scheduler_trigger()
        scheduler.add_job(
            focus_logger.initialize_focus_export_job,
            **trigger_kwargs,
        )

    def _build_scheduler_trigger(self) -> Dict[str, Any]:
        """Return scheduler configuration for the selected frequency."""
        if self.frequency == "interval":
            seconds = self.interval_seconds or 60
            return {"trigger": "interval", "seconds": seconds}

        if self.frequency == "hourly":
            minute = max(0, min(59, self.cron_offset_minute))
            return {"trigger": "cron", "minute": minute, "second": 0}

        if self.frequency == "daily":
            total_minutes = max(0, self.cron_offset_minute)
            hour = min(23, total_minutes // 60)
            minute = min(59, total_minutes % 60)
            return {"trigger": "cron", "hour": hour, "minute": minute, "second": 0}

        raise ValueError(f"Unsupported frequency: {self.frequency}")

    async def _run_scheduled_export(self) -> None:
        """Execute the scheduled export for the configured window."""
        window = self._compute_time_window(datetime.now(timezone.utc))
        await self._export_window(window=window, limit=None)

    async def _export_window(
        self,
        *,
        window: FocusTimeWindow,
        limit: Optional[int],
    ) -> None:
        data = await self._database.get_usage_data(
            limit=limit,
            start_time_utc=window.start_time,
            end_time_utc=window.end_time,
        )
        if data.is_empty():
            verbose_logger.debug("Focus export: no usage data for window %s", window)
            return

        normalized = self._transformer.transform(data)
        if normalized.is_empty():
            verbose_logger.debug("Focus export: normalized data empty for window %s", window)
            return

        await self._serialize_and_upload(normalized, window)

    def _compute_time_window(self, now: datetime) -> FocusTimeWindow:
        """Derive the time window to export based on configured frequency."""
        now_utc = now.astimezone(timezone.utc)
        if self.frequency == "hourly":
            end_time = now_utc.replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(hours=1)
        elif self.frequency == "daily":
            end_time = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(days=1)
        elif self.frequency == "interval":
            interval = timedelta(seconds=self.interval_seconds or 60)
            end_time = now_utc
            start_time = end_time - interval
        else:
            raise ValueError(f"Unsupported frequency: {self.frequency}")
        return FocusTimeWindow(
            start_time=start_time,
            end_time=end_time,
            frequency=self.frequency,
        )

    async def _serialize_and_upload(
        self, frame: pl.DataFrame, window: FocusTimeWindow
    ) -> None:
        """Serialize the normalized frame and upload via destination."""
        payload = self._serializer.serialize(frame)
        if not payload:
            verbose_logger.debug("Focus export: serializer returned empty payload")
            return
        await self._destination.deliver(
            content=payload,
            time_window=window,
            filename=self._build_filename(),
        )

    def _build_filename(self) -> str:
        """Return the canonical file name for exports."""
        if not self._serializer.extension:
            raise ValueError("Serializer must declare a file extension")
        return f"usage.{self._serializer.extension}"

    @staticmethod
    def _sum_column(frame: pl.DataFrame, column: str) -> float:
        if frame.is_empty() or column not in frame.columns:
            return 0.0
        value = frame.select(pl.col(column).sum().alias("sum")).row(0)[0]
        if value is None:
            return 0.0
        return float(value)

    @staticmethod
    def _count_unique(frame: pl.DataFrame, column: str) -> int:
        if frame.is_empty() or column not in frame.columns:
            return 0
        value = frame.select(pl.col(column).n_unique().alias("unique")).row(0)[0]
        if value is None:
            return 0
        return int(value)
