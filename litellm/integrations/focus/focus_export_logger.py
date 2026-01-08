"""Focus export logger orchestrating DB pull/transform/upload."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import polars as pl

from litellm.integrations.custom_logger import CustomLogger

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


class FocusExportLogger(CustomLogger):
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
        self.provider = (provider or os.getenv("FOCUS_EXPORT_PROVIDER") or "s3").lower()
        self.export_format = (
            export_format or os.getenv("FOCUS_EXPORT_FORMAT") or "parquet"
        ).lower()
        self.frequency = (
            frequency or os.getenv("FOCUS_EXPORT_FREQUENCY") or "hourly"
        ).lower()
        self.cron_offset_minute = (
            cron_offset_minute
            if cron_offset_minute is not None
            else int(os.getenv("FOCUS_EXPORT_CRON_OFFSET", "5"))
        )
        self.interval_seconds = (
            interval_seconds
            if interval_seconds is not None
            else os.getenv("FOCUS_EXPORT_INTERVAL_SECONDS")
        )
        self.prefix = prefix or os.getenv("FOCUS_EXPORT_PREFIX", "focus_exports")

        self._destination = self._init_destination(
            destination_config=destination_config,
        )
        self._serializer = self._init_serializer()
        self._transformer = FocusTransformer()

    def _init_serializer(self) -> FocusSerializer:
        """Return serializer implementation for requested format."""
        if self.export_format != "parquet":
            raise NotImplementedError("Only parquet export supported currently")
        return FocusParquetSerializer()

    def _init_destination(
        self,
        *,
        destination_config: Optional[dict[str, Any]],
    ) -> FocusDestination:
        """Factory for destination implementations."""
        resolved_config = self._resolve_destination_config(destination_config)
        return FocusDestinationFactory.create(
            provider=self.provider,
            prefix=self.prefix,
            config=resolved_config,
        )

    def _resolve_destination_config(
        self,
        destination_config: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Collect provider-specific configuration for destination creation."""
        raise NotImplementedError

    async def export_usage_data(self) -> None:
        """Public hook to trigger export immediately."""
        raise NotImplementedError

    async def dry_run_export_usage_data(self) -> dict:
        """Return transformed data without uploading."""
        raise NotImplementedError

    async def initialize_focus_export_job(self) -> None:
        """Entry point for scheduler jobs to run export cycle with locking."""
        raise NotImplementedError

    @staticmethod
    async def init_focus_export_background_job(
        scheduler: AsyncIOScheduler,
    ) -> None:
        """Register the export cron/interval job with the provided scheduler."""
        raise NotImplementedError

    def _compute_time_window(self, now: datetime) -> FocusTimeWindow:
        """Derive the time window to export based on configured frequency."""
        raise NotImplementedError

    def _serialize_and_upload(
        self,
        frame: pl.DataFrame,
        window: FocusTimeWindow,
    ) -> None:
        """Helper stub for serializing and delegating to destination."""
        raise NotImplementedError

    def _build_filename(self) -> str:
        """Return the canonical file name for exports."""
        if not self._serializer.extension:
            raise ValueError("Serializer must declare a file extension")
        return f"usage.{self._serializer.extension}"
