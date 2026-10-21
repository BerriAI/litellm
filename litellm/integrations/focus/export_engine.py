"""Core export engine for Focus integrations (heavy dependencies)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, SupportsFloat, SupportsIndex, SupportsInt, cast

import polars as pl

from litellm._logging import verbose_logger

from .database import FocusLiteLLMDatabase
from .destinations import FocusDestination, FocusDestinationFactory, FocusTimeWindow
from .serializers import FocusCsvSerializer, FocusParquetSerializer, FocusSerializer
from .transformer import FocusTransformer


class FocusExportEngine:
    """Engine that fetches, normalizes, and uploads Focus exports."""

    def __init__(
        self,
        *,
        provider: str,
        export_format: str,
        prefix: str,
        destination_config: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.export_format = export_format
        self.prefix = prefix
        self.destination = FocusDestinationFactory.create(
            provider=self.provider,
            prefix=self.prefix,
            config=destination_config,
        )
        self.serializer = self._init_serializer()
        self.transformer = FocusTransformer()
        self.database = FocusLiteLLMDatabase()

    @property
    def _destination(self) -> FocusDestination:
        return self.destination

    @_destination.setter
    def _destination(self, value: FocusDestination) -> None:
        self.destination = value

    @property
    def _serializer(self) -> FocusSerializer:
        return self.serializer

    @_serializer.setter
    def _serializer(self, value: FocusSerializer) -> None:
        self.serializer = value

    @property
    def _transformer(self) -> FocusTransformer:
        return self.transformer

    @_transformer.setter
    def _transformer(self, value: FocusTransformer) -> None:
        self.transformer = value

    @property
    def _database(self) -> FocusLiteLLMDatabase:
        return self.database

    @_database.setter
    def _database(self, value: FocusLiteLLMDatabase) -> None:
        self.database = value

    def _init_serializer(self) -> FocusSerializer:
        if self.export_format == "csv":
            return FocusCsvSerializer()
        if self.export_format == "parquet":
            return FocusParquetSerializer()
        raise NotImplementedError(f"Export format '{self.export_format}' not supported. Use 'parquet' or 'csv'.")

    async def dry_run_export_usage_data(self, limit: int | None) -> dict[str, Any]:
        data: Final = await self.database.get_usage_data(limit=limit)
        normalized: Final = self.transformer.transform(data)

        usage_sample: Final = data.head(min(50, len(data))).to_dicts()
        normalized_sample: Final = normalized.head(min(50, len(normalized))).to_dicts()

        summary: Final = {
            "total_records": len(normalized),
            "total_spend": self.sum_column(data, "spend"),
            "total_tokens": self.sum_column(data, "total_tokens"),
            "unique_teams": self.count_unique(data, "team_id"),
            "unique_models": self.count_unique(data, "model"),
        }

        return {
            "usage_data": usage_sample,
            "normalized_data": normalized_sample,
            "summary": summary,
        }

    async def export_all(
        self,
        *,
        limit: int | None,
    ) -> None:
        """Export all available data without time-window filtering."""
        data: Final = await self.database.get_usage_data(limit=limit)
        if data.is_empty():
            verbose_logger.debug("Focus export: no usage data available")
            return

        normalized: Final = self.transformer.transform(data)
        if normalized.is_empty():
            verbose_logger.debug("Focus export: normalized data empty")
            return

        # Build a window spanning the full data range for the filename
        from datetime import datetime, timezone

        now: Final = datetime.now(timezone.utc)
        window: Final = FocusTimeWindow(
            start_time=now.replace(hour=0, minute=0, second=0, microsecond=0),
            end_time=now,
            frequency="all",
        )
        await self._serialize_and_upload(normalized, window)

    async def export_window(
        self,
        *,
        window: FocusTimeWindow,
        limit: int | None,
    ) -> None:
        data: Final = await self.database.get_usage_data(
            limit=limit,
            start_time_utc=window.start_time,
            end_time_utc=window.end_time,
        )
        if data.is_empty():
            verbose_logger.debug("Focus export: no usage data for window %s", window)
            return

        normalized: Final = self.transformer.transform(data)
        if normalized.is_empty():
            verbose_logger.debug("Focus export: normalized data empty for window %s", window)
            return

        await self._serialize_and_upload(normalized, window)

    async def _serialize_and_upload(self, frame: pl.DataFrame, window: FocusTimeWindow) -> None:
        payload: Final = self.serializer.serialize(frame)
        if not payload:
            verbose_logger.debug("Focus export: serializer returned empty payload")
            return
        await self.destination.deliver(
            content=payload,
            time_window=window,
            filename=self.build_filename(window),
        )

    def build_filename(self, window: FocusTimeWindow) -> str:
        if not self.serializer.extension:
            raise ValueError("Serializer must declare a file extension")
        # Include time window in filename so Vantage (which deduplicates
        # by filename) doesn't overwrite previous uploads.
        start_str: Final = window.start_time.strftime("%Y%m%dT%H%M%SZ")
        end_str: Final = window.end_time.strftime("%Y%m%dT%H%M%SZ")
        return f"usage_{start_str}_{end_str}.{self.serializer.extension}"

    _build_filename = build_filename

    @staticmethod
    def sum_column(frame: pl.DataFrame, column: str) -> float:
        if frame.is_empty() or column not in frame.columns:
            return 0.0
        value: Final[object] = frame.select(pl.col(column).sum().alias("sum")).row(0)[0]
        if value is None:
            return 0.0
        return float(cast(SupportsFloat | SupportsIndex | str | bytes | bytearray, value))

    _sum_column: Final[Callable[[pl.DataFrame, str], float]] = sum_column

    @staticmethod
    def count_unique(frame: pl.DataFrame, column: str) -> int:
        if frame.is_empty() or column not in frame.columns:
            return 0
        value: Final[object] = frame.select(pl.col(column).n_unique().alias("unique")).row(0)[0]
        if value is None:
            return 0
        return int(cast(SupportsInt | SupportsIndex | str | bytes | bytearray, value))

    _count_unique: Final[Callable[[pl.DataFrame, str], int]] = count_unique
