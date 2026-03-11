"""Core export engine for Focus integrations (heavy dependencies)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import polars as pl

from litellm._logging import verbose_logger

from .database import FocusLiteLLMDatabase
from .destinations import FocusDestinationFactory, FocusTimeWindow
from .serializers import FocusParquetSerializer, FocusSerializer
from .transformer import FocusTransformer


class FocusExportEngine:
    """Engine that fetches, normalizes, and uploads Focus exports."""

    def __init__(
        self,
        *,
        provider: str,
        export_format: str,
        prefix: str,
        destination_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.provider = provider
        self.export_format = export_format
        self.prefix = prefix
        self._destination = FocusDestinationFactory.create(
            provider=self.provider,
            prefix=self.prefix,
            config=destination_config,
        )
        self._serializer = self._init_serializer()
        self._transformer = FocusTransformer()
        self._database = FocusLiteLLMDatabase()

    def _init_serializer(self) -> FocusSerializer:
        if self.export_format != "parquet":
            raise NotImplementedError("Only parquet export supported currently")
        return FocusParquetSerializer()

    async def dry_run_export_usage_data(self, limit: Optional[int]) -> Dict[str, Any]:
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

    async def export_window(
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
            verbose_logger.debug(
                "Focus export: normalized data empty for window %s", window
            )
            return

        await self._serialize_and_upload(normalized, window)

    async def _serialize_and_upload(
        self, frame: pl.DataFrame, window: FocusTimeWindow
    ) -> None:
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
