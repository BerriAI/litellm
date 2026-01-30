"""Parquet serializer for Focus export."""

from __future__ import annotations

import io

import polars as pl

from .base import FocusSerializer


class FocusParquetSerializer(FocusSerializer):
    """Serialize normalized Focus frames to Parquet bytes."""

    extension = "parquet"

    def serialize(self, frame: pl.DataFrame) -> bytes:
        """Encode the provided frame as a parquet payload."""
        target = frame if not frame.is_empty() else pl.DataFrame(schema=frame.schema)
        buffer = io.BytesIO()
        target.write_parquet(buffer, compression="snappy")
        return buffer.getvalue()
