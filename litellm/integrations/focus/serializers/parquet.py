"""Parquet serializer for Focus export."""

from __future__ import annotations

import polars as pl

from .base import FocusSerializer


class FocusParquetSerializer(FocusSerializer):
    """Placeholder Parquet serializer implementation."""

    extension = "parquet"

    def serialize(self, frame: pl.DataFrame) -> bytes:
        raise NotImplementedError
