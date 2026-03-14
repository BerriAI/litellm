"""CSV serializer for Focus export."""

from __future__ import annotations

import io

import polars as pl

from .base import FocusSerializer


class FocusCsvSerializer(FocusSerializer):
    """Serialize normalized Focus frames to CSV bytes."""

    extension = "csv"

    def serialize(self, frame: pl.DataFrame) -> bytes:
        """Encode the provided frame as a CSV payload."""
        # Cast Decimal columns to Float64 so CSV output uses standard
        # floating-point notation (e.g. "1.5") instead of fixed-point
        # strings (e.g. "1.500000") that some parsers may reject.
        decimal_cols = [
            col
            for col, dtype in zip(frame.columns, frame.dtypes)
            if isinstance(dtype, pl.Decimal)
        ]
        if decimal_cols:
            frame = frame.with_columns(
                [pl.col(c).cast(pl.Float64) for c in decimal_cols]
            )
        buffer = io.BytesIO()
        frame.write_csv(buffer)
        return buffer.getvalue()
