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
        target = frame if not frame.is_empty() else pl.DataFrame(schema=frame.schema)
        buffer = io.BytesIO()
        target.write_csv(buffer)
        return buffer.getvalue()
