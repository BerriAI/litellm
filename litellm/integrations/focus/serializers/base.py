"""Serializer abstractions for Focus export."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class FocusSerializer(ABC):
    """Base serializer turning Focus frames into bytes."""

    extension: str = ""

    @abstractmethod
    def serialize(self, frame: pl.DataFrame) -> bytes:
        """Convert the normalized Focus frame into the chosen format."""
        raise NotImplementedError
