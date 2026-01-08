"""Focus export data transformer."""

from __future__ import annotations

import polars as pl


class FocusTransformer:
    """Transforms LiteLLM DB rows into Focus-compatible schema."""

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Return a normalized frame expected by downstream serializers."""
        raise NotImplementedError
