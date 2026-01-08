"""Database access helpers for Focus export."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import polars as pl


class FocusLiteLLMDatabase:
    """Retrieves LiteLLM usage data for Focus export workflows."""

    async def get_usage_data(
        self,
        *,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Return usage data for the requested window."""
        raise NotImplementedError
