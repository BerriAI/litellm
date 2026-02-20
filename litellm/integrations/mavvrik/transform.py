"""Transform layer for the Mavvrik integration.

Converts a Polars DataFrame of LiteLLM_DailyUserSpend rows (plus JOIN columns)
directly to a CSV string. All columns returned by the SQL query are included
as-is — no schema mapping, no NDJSON wrapping.

All rows are exported for visualization and analytics purposes.
"""

import io

import polars as pl

from litellm._logging import verbose_proxy_logger


class MavvrikTransformer:
    """Transform LiteLLM spend data into a CSV string for upload."""

    def to_csv(self, df: pl.DataFrame) -> str:
        """Return the DataFrame as a CSV string.

        Args:
            df: Polars DataFrame with all columns from the DB query.

        Returns:
            CSV string (header + data rows). Empty string if DataFrame is empty.
        """
        if df.is_empty():
            verbose_proxy_logger.debug(
                "Mavvrik transform: empty DataFrame, nothing to export"
            )
            return ""

        buf = io.StringIO()
        df.write_csv(buf)
        csv_str = buf.getvalue()

        verbose_proxy_logger.debug(
            "Mavvrik transform: %d rows → %d CSV bytes", len(df), len(csv_str)
        )
        return csv_str
