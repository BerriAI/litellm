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

    def to_csv(self, df: pl.DataFrame, connection_id: str | None = None) -> str:
        """Return the DataFrame as a CSV string with connection_id added.

        Args:
            df: Polars DataFrame with all columns from the DB query.
            connection_id: Optional instance identifier to add as a column.

        Returns:
            CSV string (header + data rows). Empty string if DataFrame is empty.
        """
        if df.is_empty():
            verbose_proxy_logger.debug(
                "Mavvrik transform: empty DataFrame, nothing to export"
            )
            return ""

        # Add connection_id column if provided
        if connection_id:
            df = df.with_columns(pl.lit(connection_id).alias("connection_id"))

        buf = io.StringIO()
        df.write_csv(buf)
        csv_str = buf.getvalue()

        verbose_proxy_logger.debug(
            "Mavvrik transform: %d rows → %d CSV bytes", len(df), len(csv_str)
        )
        return csv_str
