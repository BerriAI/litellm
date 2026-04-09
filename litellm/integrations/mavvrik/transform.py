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

        Rows where successful_requests == 0 are excluded — they represent
        API calls that produced no billable output and add no value to the export.

        Args:
            df: Polars DataFrame with all columns from the DB query.
            connection_id: Optional connection identifier to add as a column.

        Returns:
            CSV string (header + data rows). Empty string if DataFrame is empty
            or all rows are filtered out.
        """
        if df.is_empty():
            verbose_proxy_logger.debug(
                "Mavvrik transform: empty DataFrame, nothing to export"
            )
            return ""

        # Filter out rows with no successful requests when the column is present
        if "successful_requests" in df.columns:
            df = df.filter(pl.col("successful_requests") > 0)
            if df.is_empty():
                verbose_proxy_logger.debug(
                    "Mavvrik transform: all rows have zero successful_requests, skipping"
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
