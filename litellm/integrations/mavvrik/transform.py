"""Transform layer for the Mavvrik integration.

Converts a Polars DataFrame of LiteLLM_DailyUserSpend rows (plus JOIN columns)
directly to a CSV string. All columns returned by the SQL query are included
as-is — no schema mapping, no NDJSON wrapping.

Rows where successful_requests == 0 are filtered out (no billable activity).
"""

import io

import polars as pl

from litellm._logging import verbose_proxy_logger


class MavvrikTransformer:
    """Transform LiteLLM spend data into a CSV string for upload."""

    def to_csv(self, df: pl.DataFrame) -> str:
        """Filter zero-request rows and return the DataFrame as a CSV string.

        Args:
            df: Polars DataFrame with all columns from the DB query.

        Returns:
            CSV string (header + data rows). Empty string if no rows remain.
        """
        if df.is_empty():
            verbose_proxy_logger.debug(
                "Mavvrik transform: empty DataFrame, nothing to export"
            )
            return ""

        # Drop rows with zero successful requests — nothing to bill
        if "successful_requests" in df.columns:
            df = df.filter(pl.col("successful_requests") > 0)

        if df.is_empty():
            verbose_proxy_logger.debug(
                "Mavvrik transform: all rows had 0 successful_requests, skipping"
            )
            return ""

        buf = io.StringIO()
        df.write_csv(buf)
        csv_str = buf.getvalue()

        verbose_proxy_logger.debug(
            "Mavvrik transform: %d rows → %d CSV bytes", len(df), len(csv_str)
        )
        return csv_str
