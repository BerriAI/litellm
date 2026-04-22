"""Exporter — fetch spend data from Postgres and transform to CSV.

Handles the "export" side of the pipeline: extracting data from LiteLLM's
database and converting it into a CSV string ready for upload.

SQL queries:
  get_usage_data()   — 4-table LEFT JOIN on LiteLLM_DailyUserSpend
  get_earliest_date() — MIN(date) for first-run start date resolution

Transform:
  to_csv()  — filter rows, add connection_id, serialise to CSV string
"""

import io
from typing import Any, List, Optional

import polars as pl

from litellm._logging import verbose_proxy_logger

# query_raw is used here instead of Prisma model methods because the query
# requires a 4-table LEFT JOIN (DailyUserSpend → VerificationToken →
# TeamTable → UserTable). Prisma's relational API cannot express a multi-hop
# JOIN in a single query without N+1 round-trips.
#
# dus.* selects all columns from LiteLLM_DailyUserSpend so that any new
# columns added to that table in future LiteLLM versions are automatically
# included in the export without requiring a code change here.
# Only specific non-overlapping columns are selected from the JOIN tables to
# avoid ambiguity (spend, user_id, team_id, created_at etc. exist in multiple
# tables and cannot be selected via wildcards).
_USAGE_QUERY = """
SELECT
    dus.*,
    vt.team_id,
    vt.key_alias    AS api_key_alias,
    vt.organization_id,
    tt.team_alias,
    ut.user_email,
    ut.user_alias
FROM "LiteLLM_DailyUserSpend" dus
LEFT JOIN "LiteLLM_VerificationToken" vt  ON dus.api_key   = vt.token
LEFT JOIN "LiteLLM_TeamTable"         tt  ON vt.team_id    = tt.team_id
LEFT JOIN "LiteLLM_UserTable"         ut  ON dus.user_id   = ut.user_id
WHERE dus.date = $1
ORDER BY dus.date, dus.user_id, dus.model ASC
"""

# Use SQL MIN() rather than Prisma find_first(order={"date": "asc"}) because
# date is stored as a STRING column (YYYY-MM-DD). String-sort is equivalent
# for well-formed dates but MIN() in SQL is authoritative and handles NULLs.
_EARLIEST_DATE_QUERY = 'SELECT MIN(date) AS earliest FROM "LiteLLM_DailyUserSpend"'


class Exporter:
    """Fetch LiteLLM spend data from Postgres and transform to CSV."""

    # ------------------------------------------------------------------
    # Database access
    # ------------------------------------------------------------------

    @property
    def _prisma_client(self):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise RuntimeError(
                "Database not connected. Connect a database to your proxy — "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    async def get_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Retrieve spend rows for a single calendar date (YYYY-MM-DD).

        Filters by dus.date so each export covers exactly one complete day.
        """
        client = self._prisma_client

        query = _USAGE_QUERY
        params: List[Any] = [date_str]

        if limit is not None:
            params.append(int(limit))
            query += " LIMIT $2"

        db_response = await client.db.query_raw(query, *params)
        return pl.DataFrame(db_response, infer_schema_length=None)

    async def get_earliest_date(self) -> Optional[str]:
        """Return the earliest date string (YYYY-MM-DD) in LiteLLM_DailyUserSpend, or None."""
        client = self._prisma_client
        rows = await client.db.query_raw(_EARLIEST_DATE_QUERY)
        if rows and rows[0].get("earliest") is not None:
            return str(rows[0]["earliest"])[:10]
        return None

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop rows with zero successful_requests — they have no billable output."""
        if "successful_requests" not in df.columns:
            return df
        return df.filter(pl.col("successful_requests") > 0)

    def to_csv(self, df: pl.DataFrame, connection_id: Optional[str] = None) -> str:
        """Serialise a DataFrame to CSV, adding a connection_id column if provided.

        Args:
            df: Polars DataFrame. Caller should run filter() first.
            connection_id: Optional identifier added as a column.

        Returns:
            CSV string (header + data rows). Empty string if DataFrame is empty.
        """
        if df.is_empty():
            verbose_proxy_logger.debug("Exporter: empty DataFrame, nothing to export")
            return ""

        if connection_id:
            df = df.with_columns(pl.lit(connection_id).alias("connection_id"))

        buf = io.StringIO()
        df.write_csv(buf)
        csv_str = buf.getvalue()

        verbose_proxy_logger.debug(
            "Exporter: %d rows → %d CSV bytes", len(df), len(csv_str)
        )
        return csv_str
