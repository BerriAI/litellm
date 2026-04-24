"""Exporter — fetch spend data from Postgres and transform to CSV.

Responsibility: extract data from LiteLLM's database and convert it to CSV.

Public interface:
  export(date_str, connection_id, limit) → (DataFrame, csv_str)
      Single entry point: fetch → serialize. Used by Service.export/dry_run.

  get_earliest_date() → Optional[str]
      Returns MIN(date) for first-run start date resolution.

Internal methods:
  _stream_pages(date_str, connection_id, page_size) → AsyncIterator[str]
  _get_usage_data(date_str, limit) → DataFrame
  _to_csv(df, connection_id) → str

DB not connected: all methods log a warning and return empty/None — never raise.
The scheduler skips the date gracefully; user-triggered endpoints surface the
missing-DB error through Settings._ensure_prisma_client() before reaching here.
"""

import io
from typing import Any, AsyncIterator, List, Optional, Tuple

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

_EARLIEST_DATE_QUERY = 'SELECT MIN(date) AS earliest FROM "LiteLLM_DailyUserSpend"'


class Exporter:
    """Fetch LiteLLM spend data from Postgres and transform to CSV."""

    # ------------------------------------------------------------------
    # DB access helper — returns None when DB not connected (never raises)
    # ------------------------------------------------------------------

    @property
    def _prisma_client(self):
        try:
            from litellm.proxy.proxy_server import prisma_client

            return prisma_client  # may be None if DB not yet connected
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def export(
        self,
        date_str: str,
        connection_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pl.DataFrame, str]:
        """Fetch and serialize spend data for one calendar date.

        All rows are exported — including failed requests. Mavvrik decides
        what to do with them on the ingestion side.

        Returns (df, csv_str). Returns (empty DataFrame, "") when no data or no DB.
        """
        df = await self._get_usage_data(date_str=date_str, limit=limit)
        csv = self._to_csv(df, connection_id=connection_id)
        return df, csv

    async def _stream_pages(
        self,
        date_str: str,
        connection_id: Optional[str] = None,
        page_size: int = 10_000,
    ) -> AsyncIterator[str]:
        """Yield CSV text in pages — one page of rows at a time (header on first page).

        Uses LIMIT/OFFSET pagination so only page_size rows are in memory at once.
        All rows exported — including failed requests.
        Yields nothing when DB is not connected or no rows exist for the date.
        """
        client = self._prisma_client
        if client is None:
            verbose_proxy_logger.warning(
                "Exporter: database not connected, skipping stream for %s", date_str
            )
            return

        header_written = False
        offset = 0

        while True:
            rows = await client.db.query_raw(
                _USAGE_QUERY + " LIMIT $2 OFFSET $3",
                date_str,
                page_size,
                offset,
            )

            if not rows:
                break

            df = pl.DataFrame(rows, infer_schema_length=None)

            buf = io.StringIO()
            if not header_written:
                if connection_id:
                    df = df.with_columns(pl.lit(connection_id).alias("connection_id"))
                df.write_csv(buf)
                header_written = True
            else:
                if connection_id:
                    df = df.with_columns(pl.lit(connection_id).alias("connection_id"))
                df.write_csv(buf, include_header=False)

            yield buf.getvalue()

            offset += page_size
            if len(rows) < page_size:
                break

    async def get_earliest_date(self) -> Optional[str]:
        """Return MIN(date) from LiteLLM_DailyUserSpend, or None.

        Returns None when DB is not connected — caller treats it as "no history".
        """
        client = self._prisma_client
        if client is None:
            verbose_proxy_logger.warning(
                "Exporter: database not connected, cannot determine earliest date"
            )
            return None

        rows = await client.db.query_raw(_EARLIEST_DATE_QUERY)
        if rows and rows[0].get("earliest") is not None:
            return str(rows[0]["earliest"])[:10]
        return None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    async def _get_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Retrieve all spend rows for a single calendar date.

        Returns empty DataFrame when DB is not connected.
        """
        client = self._prisma_client
        if client is None:
            verbose_proxy_logger.warning(
                "Exporter: database not connected, returning empty data for %s",
                date_str,
            )
            return pl.DataFrame()

        query = _USAGE_QUERY
        params: List[Any] = [date_str]

        if limit is not None:
            params.append(int(limit))
            query += " LIMIT $2"

        db_response = await client.db.query_raw(query, *params)
        return pl.DataFrame(db_response, infer_schema_length=None)

    def _to_csv(self, df: pl.DataFrame, connection_id: Optional[str] = None) -> str:
        """Serialize a DataFrame to CSV, adding connection_id column if provided."""
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
