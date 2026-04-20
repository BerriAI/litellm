"""MavvrikExporter — CustomLogger subclass for the Mavvrik cost-data integration.

Export flow (called externally by scheduler or proxy endpoints):

  1. Query MavvrikDatabase for rows on a given date.
  2. Transform rows to CSV via MavvrikTransformer.
  3. GZIP-compress the CSV in memory.
  4. Upload via MavvrikClient (3-step signed URL upload).

Environment variables (fallback when DB settings are absent):
    MAVVRIK_API_KEY              x-api-key sent to Mavvrik API
    MAVVRIK_API_ENDPOINT         Mavvrik API base URL (includes tenant)
    MAVVRIK_CONNECTION_ID        Connection/instance ID
"""

import os
from datetime import timedelta, timezone
from datetime import datetime as _dt
from typing import Any, Optional

import polars as pl

from litellm._logging import verbose_logger
from litellm.constants import MAVVRIK_MAX_FETCHED_DATA_RECORDS
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.mavvrik.client import MavvrikClient
from litellm.integrations.mavvrik.database import MavvrikDatabase
from litellm.integrations.mavvrik.transform import MavvrikTransformer


class MavvrikExporter(CustomLogger):
    """Export LiteLLM spend data to Mavvrik via signed URL uploads."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        connection_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key or os.getenv("MAVVRIK_API_KEY")
        self._api_endpoint = api_endpoint or os.getenv("MAVVRIK_API_ENDPOINT", "")
        self._connection_id = connection_id or os.getenv("MAVVRIK_CONNECTION_ID", "")

        verbose_logger.debug(
            "MavvrikExporter initialised: endpoint=%s connection_id=%s",
            self._api_endpoint,
            self._connection_id,
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key

    @property
    def api_endpoint(self) -> str:
        return self._api_endpoint

    @property
    def connection_id(self) -> str:
        return self._connection_id

    @property
    def is_valid(self) -> bool:
        """True when all required config fields are present."""
        return bool(self._api_key and self._api_endpoint and self._connection_id)

    @property
    def _client(self) -> MavvrikClient:
        """Build and return a MavvrikClient instance."""
        return MavvrikClient(
            api_key=self._api_key or "",
            api_endpoint=self._api_endpoint or "",
            connection_id=self._connection_id or "",
        )

    # ------------------------------------------------------------------
    # Core export
    # ------------------------------------------------------------------

    async def export_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ) -> int:
        """Query → transform → upload for a single calendar date (YYYY-MM-DD).

        Re-uploading the same date overwrites the previous upload — idempotent.
        Called by the scheduler and the manual /mavvrik/export endpoint.

        Returns:
            Number of records exported (0 if no data).
        """
        self._validate_config()

        verbose_logger.debug("MavvrikExporter: exporting date %s", date_str)

        db = MavvrikDatabase()
        df = await db.get_usage_data(date_str=date_str, limit=limit)

        if df.is_empty():
            verbose_logger.debug(
                "MavvrikExporter: no spend data for %s, nothing to upload", date_str
            )
            return 0

        verbose_logger.debug(
            "MavvrikExporter: %d rows fetched, transforming…", len(df)
        )

        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df, connection_id=self._connection_id)

        if not csv_payload:
            verbose_logger.debug(
                "MavvrikExporter: 0 rows after transform for %s, skipping upload",
                date_str,
            )
            return 0

        # Count rows actually exported (post-filter, matches what was uploaded).
        records_exported = csv_payload.count("\n") - 1  # subtract header row

        client = self._client
        await client.upload(csv_payload, date_str=date_str)

        verbose_logger.info(
            "MavvrikExporter: uploaded %d records (%d CSV bytes) for date %s",
            records_exported,
            len(csv_payload),
            date_str,
        )
        return records_exported

    # ------------------------------------------------------------------
    # Dry run (preview without uploading)
    # ------------------------------------------------------------------

    async def dry_run(
        self,
        date_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Return transformed records as dicts without uploading — for /mavvrik/dry-run."""
        if not date_str:
            date_str = (
                _dt.now(timezone.utc).date() - timedelta(days=1)
            ).isoformat()

        db = MavvrikDatabase()
        df = await db.get_usage_data(
            date_str=date_str,
            limit=limit or MAVVRIK_MAX_FETCHED_DATA_RECORDS,
        )

        if df.is_empty():
            return {
                "usage_data": [],
                "csv_preview": "",
                "summary": {
                    "total_records": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "unique_models": 0,
                    "unique_teams": 0,
                },
            }

        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df)

        # Apply same filter as to_csv (successful_requests > 0) so preview and
        # summary stats match what would actually be uploaded.
        if "successful_requests" in df.columns:
            df = df.filter(pl.col("successful_requests") > 0)

        usage_sample = df.head(50).to_dicts()
        total_cost = float(df["spend"].sum()) if "spend" in df.columns else 0.0
        total_tokens = (
            int((df["prompt_tokens"].sum() or 0) + (df["completion_tokens"].sum() or 0))
            if "prompt_tokens" in df.columns
            else 0
        )
        unique_models = df["model"].n_unique() if "model" in df.columns else 0
        unique_teams = df["team_id"].n_unique() if "team_id" in df.columns else 0

        return {
            "usage_data": usage_sample,
            "csv_preview": csv_payload[:5000] if csv_payload else "",
            "summary": {
                "total_records": len(df),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "unique_models": unique_models,
                "unique_teams": unique_teams,
            },
        }

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        missing = [
            name
            for name, val in [
                ("api_key", self._api_key),
                ("api_endpoint", self._api_endpoint),
                ("connection_id", self._connection_id),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"MavvrikExporter: missing required config fields: {missing}. "
                "Set via /mavvrik/init or MAVVRIK_* environment variables."
            )
