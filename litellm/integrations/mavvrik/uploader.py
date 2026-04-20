"""Uploader — export from Postgres, then upload to Mavvrik.

Upload flow (called by Orchestrator or manual endpoints):

  1. Export: query DB + transform to CSV via Exporter.
  2. Upload: send gzipped CSV to Mavvrik via Client.

Environment variables (fallback when DB settings are absent):
    MAVVRIK_API_KEY              x-api-key sent to Mavvrik API
    MAVVRIK_API_ENDPOINT         Mavvrik API base URL (includes tenant)
    MAVVRIK_CONNECTION_ID        Connection/instance ID
"""

import os
from datetime import timedelta, timezone
from datetime import datetime as _dt
from typing import Optional

import polars as pl

from litellm._logging import verbose_logger
from litellm.constants import MAVVRIK_MAX_FETCHED_DATA_RECORDS
from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.exporter import Exporter


class Uploader:
    """Fetch LiteLLM spend data, transform to CSV, and upload to Mavvrik."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        connection_id: Optional[str] = None,
    ):
        self._api_key = api_key or os.getenv("MAVVRIK_API_KEY")
        self._api_endpoint = api_endpoint or os.getenv("MAVVRIK_API_ENDPOINT", "")
        self._connection_id = connection_id or os.getenv("MAVVRIK_CONNECTION_ID", "")
        self._mavvrik_client = Client(
            api_key=self._api_key or "",
            api_endpoint=self._api_endpoint or "",
            connection_id=self._connection_id or "",
        )

        verbose_logger.debug(
            "Uploader initialised: endpoint=%s connection_id=%s",
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

    # ------------------------------------------------------------------
    # Core upload
    # ------------------------------------------------------------------

    async def upload_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ) -> int:
        """Query → transform → upload for a single calendar date (YYYY-MM-DD).

        Re-uploading the same date overwrites the previous upload — idempotent.
        Called by the orchestrator and the manual /mavvrik/export endpoint.

        Returns:
            Number of records uploaded (0 if no data).
        """
        self._validate_config()

        verbose_logger.debug("Uploader: uploading date %s", date_str)

        exporter = Exporter()
        df = await exporter.get_usage_data(date_str=date_str, limit=limit)

        if df.is_empty():
            verbose_logger.debug(
                "Uploader: no spend data for %s, nothing to upload", date_str
            )
            return 0

        verbose_logger.debug("Uploader: %d rows fetched, transforming…", len(df))

        csv_payload = exporter.to_csv(df, connection_id=self._connection_id)

        if not csv_payload:
            verbose_logger.debug(
                "Uploader: 0 rows after filter for %s, skipping upload",
                date_str,
            )
            return 0

        # Count rows actually uploaded (post-filter, matches CSV content).
        records_uploaded = csv_payload.count("\n") - 1  # subtract header row

        await self._mavvrik_client.upload(csv_payload, date_str=date_str)

        verbose_logger.info(
            "Uploader: uploaded %d records (%d CSV bytes) for date %s",
            records_uploaded,
            len(csv_payload),
            date_str,
        )
        return records_uploaded

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
            date_str = (_dt.now(timezone.utc).date() - timedelta(days=1)).isoformat()

        exporter = Exporter()
        df = await exporter.get_usage_data(
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

        csv_payload = exporter.to_csv(df)

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
                f"Uploader: missing required config fields: {missing}. "
                "Set via /mavvrik/init or MAVVRIK_* environment variables."
            )
