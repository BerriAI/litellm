"""MavvrikLogger — CustomLogger subclass for the Mavvrik cost-data integration.

Export flow (runs every MAVVRIK_EXPORT_INTERVAL_MINUTES, default 60):

  1. Read the last-exported marker date from LiteLLM_Config (format: YYYY-MM-DD).
     - First run: determined by MAVVRIK_LOOKBACK_START_DATE or MIN(date) in DB.
  2. For each date from (marker + 1 day) up to yesterday (never today):
     a. Query LiteLLM_DailyUserSpend for rows where date = that day.
     b. Transform rows to CSV via MavvrikTransformer.
     c. GZIP-compress the CSV in memory.
     d. Upload via MavvrikUploader (3-step signed URL upload).
     e. Advance the marker to that date in LiteLLM_Config.
  3. Today is never exported — daily rows are still accumulating.

Environment variables (fallback when DB settings are absent):
    MAVVRIK_API_KEY              x-api-key sent to Mavvrik API
    MAVVRIK_API_ENDPOINT         Mavvrik API base URL (includes tenant)
    MAVVRIK_CONNECTION_ID        Connection/instance ID
    MAVVRIK_LOOKBACK_START_DATE  First-run start date YYYY-MM-DD (default: all data since MIN(date))
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_LOOKBACK_START_DATE,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.mavvrik.database import LiteLLMDatabase
from litellm.integrations.mavvrik.transform import MavvrikTransformer
from litellm.integrations.mavvrik.upload import MavvrikUploader


def _utc_now() -> datetime:
    """Return current UTC datetime. Extracted for easy test patching."""
    return datetime.now(timezone.utc)


class MavvrikLogger(CustomLogger):
    """Export LiteLLM spend data to Mavvrik via signed URL uploads."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        connection_id: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("MAVVRIK_API_KEY")
        self.api_endpoint = api_endpoint or os.getenv("MAVVRIK_API_ENDPOINT", "")
        self.connection_id = connection_id or os.getenv("MAVVRIK_CONNECTION_ID", "")

        verbose_logger.debug(
            "MavvrikLogger initialised: endpoint=%s connection_id=%s",
            self.api_endpoint,
            self.connection_id,
        )

    # ------------------------------------------------------------------
    # Scheduled job entry-point
    # ------------------------------------------------------------------

    async def initialize_mavvrik_export_job(self) -> None:
        """Scheduled entry-point — honours pod lock when Redis is available."""
        from litellm.proxy.proxy_server import proxy_logging_obj
        from litellm.constants import MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME

        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

        if pod_lock_manager and pod_lock_manager.redis_cache:
            if await pod_lock_manager.acquire_lock(
                cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
            ):
                try:
                    await self._scheduled_export()
                finally:
                    await pod_lock_manager.release_lock(
                        cronjob_id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME
                    )
        else:
            await self._scheduled_export()

    async def _scheduled_export(self) -> None:
        """Export all complete days since the last marker, one file per day."""
        db = LiteLLMDatabase()
        settings = await db.get_mavvrik_settings()
        marker_str: Optional[str] = settings.get("marker")

        today = _utc_now().date()
        yesterday = today - timedelta(days=1)

        # Determine start date (day after last exported date)
        if marker_str:
            try:
                last_exported = date.fromisoformat(marker_str[:10])
                start_date = last_exported + timedelta(days=1)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikLogger: invalid marker '%s', starting from yesterday",
                    marker_str,
                )
                start_date = yesterday
        else:
            start_date = await self._resolve_first_run_start_date(db, yesterday)

        if start_date > yesterday:
            verbose_logger.debug(
                "MavvrikLogger: marker=%s is up to date, nothing to export", marker_str
            )
            return

        export_date = start_date
        while export_date <= yesterday:
            date_str = export_date.isoformat()
            verbose_logger.info("MavvrikLogger: exporting date %s", date_str)

            try:
                await self.export_usage_data(
                    date_str=date_str,
                    limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
                )
            except ValueError as exc:
                # Config error (missing credentials) — stop the entire loop,
                # no point trying remaining dates.
                verbose_logger.error(
                    "MavvrikLogger: export stopped for %s — config error: %s",
                    date_str,
                    exc,
                )
                return
            except Exception as exc:
                # Transient errors (network, upload failure) — log and skip
                # this date so remaining days still get exported.
                verbose_logger.warning(
                    "MavvrikLogger: export failed for %s (skipping, will retry next run): %s",
                    date_str,
                    exc,
                )
                export_date += timedelta(days=1)
                continue

            # Advance the remote Mavvrik marker first (best-effort), then local.
            # Local marker is the source of truth; even if the remote PATCH fails
            # the next run will re-send it. Advancing local last ensures we don't
            # skip a date if the remote call raises.
            try:
                export_epoch = int(
                    datetime(
                        export_date.year,
                        export_date.month,
                        export_date.day,
                        tzinfo=timezone.utc,
                    ).timestamp()
                )
                uploader = MavvrikUploader(
                    api_key=self.api_key or "",
                    api_endpoint=self.api_endpoint or "",
                    connection_id=self.connection_id or "",
                )
                await uploader.advance_marker(export_epoch)
            except Exception as exc:
                verbose_logger.warning(
                    "MavvrikLogger: advance_marker PATCH failed (non-fatal): %s", exc
                )

            await db.advance_marker(date_str)

            export_date += timedelta(days=1)

    async def _resolve_first_run_start_date(
        self, db: LiteLLMDatabase, yesterday: date
    ) -> date:
        """Determine the export start date on the very first run (no marker yet).

        Priority:
          1. MAVVRIK_LOOKBACK_START_DATE env var (clamped to MIN(date) in DB)
          2. MIN(date) in LiteLLM_DailyUserSpend
          3. Yesterday as fallback
        """
        requested_start: Optional[date] = None
        if MAVVRIK_LOOKBACK_START_DATE is not None:
            try:
                requested_start = date.fromisoformat(MAVVRIK_LOOKBACK_START_DATE)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikLogger: invalid MAVVRIK_LOOKBACK_START_DATE '%s' "
                    "(expected YYYY-MM-DD), falling back to earliest DB date",
                    MAVVRIK_LOOKBACK_START_DATE,
                )

        earliest_str = await db.get_earliest_date()
        earliest_db: Optional[date] = None
        if earliest_str:
            try:
                earliest_db = date.fromisoformat(earliest_str)
            except ValueError:
                pass

        if requested_start is not None and earliest_db is not None:
            start_date = max(requested_start, earliest_db)
            if start_date != requested_start:
                verbose_logger.info(
                    "MavvrikLogger: MAVVRIK_LOOKBACK_START_DATE=%s is before "
                    "earliest DB date %s — starting from %s",
                    requested_start,
                    earliest_db,
                    start_date,
                )
            else:
                verbose_logger.info(
                    "MavvrikLogger: no marker found, starting from "
                    "MAVVRIK_LOOKBACK_START_DATE %s",
                    start_date,
                )
        elif requested_start is not None:
            start_date = requested_start
            verbose_logger.info(
                "MavvrikLogger: no marker found, no DB data yet, "
                "starting from MAVVRIK_LOOKBACK_START_DATE %s",
                start_date,
            )
        elif earliest_db is not None:
            start_date = earliest_db
            verbose_logger.info(
                "MavvrikLogger: no marker found, starting from earliest DB date %s",
                start_date,
            )
        else:
            start_date = yesterday

        return start_date

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

        verbose_logger.debug("MavvrikLogger: exporting date %s", date_str)

        db = LiteLLMDatabase()
        df = await db.get_usage_data(date_str=date_str, limit=limit)

        if df.is_empty():
            verbose_logger.debug(
                "MavvrikLogger: no spend data for %s, nothing to upload", date_str
            )
            return 0

        record_count = len(df)
        verbose_logger.debug(
            "MavvrikLogger: %d rows fetched, transforming…", record_count
        )

        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df, connection_id=self.connection_id)

        if not csv_payload:
            verbose_logger.debug(
                "MavvrikLogger: 0 rows after transform for %s, skipping upload",
                date_str,
            )
            return 0

        uploader = MavvrikUploader(
            api_key=self.api_key or "",
            api_endpoint=self.api_endpoint or "",
            connection_id=self.connection_id or "",
        )
        await uploader.upload(csv_payload, date_str=date_str)

        verbose_logger.info(
            "MavvrikLogger: uploaded %d CSV bytes for date %s",
            len(csv_payload),
            date_str,
        )
        return record_count

    # ------------------------------------------------------------------
    # Dry run (preview without uploading)
    # ------------------------------------------------------------------

    async def dry_run_export_usage_data(
        self,
        date_str: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """Return transformed records as dicts without uploading — for /mavvrik/dry-run."""
        if not date_str:
            date_str = (_utc_now().date() - timedelta(days=1)).isoformat()

        db = LiteLLMDatabase()
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

        usage_sample = df.head(50).to_dicts()
        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df)

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
                ("api_key", self.api_key),
                ("api_endpoint", self.api_endpoint),
                ("connection_id", self.connection_id),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"MavvrikLogger: missing required config fields: {missing}. "
                "Set via /mavvrik/init or MAVVRIK_* environment variables."
            )
