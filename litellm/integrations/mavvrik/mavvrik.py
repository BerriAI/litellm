"""MavvrikLogger — CustomLogger subclass for the Mavvrik cost-data integration.

Export flow (runs every MAVVRIK_EXPORT_INTERVAL_MINUTES, default 60):

  1. Read the last-exported marker date from LiteLLM_Config (format: YYYY-MM-DD).
     - First run: marker = yesterday (export yesterday as the first complete day).
  2. For each date from (marker + 1 day) up to yesterday (never today):
     a. Query LiteLLM_DailyUserSpend for rows where date = that day.
     b. Transform rows to CSV via MavvrikTransformer.
     c. GZIP-compress the CSV in memory.
     d. Obtain a GCS signed URL from the Mavvrik API (x-api-key auth).
     e. Upload via GCS resumable upload (POST initiate → PUT payload).
        GCS object name = date string (e.g. "2026-02-18") — same name on
        re-upload overwrites, making exports idempotent.
     f. Advance the marker to that date in LiteLLM_Config.
  3. Today is never exported — daily rows are still accumulating.

Environment variables (fallback when DB settings are absent):
    MAVVRIK_API_KEY          x-api-key sent to Mavvrik API
    MAVVRIK_API_ENDPOINT     Mavvrik API base URL
    MAVVRIK_TENANT           Mavvrik tenant slug
    MAVVRIK_INSTANCE_ID      Instance/cluster ID (placeholder for k8s path)
    MAVVRIK_TIMEZONE         Timezone for date handling (default: UTC)
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, cast

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    MAVVRIK_EXPORT_INTERVAL_MINUTES,
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any


class MavvrikLogger(CustomLogger):
    """Export LiteLLM spend data to Mavvrik via GCS signed URL uploads."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        tenant: Optional[str] = None,
        instance_id: Optional[str] = None,
        timezone: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("MAVVRIK_API_KEY")
        self.api_endpoint = api_endpoint or os.getenv("MAVVRIK_API_ENDPOINT", "")
        self.tenant = tenant or os.getenv("MAVVRIK_TENANT", "")
        self.instance_id = instance_id or os.getenv("MAVVRIK_INSTANCE_ID", "")
        self.timezone = timezone or os.getenv("MAVVRIK_TIMEZONE", "UTC")

        verbose_logger.debug(
            "MavvrikLogger initialised: endpoint=%s tenant=%s instance=%s timezone=%s",
            self.api_endpoint,
            self.tenant,
            self.instance_id,
            self.timezone,
        )

    # ------------------------------------------------------------------
    # Scheduled job entry-point
    # ------------------------------------------------------------------

    async def initialize_mavvrik_export_job(self):
        """Scheduled entry-point — honours pod lock when Redis is available."""
        from litellm.proxy.proxy_server import proxy_logging_obj

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

    async def _scheduled_export(self):
        """Export all complete days since the last marker, one file per day."""
        from litellm.integrations.mavvrik.database import LiteLLMDatabase

        db = LiteLLMDatabase()
        settings = await db.get_mavvrik_settings()
        marker_str: Optional[str] = settings.get("marker")

        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)

        # Determine start date (day after last exported date)
        if marker_str:
            try:
                # marker is a date string "YYYY-MM-DD"
                last_exported = date.fromisoformat(marker_str[:10])
                start_date = last_exported + timedelta(days=1)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikLogger: invalid marker '%s', starting from yesterday",
                    marker_str,
                )
                start_date = yesterday
        else:
            # First run — start from yesterday
            start_date = yesterday

        if start_date > yesterday:
            verbose_logger.debug(
                "MavvrikLogger: marker=%s is up to date, nothing to export", marker_str
            )
            return

        # Export each missed day individually
        export_date = start_date
        while export_date <= yesterday:
            date_str = export_date.isoformat()  # "YYYY-MM-DD"
            verbose_logger.info("MavvrikLogger: exporting date %s", date_str)

            await self.export_usage_data(
                date_str=date_str,
                limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
            )

            # Advance marker one day at a time so partial failures don't lose progress
            await db.advance_marker(date_str)

            # Notify Mavvrik of the new marker epoch (best-effort)
            try:
                from litellm.integrations.mavvrik.mavvrik_stream_api import (
                    MavvrikStreamer,
                )

                export_epoch = int(
                    datetime(
                        export_date.year,
                        export_date.month,
                        export_date.day,
                        tzinfo=timezone.utc,
                    ).timestamp()
                )
                streamer = MavvrikStreamer(
                    api_key=self.api_key or "",
                    api_endpoint=self.api_endpoint or "",
                    tenant=self.tenant or "",
                    instance_id=self.instance_id or "",
                )
                streamer.advance_marker(export_epoch)
            except Exception as exc:
                verbose_logger.warning(
                    "MavvrikLogger: advance_marker PATCH failed (non-fatal): %s", exc
                )

            export_date += timedelta(days=1)

    # ------------------------------------------------------------------
    # Core export
    # ------------------------------------------------------------------

    async def export_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ):
        """Query → transform → upload for a single calendar date (YYYY-MM-DD).

        The GCS object is named by date_str so re-uploading the same day
        overwrites the previous file — exports are idempotent.
        Called by the scheduler and manual /mavvrik/export.
        """
        from litellm.integrations.mavvrik.database import LiteLLMDatabase
        from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer
        from litellm.integrations.mavvrik.transform import MavvrikTransformer

        self._validate_config()

        verbose_logger.debug("MavvrikLogger: exporting date %s", date_str)

        db = LiteLLMDatabase()
        df = await db.get_usage_data(date_str=date_str, limit=limit)

        if df.is_empty():
            verbose_logger.debug(
                "MavvrikLogger: no spend data for %s, nothing to upload", date_str
            )
            return

        verbose_logger.debug("MavvrikLogger: %d rows fetched, transforming…", len(df))

        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df)

        if not csv_payload:
            verbose_logger.debug(
                "MavvrikLogger: 0 rows after transform for %s, skipping upload",
                date_str,
            )
            return

        streamer = MavvrikStreamer(
            api_key=self.api_key or "",
            api_endpoint=self.api_endpoint or "",
            tenant=self.tenant or "",
            instance_id=self.instance_id or "",
        )
        streamer.upload(csv_payload, date_str=date_str)

        verbose_logger.info(
            "MavvrikLogger: uploaded %d CSV bytes for date %s",
            len(csv_payload),
            date_str,
        )

    # ------------------------------------------------------------------
    # Dry run (preview without uploading)
    # ------------------------------------------------------------------

    async def dry_run_export_usage_data(self, limit: Optional[int] = None):
        """Return transformed records as dicts without uploading — for /mavvrik/dry-run.

        Queries the most recent complete day (yesterday) so the preview reflects
        real final data rather than a partial in-progress day.
        """
        from litellm.integrations.mavvrik.database import LiteLLMDatabase
        from litellm.integrations.mavvrik.transform import MavvrikTransformer

        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

        db = LiteLLMDatabase()
        df = await db.get_usage_data(
            date_str=yesterday,
            limit=limit or MAVVRIK_MAX_FETCHED_DATA_RECORDS,
        )

        if df.is_empty():
            return {
                "usage_data": [],
                "ndjson_records": [],
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

        # Compute summary stats from the raw DataFrame
        total_cost = float(df["spend"].sum()) if "spend" in df.columns else 0.0
        total_tokens = (
            int(
                (df["prompt_tokens"].sum() or 0)
                + (df["completion_tokens"].sum() or 0)
            )
            if "prompt_tokens" in df.columns
            else 0
        )
        unique_models = (
            df["model"].n_unique() if "model" in df.columns else 0
        )
        unique_teams = (
            df["team_id"].n_unique() if "team_id" in df.columns else 0
        )

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

    def _validate_config(self):
        missing = [
            name
            for name, val in [
                ("api_key", self.api_key),
                ("api_endpoint", self.api_endpoint),
                ("tenant", self.tenant),
                ("instance_id", self.instance_id),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"MavvrikLogger: missing required config fields: {missing}. "
                "Set via /mavvrik/init or MAVVRIK_* environment variables."
            )

    # ------------------------------------------------------------------
    # Background job registration
    # ------------------------------------------------------------------

    @staticmethod
    async def init_mavvrik_background_job(scheduler: AsyncIOScheduler):
        """Register the hourly export job with APScheduler."""
        loggers: List[
            CustomLogger
        ] = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=MavvrikLogger
        )
        verbose_logger.debug("MavvrikLogger: found %d logger instance(s)", len(loggers))
        if loggers:
            mavvrik_logger = cast(MavvrikLogger, loggers[0])
            verbose_logger.debug(
                "MavvrikLogger: scheduling export job every %d minutes",
                MAVVRIK_EXPORT_INTERVAL_MINUTES,
            )
            scheduler.add_job(
                mavvrik_logger.initialize_mavvrik_export_job,
                "interval",
                minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
            )
