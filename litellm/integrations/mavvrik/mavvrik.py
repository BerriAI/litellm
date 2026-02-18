"""MavvrikLogger — CustomLogger subclass for the Mavvrik cost-data integration.

Export flow (runs every MAVVRIK_EXPORT_INTERVAL_MINUTES, default 60):

  1. Read the last-uploaded marker from LiteLLM_Config.
     - First run: start_time = now - 2× interval (backfill safety window).
     - Subsequent runs: start_time = marker timestamp (delta / incremental).
  2. Query LiteLLM_DailyUserSpend for rows with updated_at in [start, now].
  3. Transform rows to NDJSON via MavvrikTransformer.
  4. GZIP-compress the NDJSON in memory.
  5. Obtain a GCS signed URL from the Mavvrik API (x-api-key auth).
  6. Upload via GCS resumable upload (POST initiate → PUT payload).
  7. Advance the marker to `now` so the next run starts from here.

Environment variables (fallback when DB settings are absent):
    MAVVRIK_API_KEY          x-api-key sent to Mavvrik API
    MAVVRIK_API_ENDPOINT     Mavvrik API base URL
    MAVVRIK_TENANT           Mavvrik tenant slug
    MAVVRIK_INSTANCE_ID      Instance/cluster ID (placeholder for k8s path)
    MAVVRIK_TIMEZONE         Timezone for date handling (default: UTC)
"""

import os
from datetime import datetime, timedelta, timezone
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
        """Determine the export window from the marker, then export."""
        from litellm.integrations.mavvrik.database import LiteLLMDatabase

        db = LiteLLMDatabase()
        settings = await db.get_mavvrik_settings()
        marker_str: Optional[str] = settings.get("marker")

        now_utc = datetime.now(timezone.utc)

        if marker_str:
            # Delta mode — start from where we last finished
            try:
                start_utc = datetime.fromisoformat(marker_str)
                if start_utc.tzinfo is None:
                    start_utc = start_utc.replace(tzinfo=timezone.utc)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikLogger: invalid marker '%s', falling back to 2× interval window",
                    marker_str,
                )
                start_utc = now_utc - timedelta(
                    minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES * 2
                )
        else:
            # First run — use 2× interval overlap as a safety window
            start_utc = now_utc - timedelta(minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES * 2)

        await self.export_usage_data(
            limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
            start_time_utc=start_utc,
            end_time_utc=now_utc,
        )

        # Advance marker to now after a successful export
        await db.advance_marker(now_utc.isoformat())

        # Also notify Mavvrik so it knows where to start the next ingestion window
        try:
            from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer

            streamer = MavvrikStreamer(
                api_key=self.api_key,
                api_endpoint=self.api_endpoint,
                tenant=self.tenant,
                instance_id=self.instance_id,
            )
            streamer.advance_marker(int(now_utc.timestamp()))
        except Exception as exc:
            verbose_logger.warning(
                "MavvrikLogger: advance_marker PATCH failed (non-fatal): %s", exc
            )

    # ------------------------------------------------------------------
    # Core export
    # ------------------------------------------------------------------

    async def export_usage_data(
        self,
        limit: Optional[int] = None,
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ):
        """Query → transform → upload. Called by scheduler and manual /mavvrik/export."""
        from litellm.integrations.mavvrik.database import LiteLLMDatabase
        from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer
        from litellm.integrations.mavvrik.transform import MavvrikTransformer

        self._validate_config()

        verbose_logger.debug(
            "MavvrikLogger: starting export window %s → %s",
            start_time_utc,
            end_time_utc,
        )

        db = LiteLLMDatabase()
        df = await db.get_usage_data(
            limit=limit,
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
        )

        if df.is_empty():
            verbose_logger.debug(
                "MavvrikLogger: no spend data in window, nothing to upload"
            )
            return

        verbose_logger.debug("MavvrikLogger: %d rows fetched, transforming…", len(df))

        transformer = MavvrikTransformer()
        csv_payload = transformer.to_csv(df)

        if not csv_payload:
            verbose_logger.debug(
                "MavvrikLogger: 0 rows after transform, skipping upload"
            )
            return

        interval = (start_time_utc or datetime.now(timezone.utc)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        streamer = MavvrikStreamer(
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            tenant=self.tenant,
            instance_id=self.instance_id,
        )
        streamer.upload(csv_payload, interval=interval)

        verbose_logger.info(
            "MavvrikLogger: uploaded %d CSV bytes for interval %s",
            len(csv_payload),
            interval,
        )

    # ------------------------------------------------------------------
    # Dry run (preview without uploading)
    # ------------------------------------------------------------------

    async def dry_run_export_usage_data(self, limit: Optional[int] = None):
        """Return transformed records as dicts without uploading — for /mavvrik/dry-run."""
        from litellm.integrations.mavvrik.database import LiteLLMDatabase
        from litellm.integrations.mavvrik.transform import MavvrikTransformer

        db = LiteLLMDatabase()
        df = await db.get_usage_data(limit=limit or MAVVRIK_MAX_FETCHED_DATA_RECORDS)

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
