"""MavvrikScheduler — pod lock, date loop, and APScheduler job registration.

Owns all concerns around scheduling the Mavvrik export job:
  - Acquiring/releasing the pod lock (Redis-backed, multi-pod safe)
  - Iterating over complete calendar days since the last exported marker
  - Exporting each day via MavvrikExporter.export_usage_data()
  - Advancing the local (DB) and remote (Mavvrik API) markers
  - Registering the APScheduler job at startup or on /mavvrik/init

The two class-methods (register_job, register_exporter_and_job) replace
register.py entirely.  They create a MavvrikScheduler internally so callers
never need to instantiate the class by hand.
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Optional, cast

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import (
    MAVVRIK_EXPORT_INTERVAL_MINUTES,
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
    MAVVRIK_LOOKBACK_START_DATE,
    MAVVRIK_MAX_FETCHED_DATA_RECORDS,
)
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from litellm.integrations.mavvrik.exporter import MavvrikExporter
else:
    AsyncIOScheduler = Any
    MavvrikExporter = Any


class MavvrikScheduler:
    """Scheduler wrapper that drives the incremental Mavvrik export loop."""

    def __init__(self, exporter: "MavvrikExporter") -> None:
        self._exporter = exporter

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def job_name(self) -> str:
        """APScheduler job id / pod-lock cronjob id."""
        return MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME

    @property
    def interval_minutes(self) -> int:
        """How often the scheduler fires the export job."""
        return MAVVRIK_EXPORT_INTERVAL_MINUTES

    @staticmethod
    def _utc_today() -> date:
        """Return today's date in UTC.  Extracted so tests can patch it."""
        return datetime.now(timezone.utc).date()

    @property
    def _yesterday(self) -> date:
        return self._utc_today() - timedelta(days=1)

    # ------------------------------------------------------------------
    # Scheduled entry-point (called by APScheduler)
    # ------------------------------------------------------------------

    async def run_export_job(self) -> None:
        """Honour the pod lock when Redis is available, then run the export loop."""
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

        if pod_lock_manager and pod_lock_manager.redis_cache:
            if await pod_lock_manager.acquire_lock(cronjob_id=self.job_name):
                try:
                    await self._run_export_loop()
                finally:
                    await pod_lock_manager.release_lock(cronjob_id=self.job_name)
            else:
                verbose_logger.debug(
                    "MavvrikScheduler: pod lock not acquired for %s — another pod is running",
                    self.job_name,
                )
        else:
            await self._run_export_loop()

    # ------------------------------------------------------------------
    # Export loop
    # ------------------------------------------------------------------

    async def _run_export_loop(self) -> None:
        """Export every complete calendar day since the Mavvrik-side marker.

        The marker (cursor) is owned by the Mavvrik API. We call register()
        at the start of every run to retrieve it — no local DB marker needed.
        """
        from litellm.integrations.mavvrik.client import MavvrikClient
        from litellm.integrations.mavvrik.database import MavvrikDatabase

        db = MavvrikDatabase()
        yesterday = self._yesterday

        # Retrieve the current export marker from Mavvrik (source of truth).
        # Falls back to _resolve_first_run_start_date if the call fails.
        marker_str: Optional[str] = None
        try:
            client = MavvrikClient(
                api_key=self._exporter.api_key or "",
                api_endpoint=self._exporter.api_endpoint or "",
                connection_id=self._exporter.connection_id or "",
            )
            marker_str = await client.register()
            verbose_logger.debug("MavvrikScheduler: Mavvrik marker = %s", marker_str)
        except Exception as exc:
            verbose_logger.warning(
                "MavvrikScheduler: register() failed (non-fatal), "
                "falling back to first-run start date: %s",
                exc,
            )

        # Determine start date from the Mavvrik marker or first-run logic.
        if marker_str:
            try:
                last_exported = date.fromisoformat(marker_str[:10])
                start_date = last_exported + timedelta(days=1)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikScheduler: invalid marker '%s', starting from yesterday",
                    marker_str,
                )
                start_date = yesterday
        else:
            start_date = await self._resolve_first_run_start_date(db, yesterday)

        if start_date > yesterday:
            verbose_logger.debug(
                "MavvrikScheduler: marker=%s is up to date, nothing to export",
                marker_str,
            )
            return

        export_date = start_date
        while export_date <= yesterday:
            date_str = export_date.isoformat()
            verbose_logger.info("MavvrikScheduler: exporting date %s", date_str)

            try:
                await self._exporter.export_usage_data(
                    date_str=date_str,
                    limit=MAVVRIK_MAX_FETCHED_DATA_RECORDS,
                )
            except ValueError as exc:
                # Config error (missing credentials) — stop the entire loop;
                # no point trying remaining dates with a broken config.
                verbose_logger.error(
                    "MavvrikScheduler: export stopped for %s — config error: %s",
                    date_str,
                    exc,
                )
                await client.report_error(f"Config error for date {date_str}: {exc}")
                return
            except Exception as exc:
                # Transient error (network, upload failure) — log and skip
                # this date so remaining days still get exported.
                verbose_logger.warning(
                    "MavvrikScheduler: export failed for %s "
                    "(skipping, will retry next run): %s",
                    date_str,
                    exc,
                )
                await client.report_error(f"Export failed for date {date_str}: {exc}")
                export_date += timedelta(days=1)
                continue

            # Advance the marker on the Mavvrik side (source of truth).
            # No local DB marker is stored — the next run retrieves the
            # updated marker from Mavvrik via register().
            try:
                export_epoch = int(
                    datetime(
                        export_date.year,
                        export_date.month,
                        export_date.day,
                        tzinfo=timezone.utc,
                    ).timestamp()
                )
                await client.advance_marker(export_epoch)
            except Exception as exc:
                verbose_logger.warning(
                    "MavvrikScheduler: advance_marker PATCH failed (non-fatal): %s",
                    exc,
                )

            export_date += timedelta(days=1)

    # ------------------------------------------------------------------
    # First-run start date resolution
    # ------------------------------------------------------------------

    async def _resolve_first_run_start_date(
        self,
        db: Any,  # MavvrikDatabase — Any to avoid circular import at class level
        yesterday: date,
    ) -> date:
        """Determine the export start date on the very first run (no marker yet).

        Priority:
          1. MAVVRIK_LOOKBACK_START_DATE env var (clamped to MIN(date) in DB)
          2. MIN(date) in LiteLLM_DailyUserSpend
          3. Yesterday as a last-resort fallback
        """
        requested_start: Optional[date] = None
        if MAVVRIK_LOOKBACK_START_DATE is not None:
            try:
                requested_start = date.fromisoformat(MAVVRIK_LOOKBACK_START_DATE)
            except ValueError:
                verbose_logger.warning(
                    "MavvrikScheduler: invalid MAVVRIK_LOOKBACK_START_DATE '%s' "
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
                    "MavvrikScheduler: MAVVRIK_LOOKBACK_START_DATE=%s is before "
                    "earliest DB date %s — starting from %s",
                    requested_start,
                    earliest_db,
                    start_date,
                )
            else:
                verbose_logger.info(
                    "MavvrikScheduler: no marker found, starting from "
                    "MAVVRIK_LOOKBACK_START_DATE %s",
                    start_date,
                )
        elif requested_start is not None:
            start_date = requested_start
            verbose_logger.info(
                "MavvrikScheduler: no marker found, no DB data yet, "
                "starting from MAVVRIK_LOOKBACK_START_DATE %s",
                start_date,
            )
        elif earliest_db is not None:
            start_date = earliest_db
            verbose_logger.info(
                "MavvrikScheduler: no marker found, starting from earliest DB date %s",
                start_date,
            )
        else:
            start_date = yesterday

        return start_date

    # ------------------------------------------------------------------
    # Class-method helpers (replace register.py functions)
    # ------------------------------------------------------------------

    @classmethod
    def register_job(cls, scheduler: "AsyncIOScheduler") -> None:
        """Register the Mavvrik hourly export job with APScheduler.

        Called from proxy_server.py at startup.  Only registers when a
        MavvrikExporter instance already exists in the callback manager
        (i.e. the operator added ``callbacks: ["mavvrik"]`` in config.yaml).
        """
        from litellm.integrations.mavvrik.exporter import MavvrikExporter as _Exporter

        loggers: List[CustomLogger] = (
            litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=_Exporter
            )
        )
        verbose_logger.debug(
            "MavvrikScheduler.register_job: found %d exporter instance(s)",
            len(loggers),
        )

        if loggers:
            mavvrik_exporter = cast(_Exporter, loggers[0])
            mavvrik_scheduler = cls(exporter=mavvrik_exporter)
            verbose_logger.debug(
                "MavvrikScheduler: scheduling export job every %d minutes",
                MAVVRIK_EXPORT_INTERVAL_MINUTES,
            )
            scheduler.add_job(
                mavvrik_scheduler.run_export_job,
                "interval",
                minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
                id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
                replace_existing=True,
            )

    @classmethod
    async def register_exporter_and_job(
        cls,
        api_key: str,
        api_endpoint: str,
        connection_id: str,
        scheduler: Optional["AsyncIOScheduler"],
    ) -> None:
        """Create a MavvrikExporter, register it in the callback manager, and schedule.

        Called from POST /mavvrik/init so that exports begin immediately without
        a proxy restart.  All scheduling logic lives here, not in the endpoint.
        """
        from litellm.integrations.mavvrik.exporter import MavvrikExporter as _Exporter

        # Remove any existing MavvrikExporter instances to avoid accumulating
        # duplicates on repeated /mavvrik/init calls.
        for cb_attr in (
            "success_callbacks",
            "failure_callbacks",
            "async_success_callbacks",
        ):
            cb_list = getattr(litellm.logging_callback_manager, cb_attr, None)
            if cb_list is not None:
                setattr(
                    litellm.logging_callback_manager,
                    cb_attr,
                    [cb for cb in cb_list if not isinstance(cb, _Exporter)],
                )

        mavvrik_exporter = _Exporter(
            api_key=api_key,
            api_endpoint=api_endpoint,
            connection_id=connection_id,
        )
        litellm.logging_callback_manager.add_litellm_success_callback(mavvrik_exporter)
        litellm.logging_callback_manager.add_litellm_async_success_callback(
            mavvrik_exporter
        )

        # Remove the plain string entry added by callbacks: ["mavvrik"] in config.yaml
        for cb_list in (litellm.success_callback, litellm._async_success_callback):
            if "mavvrik" in cb_list:
                cb_list.remove("mavvrik")

        if scheduler is not None:
            mavvrik_scheduler = cls(exporter=mavvrik_exporter)
            scheduler.add_job(
                mavvrik_scheduler.run_export_job,
                "interval",
                minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
                id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
                replace_existing=True,
            )
            verbose_proxy_logger.info(
                "Mavvrik background export job scheduled every %d min",
                MAVVRIK_EXPORT_INTERVAL_MINUTES,
            )
        else:
            verbose_proxy_logger.warning(
                "Mavvrik: scheduler not available, background job not registered"
            )
