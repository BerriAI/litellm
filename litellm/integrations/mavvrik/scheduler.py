"""MavvrikScheduler — APScheduler job registration and pod lock.

Owns infrastructure concerns only:
  - Registering / removing APScheduler interval jobs
  - Acquiring / releasing the pod lock (Redis-backed, multi-pod safe)

Business orchestration (register → date loop → export → advance marker) is
delegated to MavvrikOrchestrator.
"""

from typing import TYPE_CHECKING, Any, List, Optional, cast

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import (
    MAVVRIK_EXPORT_INTERVAL_MINUTES,
    MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
)
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from litellm.integrations.mavvrik.exporter import MavvrikExporter
else:
    AsyncIOScheduler = Any
    MavvrikExporter = Any


class MavvrikScheduler:
    """Thin scheduler wrapper: pod lock + APScheduler, delegates to orchestrator."""

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

    # ------------------------------------------------------------------
    # Scheduled entry-point (called by APScheduler)
    # ------------------------------------------------------------------

    async def run_export_job(self) -> None:
        """Honour the pod lock when Redis is available, then run the orchestrator."""
        from litellm.integrations.mavvrik.orchestrator import MavvrikOrchestrator
        from litellm.proxy.proxy_server import proxy_logging_obj

        orchestrator = MavvrikOrchestrator(exporter=self._exporter)
        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

        if pod_lock_manager and pod_lock_manager.redis_cache:
            if await pod_lock_manager.acquire_lock(cronjob_id=self.job_name):
                try:
                    await orchestrator.run()
                finally:
                    await pod_lock_manager.release_lock(cronjob_id=self.job_name)
            else:
                verbose_logger.debug(
                    "MavvrikScheduler: pod lock not acquired for %s — another pod is running",
                    self.job_name,
                )
        else:
            await orchestrator.run()

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
