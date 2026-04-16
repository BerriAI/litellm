"""Scheduler registration and setup detection for the Mavvrik integration.

This module owns all concerns around wiring Mavvrik into the LiteLLM proxy:
  - Detecting whether Mavvrik is configured (env vars or DB settings)
  - Registering the hourly background export job with APScheduler
  - Registering a logger instance into the callback manager (called from /mavvrik/init)

proxy_server.py and mavvrik_endpoints.py delegate to this module entirely
so no scheduling logic leaks into the endpoint layer.
"""

import os
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
else:
    AsyncIOScheduler = Any

_CONFIG_KEY = "mavvrik_settings"


async def is_mavvrik_setup() -> bool:
    """Return True if Mavvrik credentials exist in env vars or the database."""
    if all(
        os.getenv(v)
        for v in ("MAVVRIK_API_KEY", "MAVVRIK_API_ENDPOINT", "MAVVRIK_CONNECTION_ID")
    ):
        return True
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return False
        row = await prisma_client.db.litellm_config.find_first(
            where={"param_name": _CONFIG_KEY}
        )
        return row is not None and row.param_value is not None
    except Exception:
        return False


async def register_background_job(scheduler: AsyncIOScheduler) -> None:
    """Register the Mavvrik hourly export job with APScheduler.

    Called from proxy_server.py at startup. Only registers if a MavvrikExporter
    instance already exists in the callback manager (i.e. the operator has
    added ``callbacks: ["mavvrik"]`` in config.yaml).
    """
    from litellm.integrations.mavvrik.exporter import MavvrikExporter

    loggers: List[CustomLogger] = (
        litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=MavvrikExporter
        )
    )
    verbose_logger.debug("MavvrikExporter: found %d exporter instance(s)", len(loggers))

    if loggers:
        mavvrik_logger = cast(MavvrikExporter, loggers[0])
        verbose_logger.debug(
            "MavvrikLogger: scheduling export job every %d minutes",
            MAVVRIK_EXPORT_INTERVAL_MINUTES,
        )
        scheduler.add_job(
            mavvrik_logger.initialize_mavvrik_export_job,
            "interval",
            minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
            id=MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME,
            replace_existing=True,
        )


async def register_logger_and_job(
    api_key: str,
    api_endpoint: str,
    connection_id: str,
    scheduler: Optional[AsyncIOScheduler],
) -> None:
    """Create a MavvrikExporter, register it in the callback manager, and schedule the job.

    Called from POST /mavvrik/init so that exports begin immediately without
    a proxy restart. All scheduling logic lives here, not in the endpoint handler.
    """
    from litellm.integrations.mavvrik.exporter import MavvrikExporter

    # Remove any existing MavvrikExporter instances to avoid accumulating duplicates
    # on repeated /mavvrik/init calls. We mutate the internal callback lists directly
    # rather than relying on remove_litellm_callback_by_type() which may not exist
    # in all LiteLLM versions.
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
                [cb for cb in cb_list if not isinstance(cb, MavvrikExporter)],
            )

    mavvrik_logger = MavvrikExporter(
        api_key=api_key,
        api_endpoint=api_endpoint,
        connection_id=connection_id,
    )
    litellm.logging_callback_manager.add_litellm_success_callback(mavvrik_logger)
    litellm.logging_callback_manager.add_litellm_async_success_callback(mavvrik_logger)

    # Remove the plain string entry added by callbacks: ["mavvrik"] in config.yaml
    for cb_list in (litellm.success_callback, litellm._async_success_callback):
        if "mavvrik" in cb_list:
            cb_list.remove("mavvrik")

    if scheduler is not None:
        scheduler.add_job(
            mavvrik_logger.initialize_mavvrik_export_job,
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
