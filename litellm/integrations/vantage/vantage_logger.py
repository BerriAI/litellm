"""Vantage logger — thin wrapper around the Focus export pipeline.

Configures FocusLogger to use the Vantage API destination with CSV format
so users can simply set ``success_callback: ["vantage"]`` in their proxy config.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.focus.focus_logger import FocusLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any

VANTAGE_USAGE_DATA_JOB_NAME = "vantage_export_usage_data"


class VantageLogger(FocusLogger):
    """FocusLogger pre-configured for Vantage (CSV format, Vantage API destination).

    Environment Variables:
        VANTAGE_API_KEY: Vantage API key for authentication
        VANTAGE_INTEGRATION_TOKEN: Vantage integration token for the cost-import endpoint
        VANTAGE_BASE_URL: Optional base URL override (default: https://api.vantage.sh)
        VANTAGE_EXPORT_FREQUENCY: Export frequency — "hourly" (default), "daily", or "interval"
        VANTAGE_EXPORT_INTERVAL_SECONDS: Interval in seconds when frequency is "interval"
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        integration_token: Optional[str] = None,
        base_url: Optional[str] = None,
        frequency: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv("VANTAGE_API_KEY")
        resolved_token = integration_token or os.getenv("VANTAGE_INTEGRATION_TOKEN")
        resolved_base_url = base_url or os.getenv(
            "VANTAGE_BASE_URL", "https://api.vantage.sh"
        )
        resolved_frequency = (
            frequency or os.getenv("VANTAGE_EXPORT_FREQUENCY") or "hourly"
        ).lower()

        raw_interval = interval_seconds or os.getenv("VANTAGE_EXPORT_INTERVAL_SECONDS")
        resolved_interval: Optional[int] = None
        if raw_interval is not None:
            try:
                resolved_interval = int(raw_interval)
            except (ValueError, TypeError):
                verbose_logger.warning(
                    "Invalid VANTAGE_EXPORT_INTERVAL_SECONDS value: %s, ignoring",
                    raw_interval,
                )

        destination_config: Dict[str, Any] = {}
        if resolved_api_key:
            destination_config["api_key"] = resolved_api_key
        if resolved_token:
            destination_config["integration_token"] = resolved_token
        if resolved_base_url:
            destination_config["base_url"] = resolved_base_url

        super().__init__(
            provider="vantage",
            export_format="csv",
            frequency=resolved_frequency,
            interval_seconds=resolved_interval,
            prefix="vantage_exports",
            destination_config=destination_config,
            **kwargs,
        )

        verbose_logger.debug(
            "VantageLogger initialized (integration_token=%s)",
            resolved_token[:4] + "***"
            if resolved_token and len(resolved_token) > 4
            else "***",
        )

    async def initialize_focus_export_job(self) -> None:
        """Override to use the Vantage-specific pod lock key.

        Without this, VantageLogger and FocusLogger would compete for the
        same ``FOCUS_USAGE_DATA_JOB_NAME`` lock, causing one to silently
        skip its export cycle when both are configured simultaneously.
        """
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager = None
        if proxy_logging_obj is not None:
            writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
            if writer is not None:
                pod_lock_manager = getattr(writer, "pod_lock_manager", None)

        if pod_lock_manager and pod_lock_manager.redis_cache:
            acquired = await pod_lock_manager.acquire_lock(
                cronjob_id=VANTAGE_USAGE_DATA_JOB_NAME
            )
            if not acquired:
                verbose_logger.debug("Vantage export: unable to acquire pod lock")
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(
                    cronjob_id=VANTAGE_USAGE_DATA_JOB_NAME
                )
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_vantage_background_job(
        scheduler: AsyncIOScheduler,
    ) -> None:
        """Register the Vantage export job with the provided scheduler."""
        vantage_loggers: List[
            CustomLogger
        ] = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=VantageLogger
        )
        if not vantage_loggers:
            verbose_logger.debug("No Vantage logger registered; skipping scheduler")
            return

        vantage_logger = cast(VantageLogger, vantage_loggers[0])
        trigger_kwargs = vantage_logger._build_scheduler_trigger()
        scheduler.add_job(
            vantage_logger.initialize_focus_export_job,
            **trigger_kwargs,
        )


__all__ = ["VantageLogger"]
