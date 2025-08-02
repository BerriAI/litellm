"""
AnyCost Integration for LiteLLM

CloudZero AnyCost integration that sends daily spend data to CloudZero for cost
tracking and analytics.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from litellm._logging import verbose_logger
from litellm.integrations._base_custom_logger import CustomLogger
from litellm.integrations.anycost.utils import AnyCostUtils
from litellm.types.integrations.anycost import AnyCostConfig


class AnyCostLogger(CustomLogger):
    """
    AnyCost logger for sending data to CloudZero's AnyCost platform.

    This logger aggregates daily spend data and sends it to CloudZero in either
    CBF (Common Bill Format) or Telemetry API format with detailed breakdowns by:
    - Team (from team_id)
    - Service (from API key + key_alias)
    - Provider/Model (from custom_llm_provider + model)
    """

    def __init__(
        self,
        config: Optional[AnyCostConfig] = None,
        **kwargs
    ):
        """
        Initialize AnyCost logger

        Args:
            config: AnyCost configuration object
            **kwargs: Additional configuration options
        """
        super().__init__()

        # Load configuration
        if config is None:
            config = AnyCostConfig()

        self.config = config
        self.utils = AnyCostUtils(config)

        # Background task for daily data sending
        self._daily_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Validate configuration
        self._validate_config()

        verbose_logger.debug(
            f"AnyCost Logger initialized with charge_by={self.config.charge_by}"
        )

    def _validate_config(self) -> None:
        """Validate the configuration"""
        if self.config.cbf_file_format and not self.config.s3_bucket:
            raise ValueError(
                "S3 bucket is required when using CBF file format. "
                "Set ANYCOST_S3_BUCKET environment variable."
            )

        if self.config.telemetry_format and not self.config.api_key:
            raise ValueError(
                "API key is required when using telemetry format. "
                "Set CLOUDZERO_API_KEY environment variable."
            )

        if not self.config.cbf_file_format and not self.config.telemetry_format:
            raise ValueError(
                "At least one output format must be enabled. "
                "Set ANYCOST_CBF_FORMAT=true or ANYCOST_TELEMETRY_FORMAT=true."
            )


    async def start_background_tasks(self) -> None:
        """Start background tasks for daily data sending"""
        if not self.config.send_daily_reports:
            verbose_logger.debug("Daily reports disabled for AnyCost")
            return

        if self._daily_task is not None:
            verbose_logger.debug("AnyCost daily task already running")
            return

        self._daily_task = asyncio.create_task(self._daily_task_loop())
        verbose_logger.debug("Started AnyCost daily task")

    async def stop_background_tasks(self) -> None:
        """Stop background tasks"""
        self._shutdown = True

        if self._daily_task is not None:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass
            self._daily_task = None

        verbose_logger.debug("Stopped AnyCost background tasks")

    async def _daily_task_loop(self) -> None:
        """Background task that runs daily to send aggregated data"""
        verbose_logger.debug("AnyCost daily task loop started")

        while not self._shutdown:
            try:
                # Calculate next run time (1 AM UTC)
                now = datetime.now(timezone.utc)
                next_run = now.replace(hour=1, minute=0, second=0, microsecond=0)

                # If we've passed 1 AM today, schedule for tomorrow
                if now >= next_run:
                    next_run = next_run.replace(day=next_run.day + 1)

                # Wait until next run time
                sleep_seconds = (next_run - now).total_seconds()
                verbose_logger.debug(
                    f"AnyCost: Next daily run in {sleep_seconds} seconds at {next_run}"
                )

                await asyncio.sleep(sleep_seconds)

                if not self._shutdown:
                    await self._send_daily_data()

            except asyncio.CancelledError:
                verbose_logger.debug("AnyCost daily task cancelled")
                break
            except Exception as e:
                verbose_logger.error(f"AnyCost daily task error: {e}")
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)

    async def _send_daily_data(self) -> None:
        """Send yesterday's aggregated data to CloudZero with detailed breakdowns"""
        try:
            yesterday = datetime.now(timezone.utc).replace(
                day=datetime.now(timezone.utc).day - 1,
                hour=0, minute=0, second=0, microsecond=0
            )

            verbose_logger.debug(f"AnyCost: Sending daily data for {yesterday.date()}")

            # Send CBF data if enabled
            if self.config.cbf_file_format:
                success = await self.utils.send_cbf_data(yesterday)
                if success:
                    verbose_logger.info(
                        f"AnyCost: Successfully sent CBF data for {yesterday.date()}"
                    )
                else:
                    verbose_logger.error(
                        f"AnyCost: Failed to send CBF data for {yesterday.date()}"
                    )

            # Send telemetry data if enabled
            if self.config.telemetry_format:
                success = await self.utils.send_telemetry_data(yesterday)
                if success:
                    verbose_logger.info(
                        f"AnyCost: Successfully sent telemetry data for {yesterday.date()}"
                    )
                else:
                    verbose_logger.error(
                        f"AnyCost: Failed to send telemetry data for {yesterday.date()}"
                    )

        except Exception as e:
            verbose_logger.error(f"AnyCost: Error sending daily data: {e}")

    async def send_data_for_date(self, date: datetime) -> bool:
        """
        Manually send data for a specific date (useful for backfilling or testing)

        This will emit data with detailed breakdowns by:
        - Team (from team_id in daily spend data)
        - Service (from API key + key_alias lookup)
        - Provider/Model (from custom_llm_provider + model)

        Args:
            date: Date to send data for

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            success = True

            if self.config.cbf_file_format:
                cbf_success = await self.utils.send_cbf_data(date)
                success = success and cbf_success

            if self.config.telemetry_format:
                telemetry_success = await self.utils.send_telemetry_data(date)
                success = success and telemetry_success

            return success

        except Exception as e:
            verbose_logger.error(f"AnyCost: Error sending data for {date}: {e}")
            return False
