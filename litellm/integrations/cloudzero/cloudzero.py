import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, cast

import litellm
from litellm._logging import verbose_logger
from litellm.constants import CLOUDZERO_EXPORT_INTERVAL_MINUTES
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any


class CloudZeroLogger(CustomLogger):
    """
    CloudZero Logger for exporting LiteLLM usage data to CloudZero AnyCost API.

    Environment Variables:
        CLOUDZERO_API_KEY: CloudZero API key for authentication
        CLOUDZERO_CONNECTION_ID: CloudZero connection ID for data submission
        CLOUDZERO_TIMEZONE: Timezone for date handling (default: UTC)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        connection_id: Optional[str] = None,
        timezone: Optional[str] = None,
        **kwargs,
    ):
        """Initialize CloudZero logger with configuration from parameters or environment variables."""
        super().__init__(**kwargs)

        # Get configuration from parameters first, fall back to environment variables
        self.api_key = api_key or os.getenv("CLOUDZERO_API_KEY")
        self.connection_id = connection_id or os.getenv("CLOUDZERO_CONNECTION_ID")
        self.timezone = timezone or os.getenv("CLOUDZERO_TIMEZONE", "UTC")
        verbose_logger.debug(
            f"CloudZero Logger initialized with connection ID: {self.connection_id}, timezone: {self.timezone}"
        )

    async def initialize_cloudzero_export_job(self):
        """
        Handler for initializing CloudZero export job.

        Runs when CloudZero logger starts up.

        - If redis cache is available, we use the pod lock manager to acquire a lock and export the data.
            - Ensures only one pod exports the data at a time.
        - If redis cache is not available, we export the data directly.
        """
        from litellm.constants import (
            CLOUDZERO_EXPORT_USAGE_DATA_JOB_NAME,
        )
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

        # if using redis, ensure only one pod exports the data at a time
        if pod_lock_manager and pod_lock_manager.redis_cache:
            if await pod_lock_manager.acquire_lock(
                cronjob_id=CLOUDZERO_EXPORT_USAGE_DATA_JOB_NAME
            ):
                try:
                    await self._hourly_usage_data_export()
                finally:
                    await pod_lock_manager.release_lock(
                        cronjob_id=CLOUDZERO_EXPORT_USAGE_DATA_JOB_NAME
                    )
        else:
            # if not using redis, export the data directly
            await self._hourly_usage_data_export()

    async def _hourly_usage_data_export(self):
        """
        Exports the hourly usage data to CloudZero.

        Start time: 1 hour ago
        End time: current time
        """
        from datetime import timedelta, timezone

        from litellm.constants import CLOUDZERO_MAX_FETCHED_DATA_RECORDS

        current_time_utc = datetime.now(timezone.utc)
        # Mitigates the possibility of missing spend if an hour is skipped due to a restart in an ephemeral environment
        one_hour_ago_utc = current_time_utc - timedelta(
            minutes=CLOUDZERO_EXPORT_INTERVAL_MINUTES * 2
        )
        await self.export_usage_data(
            limit=CLOUDZERO_MAX_FETCHED_DATA_RECORDS,
            operation="replace_hourly",
            start_time_utc=one_hour_ago_utc,
            end_time_utc=current_time_utc,
        )

    async def export_usage_data(
        self,
        limit: Optional[int] = None,
        operation: str = "replace_hourly",
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None,
    ):
        """
        Exports the usage data to CloudZero.

        - Reads data from the DB
        - Transforms the data to the CloudZero format
        - Sends the data to CloudZero

        Args:
            limit: Optional limit on number of records to export
            operation: CloudZero operation type ("replace_hourly" or "sum")
        """
        from litellm.integrations.cloudzero.cz_stream_api import CloudZeroStreamer
        from litellm.integrations.cloudzero.database import LiteLLMDatabase
        from litellm.integrations.cloudzero.transform import CBFTransformer

        try:
            verbose_logger.debug("CloudZero Logger: Starting usage data export")

            # Validate required configuration
            if not self.api_key or not self.connection_id:
                raise ValueError(
                    "CloudZero configuration missing. Please set CLOUDZERO_API_KEY and CLOUDZERO_CONNECTION_ID environment variables."
                )

            # Initialize database connection and load data
            database = LiteLLMDatabase()
            verbose_logger.debug("CloudZero Logger: Loading usage data from database")
            data = await database.get_usage_data(
                limit=limit, start_time_utc=start_time_utc, end_time_utc=end_time_utc
            )

            if data.is_empty():
                verbose_logger.debug("CloudZero Logger: No usage data found to export")
                return

            verbose_logger.debug(f"CloudZero Logger: Processing {len(data)} records")

            # Transform data to CloudZero CBF format
            transformer = CBFTransformer()
            cbf_data = transformer.transform(data)

            if cbf_data.is_empty():
                verbose_logger.warning(
                    "CloudZero Logger: No valid data after transformation"
                )
                return

            # Send data to CloudZero
            streamer = CloudZeroStreamer(
                api_key=self.api_key,
                connection_id=self.connection_id,
                user_timezone=self.timezone,
            )

            verbose_logger.debug(
                f"CloudZero Logger: Transmitting {len(cbf_data)} records to CloudZero"
            )
            streamer.send_batched(cbf_data, operation=operation)

            verbose_logger.debug(
                f"CloudZero Logger: Successfully exported {len(cbf_data)} records to CloudZero"
            )

        except Exception as e:
            verbose_logger.error(
                f"CloudZero Logger: Error exporting usage data: {str(e)}"
            )
            raise

    async def dry_run_export_usage_data(self, limit: Optional[int] = 10000):
        """
        Returns the data that would be exported to CloudZero without actually sending it.

        Args:
            limit: Limit number of records to display (default: 10000)

        Returns:
            dict: Contains usage_data, cbf_data, and summary statistics
        """
        from litellm.integrations.cloudzero.database import LiteLLMDatabase
        from litellm.integrations.cloudzero.transform import CBFTransformer

        try:
            verbose_logger.debug("CloudZero Logger: Starting dry run export")

            # Initialize database connection and load data
            database = LiteLLMDatabase()
            verbose_logger.debug("CloudZero Logger: Loading usage data for dry run")
            data = await database.get_usage_data(limit=limit)

            if data.is_empty():
                verbose_logger.warning("CloudZero Dry Run: No usage data found")
                return {
                    "usage_data": [],
                    "cbf_data": [],
                    "summary": {
                        "total_records": 0,
                        "total_cost": 0,
                        "total_tokens": 0,
                        "unique_accounts": 0,
                        "unique_services": 0,
                    },
                }

            verbose_logger.debug(
                f"CloudZero Dry Run: Processing {len(data)} records..."
            )

            # Convert usage data to dict format for response
            usage_data_sample = data.head(50).to_dicts()  # Return first 50 rows

            # Transform data to CloudZero CBF format
            transformer = CBFTransformer()
            cbf_data = transformer.transform(data)

            if cbf_data.is_empty():
                verbose_logger.warning(
                    "CloudZero Dry Run: No valid data after transformation"
                )
                return {
                    "usage_data": usage_data_sample,
                    "cbf_data": [],
                    "summary": {
                        "total_records": len(usage_data_sample),
                        "total_cost": sum(
                            row.get("spend", 0) for row in usage_data_sample
                        ),
                        "total_tokens": sum(
                            row.get("prompt_tokens", 0)
                            + row.get("completion_tokens", 0)
                            for row in usage_data_sample
                        ),
                        "unique_accounts": 0,
                        "unique_services": 0,
                    },
                }

            # Convert CBF data to dict format for response
            cbf_data_dict = cbf_data.to_dicts()

            # Calculate summary statistics
            total_cost = sum(record.get("cost/cost", 0) for record in cbf_data_dict)
            unique_accounts = len(
                set(
                    record.get("resource/account", "")
                    for record in cbf_data_dict
                    if record.get("resource/account")
                )
            )
            unique_services = len(
                set(
                    record.get("resource/service", "")
                    for record in cbf_data_dict
                    if record.get("resource/service")
                )
            )
            total_tokens = sum(
                record.get("usage/amount", 0) for record in cbf_data_dict
            )

            verbose_logger.debug(
                f"CloudZero Logger: Dry run completed for {len(cbf_data)} records"
            )

            return {
                "usage_data": usage_data_sample,
                "cbf_data": cbf_data_dict,
                "summary": {
                    "total_records": len(cbf_data_dict),
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "unique_accounts": unique_accounts,
                    "unique_services": unique_services,
                },
            }

        except Exception as e:
            verbose_logger.error(f"CloudZero Logger: Error in dry run export: {str(e)}")
            verbose_logger.error(f"CloudZero Dry Run Error: {str(e)}")
            raise

    def _display_cbf_data_on_screen(self, cbf_data):
        """Display CBF transformed data in a formatted table on screen."""
        from rich.box import SIMPLE
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if cbf_data.is_empty():
            console.print("[yellow]No CBF data to display[/yellow]")
            return

        console.print(
            f"\n[bold green]💰 CloudZero CBF Transformed Data ({len(cbf_data)} records)[/bold green]"
        )

        # Convert to dicts for easier processing
        records = cbf_data.to_dicts()

        # Create main CBF table
        cbf_table = Table(
            show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1)
        )
        cbf_table.add_column("time/usage_start", style="blue", no_wrap=False)
        cbf_table.add_column("cost/cost", style="green", justify="right", no_wrap=False)
        cbf_table.add_column(
            "entity_type", style="magenta", justify="right", no_wrap=False
        )
        cbf_table.add_column(
            "entity_id", style="magenta", justify="right", no_wrap=False
        )
        cbf_table.add_column("team_id", style="cyan", no_wrap=False)
        cbf_table.add_column("team_alias", style="cyan", no_wrap=False)
        cbf_table.add_column("api_key_alias", style="yellow", no_wrap=False)
        cbf_table.add_column(
            "usage/amount", style="yellow", justify="right", no_wrap=False
        )
        cbf_table.add_column("resource/id", style="magenta", no_wrap=False)
        cbf_table.add_column("resource/service", style="cyan", no_wrap=False)
        cbf_table.add_column("resource/account", style="white", no_wrap=False)
        cbf_table.add_column("resource/region", style="dim", no_wrap=False)

        for record in records:
            # Use proper CBF field names
            time_usage_start = str(record.get("time/usage_start", "N/A"))
            cost_cost = str(record.get("cost/cost", 0))
            usage_amount = str(record.get("usage/amount", 0))
            resource_id = str(record.get("resource/id", "N/A"))
            resource_service = str(record.get("resource/service", "N/A"))
            resource_account = str(record.get("resource/account", "N/A"))
            resource_region = str(record.get("resource/region", "N/A"))
            entity_type = str(record.get("entity_type", "N/A"))
            entity_id = str(record.get("entity_id", "N/A"))
            team_id = str(record.get("resource/tag:team_id", "N/A"))
            team_alias = str(record.get("resource/tag:team_alias", "N/A"))
            api_key_alias = str(record.get("resource/tag:api_key_alias", "N/A"))

            cbf_table.add_row(
                time_usage_start,
                cost_cost,
                entity_type,
                entity_id,
                team_id,
                team_alias,
                api_key_alias,
                usage_amount,
                resource_id,
                resource_service,
                resource_account,
                resource_region,
            )

        console.print(cbf_table)

        # Show summary statistics
        total_cost = sum(record.get("cost/cost", 0) for record in records)
        unique_accounts = len(
            set(
                record.get("resource/account", "")
                for record in records
                if record.get("resource/account")
            )
        )
        unique_services = len(
            set(
                record.get("resource/service", "")
                for record in records
                if record.get("resource/service")
            )
        )

        # Count total tokens from usage metrics
        total_tokens = sum(record.get("usage/amount", 0) for record in records)

        console.print("\n[bold blue]📊 CBF Summary[/bold blue]")
        console.print(f"  Records: {len(records):,}")
        console.print(f"  Total Cost: ${total_cost:.2f}")
        console.print(f"  Total Tokens: {total_tokens:,}")
        console.print(f"  Unique Accounts: {unique_accounts}")
        console.print(f"  Unique Services: {unique_services}")

        console.print(
            "\n[dim]💡 This is the CloudZero CBF format ready for AnyCost ingestion[/dim]"
        )

    @staticmethod
    async def init_cloudzero_background_job(scheduler: AsyncIOScheduler):
        """
        Initialize the CloudZero background job.

        Starts the background job that exports the usage data to CloudZero every hour.
        """
        from litellm.constants import CLOUDZERO_EXPORT_INTERVAL_MINUTES
        from litellm.integrations.custom_logger import CustomLogger

        prometheus_loggers: List[
            CustomLogger
        ] = litellm.logging_callback_manager.get_custom_loggers_for_type(
            callback_type=CloudZeroLogger
        )
        # we need to get the initialized prometheus logger instance(s) and call logger.initialize_remaining_budget_metrics() on them
        verbose_logger.debug("found %s cloudzero loggers", len(prometheus_loggers))
        if len(prometheus_loggers) > 0:
            cloudzero_logger = cast(CloudZeroLogger, prometheus_loggers[0])
            verbose_logger.debug(
                "Initializing remaining budget metrics as a cron job executing every %s minutes"
                % CLOUDZERO_EXPORT_INTERVAL_MINUTES
            )
            scheduler.add_job(
                cloudzero_logger.initialize_cloudzero_export_job,
                "interval",
                minutes=CLOUDZERO_EXPORT_INTERVAL_MINUTES,
            )
