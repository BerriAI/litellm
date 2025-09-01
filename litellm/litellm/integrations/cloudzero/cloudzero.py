import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

from .cz_stream_api import CloudZeroStreamer
from .database import LiteLLMDatabase
from .transform import CBFTransformer


class CloudZeroLogger(CustomLogger):
    """
    CloudZero Logger for exporting LiteLLM usage data to CloudZero AnyCost API.
    
    Environment Variables:
        CLOUDZERO_API_KEY: CloudZero API key for authentication
        CLOUDZERO_CONNECTION_ID: CloudZero connection ID for data submission
        CLOUDZERO_TIMEZONE: Timezone for date handling (default: UTC)
    """

    def __init__(self, api_key: Optional[str] = None, connection_id: Optional[str] = None, timezone: Optional[str] = None, **kwargs):
        """Initialize CloudZero logger with configuration from parameters or environment variables."""
        super().__init__(**kwargs)
        
        # Get configuration from parameters first, fall back to environment variables
        self.api_key = api_key or os.getenv("CLOUDZERO_API_KEY")
        self.connection_id = connection_id or os.getenv("CLOUDZERO_CONNECTION_ID") 
        self.timezone = timezone or os.getenv("CLOUDZERO_TIMEZONE", "UTC")

    async def export_usage_data(self, target_hour: datetime, limit: Optional[int] = 1000, operation: str = "replace_hourly"):
        """
        Exports the usage data for a specific hour to CloudZero.

        - Reads spend logs from the DB for the specified hour
        - Transforms the data to the CloudZero format
        - Sends the data to CloudZero
        
        Args:
            target_hour: The specific hour to export data for
            limit: Optional limit on number of records to export (default: 1000)
            operation: CloudZero operation type ("replace_hourly" or "sum")
        """
        try:
            verbose_logger.debug("CloudZero Logger: Starting usage data export")
            
            # Validate required configuration
            if not self.api_key or not self.connection_id:
                raise ValueError(
                    "CloudZero configuration missing. Please set CLOUDZERO_API_KEY and CLOUDZERO_CONNECTION_ID environment variables."
                )

            # Fetch and transform data using helper
            cbf_data = await self._fetch_cbf_data_for_hour(target_hour, limit)
            
            if cbf_data.is_empty():
                verbose_logger.info("CloudZero Logger: No usage data found to export")
                return

            # Send data to CloudZero
            streamer = CloudZeroStreamer(
                api_key=self.api_key,
                connection_id=self.connection_id,
                user_timezone=self.timezone
            )
            
            verbose_logger.debug(f"CloudZero Logger: Transmitting {len(cbf_data)} records to CloudZero")
            streamer.send_batched(cbf_data, operation=operation)
            
            verbose_logger.info(f"CloudZero Logger: Successfully exported {len(cbf_data)} records to CloudZero")
            
        except Exception as e:
            verbose_logger.error(f"CloudZero Logger: Error exporting usage data: {str(e)}")
            raise

    async def _fetch_cbf_data_for_hour(self, target_hour: datetime, limit: Optional[int] = 1000):
        """
        Helper method to fetch usage data for a specific hour and transform it to CloudZero CBF format.
        
        Args:
            target_hour: The specific hour to fetch data for
            limit: Optional limit on number of records to fetch (default: 1000)
            
        Returns:
            CBF formatted data ready for CloudZero ingestion
        """
        # Initialize database connection and load data
        database = LiteLLMDatabase()
        verbose_logger.debug(f"CloudZero Logger: Loading spend logs for hour {target_hour}")
        data = await database.get_usage_data_for_hour(target_hour=target_hour, limit=limit)
        
        if data.is_empty():
            verbose_logger.info("CloudZero Logger: No usage data found for the specified hour")
            return data  # Return empty data

        verbose_logger.debug(f"CloudZero Logger: Processing {len(data)} records")
        
        # Transform data to CloudZero CBF format
        transformer = CBFTransformer()
        cbf_data = transformer.transform(data)
        
        if cbf_data.is_empty():
            verbose_logger.warning("CloudZero Logger: No valid data after transformation")
            
        return cbf_data

    async def dry_run_export_usage_data(self, target_hour: datetime, limit: Optional[int] = 1000):
        """
        Only prints the spend logs data for a specific hour that would be exported to CloudZero.
        
        Args:
            target_hour: The specific hour to export data for
            limit: Limit number of records to display (default: 1000)
        """
        try:
            verbose_logger.debug("CloudZero Logger: Starting dry run export")
            
            # Fetch and transform data using helper
            cbf_data = await self._fetch_cbf_data_for_hour(target_hour, limit)
            
            if cbf_data.is_empty():
                verbose_logger.warning("CloudZero Dry Run: No usage data found")
                return

            # Display the transformed data on screen
            self._display_cbf_data_on_screen(cbf_data)
            
            verbose_logger.info(f"CloudZero Logger: Dry run completed for {len(cbf_data)} records")
            
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

        console.print(f"\n[bold green]💰 CloudZero CBF Transformed Data ({len(cbf_data)} records)[/bold green]")

        # Convert to dicts for easier processing
        records = cbf_data.to_dicts()

        # Create main CBF table
        cbf_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        cbf_table.add_column("time/usage_start", style="blue", no_wrap=False)
        cbf_table.add_column("cost/cost", style="green", justify="right", no_wrap=False)
        cbf_table.add_column("usage/amount", style="yellow", justify="right", no_wrap=False)
        cbf_table.add_column("resource/id", style="magenta", no_wrap=False)
        cbf_table.add_column("resource/service", style="cyan", no_wrap=False)
        cbf_table.add_column("resource/account", style="white", no_wrap=False)
        cbf_table.add_column("resource/region", style="dim", no_wrap=False)

        for record in records:
            # Use proper CBF field names
            time_usage_start = str(record.get('time/usage_start', 'N/A'))
            cost_cost = str(record.get('cost/cost', 0))
            usage_amount = str(record.get('usage/amount', 0))
            resource_id = str(record.get('resource/id', 'N/A'))
            resource_service = str(record.get('resource/service', 'N/A'))
            resource_account = str(record.get('resource/account', 'N/A'))
            resource_region = str(record.get('resource/region', 'N/A'))

            cbf_table.add_row(
                time_usage_start,
                cost_cost,
                usage_amount,
                resource_id,
                resource_service,
                resource_account,
                resource_region
            )

        console.print(cbf_table)

        # Show summary statistics
        total_cost = sum(record.get('cost/cost', 0) for record in records)
        unique_accounts = len(set(record.get('resource/account', '') for record in records if record.get('resource/account')))
        unique_services = len(set(record.get('resource/service', '') for record in records if record.get('resource/service')))

        # Count total tokens from usage metrics
        total_tokens = sum(record.get('usage/amount', 0) for record in records)

        console.print("\n[bold blue]📊 CBF Summary[/bold blue]")
        console.print(f"  Records: {len(records):,}")
        console.print(f"  Total Cost: ${total_cost:.2f}")
        console.print(f"  Total Tokens: {total_tokens:,}")
        console.print(f"  Unique Accounts: {unique_accounts}")
        console.print(f"  Unique Services: {unique_services}")

        console.print("\n[dim]💡 This is the CloudZero CBF format ready for AnyCost ingestion[/dim]")

    async def init_background_job(self, redis_cache=None):
        """
        Initialize a background job that exports usage data every hour.
        Uses PodLockManager to ensure only one instance runs the export at a time.
        
        Args:
            redis_cache: Redis cache instance for pod locking
        """
        from litellm.proxy.db.db_transaction_queue.pod_lock_manager import (
            PodLockManager,
        )
        
        lock_manager = PodLockManager(redis_cache=redis_cache)
        cronjob_id = "cloudzero_hourly_export"
        
        async def hourly_export_task():
            while True:
                try:
                    # Calculate the previous completed hour
                    now = datetime.utcnow()
                    target_hour = now.replace(minute=0, second=0, microsecond=0)
                    # Export data for the previous hour to ensure all data is available
                    target_hour = target_hour - timedelta(hours=1)
                    
                    # Try to acquire lock
                    lock_acquired = await lock_manager.acquire_lock(cronjob_id)
                    
                    if lock_acquired:
                        try:
                            verbose_logger.info(f"CloudZero Background Job: Starting export for hour {target_hour}")
                            await self.export_usage_data(target_hour)
                            verbose_logger.info(f"CloudZero Background Job: Completed export for hour {target_hour}")
                        finally:
                            # Always release the lock
                            await lock_manager.release_lock(cronjob_id)
                    else:
                        verbose_logger.debug("CloudZero Background Job: Another instance is already running the export")
                    
                    # Wait until the next hour
                    next_hour = (datetime.utcnow() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    sleep_seconds = (next_hour - datetime.utcnow()).total_seconds()
                    await asyncio.sleep(sleep_seconds)
                    
                except Exception as e:
                    verbose_logger.error(f"CloudZero Background Job: Error in hourly export task: {str(e)}")
                    # Sleep for 5 minutes before retrying on error
                    await asyncio.sleep(300)
        
        # Start the background task
        asyncio.create_task(hourly_export_task())
        verbose_logger.debug("CloudZero Background Job: Initialized hourly export task")