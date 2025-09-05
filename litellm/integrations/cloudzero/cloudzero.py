import os
from datetime import datetime
from typing import Optional

import polars as pl

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

    async def init_background_job(self):
        """
        Initialize the background job.
        """
        pass

    async def export_usage_data(
        self, 
        limit: Optional[int] = None, 
        operation: str = "replace_hourly",
        start_time_utc: Optional[datetime] = None,
        end_time_utc: Optional[datetime] = None
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
                limit=limit,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc
            )
            
            if data.is_empty():
                verbose_logger.info("CloudZero Logger: No usage data found to export")
                return

            verbose_logger.debug(f"CloudZero Logger: Processing {len(data)} records")
            
            # Transform data to CloudZero CBF format
            transformer = CBFTransformer()
            cbf_data = transformer.transform(data)
            
            if cbf_data.is_empty():
                verbose_logger.warning("CloudZero Logger: No valid data after transformation")
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

    async def dry_run_export_usage_data(self, limit: Optional[int] = 10000):
        """
        Only prints the data that would be exported to CloudZero.
        
        Args:
            limit: Limit number of records to display (default: 10000)
        """
        try:
            verbose_logger.debug("CloudZero Logger: Starting dry run export")
            
            # Initialize database connection and load data
            database = LiteLLMDatabase()
            verbose_logger.debug("CloudZero Logger: Loading usage data for dry run")
            data = await database.get_usage_data(limit=limit)
            
            if data.is_empty():
                verbose_logger.warning("CloudZero Dry Run: No usage data found")
                return

            verbose_logger.debug(f"CloudZero Dry Run: Processing {len(data)} records...")
            
            # Pretty print first 20 rows of usage data with team information highlighted
            from rich.box import SIMPLE
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            console.print(f"\n[bold blue]📊 Usage Data Sample (first 20 rows):[/bold blue]")
            
            # Create a focused table showing key fields including team info
            usage_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
            usage_table.add_column("Date", style="blue", no_wrap=True)
            usage_table.add_column("Model", style="green", no_wrap=True)
            usage_table.add_column("Provider", style="cyan", no_wrap=True)
            usage_table.add_column("Team ID", style="magenta", no_wrap=True)
            usage_table.add_column("Team Alias", style="magenta", no_wrap=True)
            usage_table.add_column("API Key Alias", style="yellow", no_wrap=True)
            usage_table.add_column("Tokens", style="white", justify="right")
            usage_table.add_column("Cost", style="green", justify="right")
            usage_table.add_column("Requests", style="dim", justify="right")
            
            # Show first 20 rows with key information
            sample_data = data.head(20).to_dicts()
            for row in sample_data:
                usage_table.add_row(
                    str(row.get('date', 'N/A'))[:10],  # Just the date part
                    str(row.get('model', 'N/A'))[:20],  # Truncate long model names
                    str(row.get('custom_llm_provider', 'N/A'))[:15],
                    str(row.get('team_id', 'null'))[:20],
                    str(row.get('team_alias', 'null'))[:20],
                    str(row.get('api_key_alias', 'N/A'))[:20] if row.get('api_key_alias') else 'N/A',
                    f"{row.get('prompt_tokens', 0) + row.get('completion_tokens', 0):,}",
                    f"${row.get('spend', 0):.4f}",
                    str(row.get('successful_requests', 0))
                )
            
            console.print(usage_table)
            
            # Also show the full polars output for debugging
            console.print(f"\n[bold dim]📋 Full Data Schema (first 5 rows):[/bold dim]")
            with pl.Config(
                tbl_rows=5,
                tbl_cols=20,
                fmt_str_lengths=30,
                tbl_width_chars=150
            ):
                console.print(str(data.head(5)))

            # Transform data to CloudZero CBF format


            transformer = CBFTransformer()
            cbf_data = transformer.transform(data)
            
            if cbf_data.is_empty():
                verbose_logger.warning("CloudZero Dry Run: No valid data after transformation")
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
        cbf_table.add_column("entity_type", style="magenta", justify="right", no_wrap=False)
        cbf_table.add_column("entity_id", style="magenta", justify="right", no_wrap=False)
        cbf_table.add_column("team_id", style="cyan", no_wrap=False)
        cbf_table.add_column("team_alias", style="cyan", no_wrap=False)
        cbf_table.add_column("api_key_alias", style="yellow", no_wrap=False)
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
            entity_type = str(record.get('entity_type', 'N/A'))
            entity_id = str(record.get('entity_id', 'N/A'))
            team_id = str(record.get('resource/tag:team_id', 'N/A'))
            team_alias = str(record.get('resource/tag:team_alias', 'N/A'))
            api_key_alias = str(record.get('resource/tag:api_key_alias', 'N/A'))

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