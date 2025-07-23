# Copyright 2025 CloudZero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGELOG: 2025-01-19 - Added configuration file support and --version option (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from click to typer for better CLI experience (erik.peterson)
# CHANGELOG: 2025-01-19 - Added pathlib for file operations (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial CLI implementation with click (erik.peterson)

"""Command line interface for LiteLLM to CloudZero ETL tool."""

import asyncio
import json
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import polars as pl
import typer
from rich.console import Console

from .analysis import DataAnalyzer
from .config import Config
from .database import LiteLLMDatabase
from .output import CloudZeroStreamer
from .transform import CBFTransformer

app = typer.Typer(
    name="litellm-cz-etl",
    help="Transform LiteLLM database data into CloudZero AnyCost CBF format",
    rich_markup_mode="rich"
)
console = Console()


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print("ll2cz version 0.1.0")
        raise typer.Exit()




config_app = typer.Typer(help="Configuration management commands")


@config_app.command("example")
def config_example():
    """Create an example configuration file at ~/.ll2cz/config.yml."""
    config = Config()
    config.create_example_config()


@config_app.command("status")
def config_status():
    """Show current configuration status."""
    config = Config()
    config.show_config_status()


@config_app.command("edit")
def config_edit():
    """Interactively edit configuration values."""
    config = Config()
    config.interactive_edit_config()


app.add_typer(config_app, name="config")


@app.command()
def transform(
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    output_file: Annotated[Optional[str], typer.Option("--output", help="Output CSV file name")] = None,
    screen: Annotated[bool, typer.Option("--screen", help="Display transformed data on screen in a formatted table")] = False,
    limit: Annotated[int, typer.Option("--limit", help="Limit number of records to transform for screen output")] = 10000,
) -> None:
    """Transform LiteLLM database data into CloudZero AnyCost CBF format."""

    # Load configuration and merge with CLI arguments (CLI takes priority)
    config = Config()
    db_connection = config.get_database_connection(db_connection)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config-example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    try:
        database = LiteLLMDatabase()

        console.print("[blue]Loading data from LiteLLM database...[/blue]")
        # Limit data if screen output is requested
        data_limit = limit if screen else None
        data = asyncio.run(database.get_usage_data(limit=data_limit))

        if data.is_empty():
            console.print("[yellow]No data found in database[/yellow]")
            return

        console.print(f"[blue]Processing {len(data)} records...[/blue]")
        transformer = CBFTransformer()
        cbf_data = transformer.transform(data)

        if screen:
            _display_cbf_data_on_screen(cbf_data)
        else:
            console.print("[red]Error: Must specify either --screen or --output[/red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def transmit(
    mode: Annotated[str, typer.Argument(help="Transmission mode: 'day' for single day, 'month' for month, or 'all' for all data")],
    date_spec: Annotated[Optional[str], typer.Argument(help="Date specification: DD-MM-YYYY for day mode, MM-YYYY for month mode, ignored for all mode")] = None,
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    cz_api_key: Annotated[Optional[str], typer.Option("--cz-api-key", help="CloudZero API key")] = None,
    cz_connection_id: Annotated[Optional[str], typer.Option("--cz-connection-id", help="CloudZero connection ID")] = None,
    append: Annotated[bool, typer.Option("--append", help="Use 'sum' operation to append data instead of 'replace_hourly'")] = False,
    timezone: Annotated[Optional[str], typer.Option("--timezone", help="Timezone for date handling (e.g., 'US/Eastern', 'UTC'). Defaults to UTC")] = None,
    test: Annotated[bool, typer.Option("--test", help="Test mode: process only 5 records and show JSON payloads instead of transmitting (API keys still required)")] = False,
    limit: Annotated[Optional[int], typer.Option("--limit", help="Limit number of records to process (default: all records)")] = None,
) -> None:
    """Transform LiteLLM data and transmit to CloudZero AnyCost API.
    
    MODES:
      day    - Send data for a specific day (default: today, or specify DD-MM-YYYY)
      month  - Send data for a specific month (default: current month, or specify MM-YYYY) 
      all    - Send all available data (batched by day)
      
    EXAMPLES:
      ll2cz transmit day                    # Send today's data
      ll2cz transmit day 15-01-2024         # Send data for January 15, 2024
      ll2cz transmit month                  # Send current month's data
      ll2cz transmit month 01-2024          # Send January 2024 data
      ll2cz transmit all                    # Send all data in daily batches
    """

    # Validate mode parameter
    if mode not in ['day', 'month', 'all']:
        console.print("[red]Error: Mode must be 'day', 'month', or 'all'[/red]")
        raise typer.Exit(1)

    # Load configuration and merge with CLI arguments (CLI takes priority)
    config = Config()
    db_connection = config.get_database_connection(db_connection)
    cz_api_key = config.get_cz_api_key(cz_api_key)
    cz_connection_id = config.get_cz_connection_id(cz_connection_id)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    if not cz_api_key or not cz_connection_id:
        console.print("[red]Error: --cz-api-key and --cz-connection-id are required[/red]")
        console.print("[blue]You can set them via CLI arguments or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    # Set up timezone
    user_timezone = timezone or 'UTC'

    try:
        # Parse date specification and determine date range
        date_filter = _parse_date_specification(mode, date_spec, user_timezone)

        database = LiteLLMDatabase()

        console.print(f"[blue]Loading {mode} data from LiteLLM database...[/blue]")
        if date_filter:
            console.print(f"[dim]Date filter: {date_filter.get('description', 'Unknown filter')}[/dim]")

        # Load data with appropriate filtering
        if test:
            data = asyncio.run(database.get_usage_data(limit=5))
        else:
            data = _load_filtered_data(database, date_filter, limit)

        if data.is_empty():
            console.print("[yellow]No data found for the specified criteria[/yellow]")
            return

        console.print(f"[blue]Processing {len(data)} records...[/blue]")
        transformer = CBFTransformer()
        cbf_data = transformer.transform(data)

        # Determine operation mode
        operation = "sum" if append else "replace_hourly"

        if test:
            _display_enhanced_test_payloads(cbf_data, operation, mode)
        else:
            console.print(f"[blue]Transmitting to CloudZero AnyCost API using operation: '{operation}'[/blue]")
            streamer = CloudZeroStreamer(cz_api_key, cz_connection_id, user_timezone)
            streamer.send_batched(cbf_data, operation=operation)
            console.print(f"[green]✓ Successfully transmitted {len(cbf_data)} records to CloudZero AnyCost API[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


analyze_app = typer.Typer(help="Analysis and data exploration commands")


@analyze_app.command("data")
def analyze_data(
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of records to analyze")] = 10000,
    json_output: Annotated[Optional[str], typer.Option("--json", help="JSON output file for analysis results")] = None,
    force_refresh: Annotated[bool, typer.Option("--force-refresh", help="Force refresh cache from server")] = False,
    show_raw: Annotated[bool, typer.Option("--show-raw", help="Show raw data tables instead of analysis")] = False,
    table: Annotated[Optional[str], typer.Option("--table", help="Show specific table only (for --show-raw): 'user', 'team', 'tag', or 'all'")] = "all",
) -> None:
    """Analyze LiteLLM database data and show insights, or display raw data tables."""

    # Load configuration
    config = Config()
    db_connection = config.get_database_connection(db_connection)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config-example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    if show_raw and table not in ["all", "user", "team", "tag"]:
        console.print("[red]Error: --table must be one of: all, user, team, tag[/red]")
        raise typer.Exit(1)

    try:
        # Use cached database
        database = LiteLLMDatabase()


        if show_raw:
            # Show raw data tables (former show-data functionality)
            if table == "all":
                console.print(f"[blue]Showing {limit:,} records from each LiteLLM table...[/blue]")
                _show_all_tables_data_cached(database, limit, force_refresh)
            else:
                table_title: str = table.title() if table else "Unknown"
                console.print(f"[blue]Showing {limit:,} records from LiteLLM_{table_title}Spend table...[/blue]")
                _show_single_table_data_cached(database, table, limit, force_refresh)  # type: ignore[arg-type]
        else:
            # Show data analysis (former analysis functionality)
            console.print(f"[blue]Running data analysis on {limit:,} records...[/blue]")
            analyzer = DataAnalyzer(database)
            results = analyzer.analyze(limit=limit, force_refresh=force_refresh)

            console.print("\n[bold]Analysis Results:[/bold]")
            console.print("=" * 50)
            analyzer.print_results(results)

            if json_output:
                json_path = Path(json_output)
                json_path.write_text(json.dumps(results, indent=2, default=str))
                console.print(f"[green]Analysis results saved to {json_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@analyze_app.command("czrn")
def analyze_czrn(
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of records to analyze for CZRN generation")] = 10000,
    force_refresh: Annotated[bool, typer.Option("--force-refresh", help="Force refresh cache from server")] = False,
) -> None:
    """Analyze CZRN generation from LiteLLM database data."""

    # Load configuration
    config = Config()
    db_connection = config.get_database_connection(db_connection)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config-example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    try:
        database = LiteLLMDatabase()

        console.print(f"[blue]Running CZRN analysis on {limit:,} records...[/blue]")
        analyzer = DataAnalyzer(database)
        analyzer.czrn_analysis(limit=limit, force_refresh=force_refresh)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)




def _show_all_tables_data_cached(database: LiteLLMDatabase, limit: int, force_refresh: bool) -> None:
    """Show data from all three LiteLLM tables individually using cache."""

    # If force refresh, make sure we refresh the cache first
    if force_refresh:
        try:
            _ = asyncio.run(database.get_usage_data(limit=1))
        except Exception:
            pass  # Ignore errors during refresh, will show cached data

    # Show table breakdown first
    table_info = asyncio.run(database.get_table_info())
    breakdown = table_info['table_breakdown']
    console.print("\n[bold blue]📊 Table Overview[/bold blue]")
    console.print(f"  User records: {breakdown['user_spend']:,}")
    console.print(f"  Team records: {breakdown['team_spend']:,}")
    console.print(f"  Tag records: {breakdown['tag_spend']:,}")
    console.print(f"  Total records: {sum(breakdown.values()):,}")

    # Show each table individually
    tables_to_show = [
        ("user", "LiteLLM_DailyUserSpend", "👤"),
        ("team", "LiteLLM_DailyTeamSpend", "👥"),
        ("tag", "LiteLLM_DailyTagSpend", "🏷️")
    ]

    for table_type, table_name, emoji in tables_to_show:
        if breakdown[f'{table_type}_spend'] > 0:
            console.print(f"\n[bold green]{emoji} {table_name}[/bold green]")
            _show_single_table_data_cached(database, table_type, limit, force_refresh)
        else:
            console.print(f"\n[bold yellow]{emoji} {table_name}[/bold yellow]")
            console.print(f"[dim]No records found in {table_name}[/dim]")


def _show_single_table_data_cached(database: LiteLLMDatabase, table_type: str, limit: int, force_refresh: bool) -> None:
    """Show data from a specific LiteLLM table using cache."""
    from rich.box import SIMPLE
    from rich.table import Table

    table_name = f"LiteLLM_Daily{table_type.title()}Spend"
    console.print(f"\n[bold green]📋 Raw Data from {table_name}[/bold green]")

    # Get data from cache filtered by entity type
    all_data = asyncio.run(database.get_usage_data(limit=limit))
    # Filter by entity type
    data = all_data.filter(pl.col('entity_type') == table_type) if not all_data.is_empty() else all_data

    if not data.is_empty():
        table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))

        # Add columns dynamically based on data
        for col in data.columns:
            table.add_column(col, style="white", no_wrap=False)

        # Add rows
        for row in data.to_dicts():
            table.add_row(*[str(row.get(col, '')) for col in data.columns])

        console.print(table)
        console.print(f"[dim]💡 Showing {len(data):,} records from {table_name}[/dim]")
    else:
        console.print(f"[yellow]No data found in {table_name}[/yellow]")



@analyze_app.command("spend")
def analyze_spend(
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    limit: Annotated[int, typer.Option("--limit", help="Number of records to analyze for spend analysis")] = 10000,
    force_refresh: Annotated[bool, typer.Option("--force-refresh", help="Force refresh cache from server")] = False,
) -> None:
    """Analyze spending patterns based on LiteLLM team and user data."""

    # Load configuration
    config = Config()
    db_connection = config.get_database_connection(db_connection)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config-example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    try:
        # Use cached database
        database = LiteLLMDatabase()

        console.print(f"[blue]Running spend analysis on {limit:,} records...[/blue]")
        analyzer = DataAnalyzer(database)
        analyzer.spend_analysis(limit=limit, force_refresh=force_refresh)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@analyze_app.command("schema")
def analyze_schema(
    db_connection: Annotated[Optional[str], typer.Option("--input", help="LiteLLM PostgreSQL database connection URL")] = None,
    output_file: Annotated[Optional[str], typer.Option("--output", help="Output file for schema documentation")] = None,
) -> None:
    """Discover and document all tables in the LiteLLM database."""

    # Load configuration
    config = Config()
    db_connection = config.get_database_connection(db_connection)

    if not db_connection:
        console.print("[red]Error: --input (database connection) is required[/red]")
        console.print("[blue]You can set it via CLI argument or in ~/.ll2cz/config.yml[/blue]")
        console.print("[blue]Run 'll2cz config-example' to create a sample config file[/blue]")
        raise typer.Exit(1)

    try:
        database = LiteLLMDatabase()

        console.print("[blue]Discovering all LiteLLM tables in database...[/blue]")
        schema_info = asyncio.run(database.discover_all_tables())

        # Display summary
        console.print("\n[bold green]📋 Database Schema Discovery Complete[/bold green]")
        console.print(f"  Total LiteLLM tables found: {schema_info['table_count']}")

        # Display table overview
        _display_schema_overview(schema_info)

        # Generate documentation if requested
        if output_file:
            _generate_complete_schema_docs(schema_info, output_file)
            console.print(f"\n[green]📄 Complete schema documentation saved to {output_file}[/green]")
        else:
            console.print("\n[dim]💡 Use --output filename.md to save complete documentation[/dim]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


app.add_typer(analyze_app, name="analyze")


def _display_schema_overview(schema_info: dict) -> None:
    """Display a summary overview of discovered tables."""
    from rich.box import SIMPLE
    from rich.table import Table

    console.print("\n[bold cyan]📊 Table Overview[/bold cyan]")

    table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
    table.add_column("Table Name", style="green", no_wrap=False)
    table.add_column("Columns", style="blue", justify="right", no_wrap=False)
    table.add_column("Row Count", style="yellow", justify="right", no_wrap=False)
    table.add_column("Primary Keys", style="magenta", no_wrap=False)
    table.add_column("Foreign Keys", style="red", justify="right", no_wrap=False)
    table.add_column("Indexes", style="cyan", justify="right", no_wrap=False)

    for table_name, table_info in schema_info['tables'].items():
        fk_count = len(table_info['foreign_keys'])
        idx_count = len(table_info['indexes'])
        col_count = len(table_info['columns'])

        pk_display = ', '.join(table_info['primary_keys']) if table_info['primary_keys'] else 'None'

        table.add_row(
            table_name,
            str(col_count),
            f"{table_info['row_count']:,}",
            pk_display,
            str(fk_count),
            str(idx_count)
        )

    console.print(table)


def _generate_complete_schema_docs(schema_info: dict, output_file: str) -> None:
    """Generate complete schema documentation file."""
    from pathlib import Path

    # Create comprehensive documentation
    doc_content = _create_complete_documentation(schema_info)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc_content)


def _display_cbf_data_on_screen(cbf_data: pl.DataFrame) -> None:
    """Display CBF transformed data in a nicely formatted table on screen."""
    if cbf_data.is_empty():
        console.print("[yellow]No CBF data to display[/yellow]")
        return

    console.print(f"\n[bold green]💰 CloudZero CBF Transformed Data ({len(cbf_data)} records)[/bold green]")

    # Convert to dicts for easier processing
    records = cbf_data.to_dicts()

    # Create main CBF table
    from rich.box import SIMPLE
    from rich.table import Table

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



def _create_complete_documentation(schema_info: dict) -> str:
    """Create complete documentation content."""
    doc = []
    doc.append("# Complete LiteLLM Database Schema Documentation")
    doc.append("")
    doc.append("## Overview")
    doc.append("")
    doc.append(f"This document provides a comprehensive overview of all {schema_info['table_count']} LiteLLM tables discovered in the database.")
    doc.append("")
    doc.append("## Table Summary")
    doc.append("")
    doc.append("| Table Name | Columns | Rows | Primary Keys | Foreign Keys | Indexes |")
    doc.append("|------------|---------|------|--------------|--------------|---------|")

    for table_name, table_info in schema_info['tables'].items():
        pk_display = ', '.join(table_info['primary_keys']) if table_info['primary_keys'] else 'None'
        doc.append(f"| {table_name} | {len(table_info['columns'])} | {table_info['row_count']:,} | {pk_display} | {len(table_info['foreign_keys'])} | {len(table_info['indexes'])} |")

    doc.append("")
    doc.append("## Detailed Table Schemas")
    doc.append("")

    for table_name, table_info in schema_info['tables'].items():
        doc.append(f"### {table_name}")
        doc.append("")
        doc.append(f"**Row Count:** {table_info['row_count']:,}")
        doc.append("")

        # Columns
        doc.append("#### Columns")
        doc.append("")
        doc.append("| Column | Type | Nullable | Default | Length/Precision |")
        doc.append("|--------|------|----------|---------|------------------|")

        for col in table_info['columns']:
            nullable = "YES" if col['is_nullable'] == 'YES' else "NO"
            default = col['column_default'] or 'None'
            if len(default) > 30:
                default = default[:27] + "..."

            type_info = col['data_type']
            if col['character_maximum_length']:
                type_info += f"({col['character_maximum_length']})"
            elif col['numeric_precision']:
                type_info += f"({col['numeric_precision']}"
                if col['numeric_scale']:
                    type_info += f",{col['numeric_scale']}"
                type_info += ")"

            doc.append(f"| {col['column_name']} | {type_info} | {nullable} | {default} | {col['character_maximum_length'] or col['numeric_precision'] or ''} |")

        # Primary Keys
        if table_info['primary_keys']:
            doc.append("")
            doc.append("#### Primary Keys")
            doc.append("")
            for pk in table_info['primary_keys']:
                doc.append(f"- `{pk}`")

        # Foreign Keys
        if table_info['foreign_keys']:
            doc.append("")
            doc.append("#### Foreign Keys")
            doc.append("")
            doc.append("| Column | References |")
            doc.append("|--------|------------|")
            for fk in table_info['foreign_keys']:
                doc.append(f"| {fk['column_name']} | {fk['foreign_table_name']}.{fk['foreign_column_name']} |")

        # Indexes
        if table_info['indexes']:
            doc.append("")
            doc.append("#### Indexes")
            doc.append("")
            doc.append("| Index Name | Columns | Unique |")
            doc.append("|------------|---------|--------|")
            for idx in table_info['indexes']:
                unique = "YES" if idx['is_unique'] else "NO"
                columns = ', '.join(idx['column_names']) if isinstance(idx['column_names'], list) else str(idx['column_names'])
                doc.append(f"| {idx['index_name']} | {columns} | {unique} |")

        doc.append("")
        doc.append("---")
        doc.append("")

    # ERD Section
    doc.append("## Entity Relationship Diagram")
    doc.append("")
    doc.append("```mermaid")
    doc.append("erDiagram")

    for table_name, table_info in schema_info['tables'].items():
        doc.append(f"    {table_name} {{")
        for col in table_info['columns']:
            type_display = col['data_type']
            pk_marker = " PK" if col['column_name'] in table_info['primary_keys'] else ""
            doc.append(f"        {type_display} {col['column_name']}{pk_marker}")
        doc.append("    }")
        doc.append("")

    # Add relationships
    for table_name, table_info in schema_info['tables'].items():
        for fk in table_info['foreign_keys']:
            doc.append(f"    {table_name} ||--o| {fk['foreign_table_name']} : {fk['column_name']}")

    doc.append("```")
    doc.append("")

    return '\n'.join(doc)


def _parse_date_specification(mode: str, date_spec: Optional[str], user_timezone: str) -> Optional[dict]:
    """Parse date specification and return filter criteria."""
    from datetime import datetime
    from datetime import timezone as tz

    try:
        # Set up timezone
        if user_timezone == 'UTC':
            tz_info = tz.utc
        else:
            try:
                tz_info = zoneinfo.ZoneInfo(user_timezone)
            except zoneinfo.ZoneInfoNotFoundError:
                console.print(f"[yellow]Warning: Unknown timezone '{user_timezone}', using UTC[/yellow]")
                tz_info = tz.utc

        now = datetime.now(tz_info)

        if mode == 'day':
            if date_spec:
                # Parse DD-MM-YYYY format
                try:
                    day_obj = datetime.strptime(date_spec, '%d-%m-%Y').replace(tzinfo=tz_info)
                    start_date = day_obj.strftime('%Y-%m-%d')
                    end_date = start_date  # Same day
                except ValueError:
                    console.print(f"[red]Error: Invalid date format '{date_spec}'. Use DD-MM-YYYY (e.g., 15-01-2024)[/red]")
                    raise typer.Exit(1)
            else:
                # Use current day
                start_date = now.strftime('%Y-%m-%d')
                end_date = start_date

            return {'start_date': start_date, 'end_date': end_date, 'description': f"Day: {start_date}"}

        elif mode == 'month':
            if date_spec:
                # Parse MM-YYYY format
                try:
                    month_obj = datetime.strptime(date_spec, '%m-%Y').replace(tzinfo=tz_info)
                    # Get first and last day of the month
                    start_date = month_obj.strftime('%Y-%m-01')
                    if month_obj.month == 12:
                        last_day = 31
                    else:
                        # Get last day of month
                        import datetime as dt
                        next_month = month_obj.replace(month=month_obj.month + 1, day=1)
                        last_day = (next_month - dt.timedelta(days=1)).day
                    end_date = month_obj.strftime(f'%Y-%m-{last_day:02d}')
                except ValueError:
                    console.print(f"[red]Error: Invalid month format '{date_spec}'. Use MM-YYYY (e.g., 01-2024)[/red]")
                    raise typer.Exit(1)
            else:
                # Use current month
                start_date = now.strftime('%Y-%m-01')
                # Get last day of current month
                if now.month == 12:
                    last_day = 31
                else:
                    import datetime as dt
                    next_month = now.replace(month=now.month + 1, day=1)
                    last_day = (next_month - dt.timedelta(days=1)).day
                end_date = now.strftime(f'%Y-%m-{last_day:02d}')

            return {'start_date': start_date, 'end_date': end_date, 'description': f"Month: {start_date} to {end_date}"}

        elif mode == 'all':
            # No date filtering for 'all' mode
            return {'description': "All available data"}

    except Exception as e:
        console.print(f"[red]Error parsing date specification: {e}[/red]")
        raise typer.Exit(1)


def _load_filtered_data(database, date_filter: Optional[dict], limit: Optional[int]):
    """Load data with optional date filtering."""
    import polars as pl

    # For now, load all data and filter in memory
    # In a production system, you'd want to filter at the database level
    data = asyncio.run(database.get_usage_data(limit=limit))

    if not date_filter or 'start_date' not in date_filter:
        return data

    # Filter data by date range if specified
    if 'date' in data.columns:
        start_date = date_filter['start_date']
        end_date = date_filter['end_date']

        # Convert to date strings for comparison
        filtered_data = data.filter(
            (pl.col('date') >= start_date) & (pl.col('date') <= end_date)
        )

        console.print(f"[dim]Filtered from {len(data)} to {len(filtered_data)} records for date range[/dim]")
        return filtered_data

    return data


def _display_enhanced_test_payloads(cbf_data: pl.DataFrame, operation: str, mode: str) -> None:
    """Display enhanced test payloads with operation and mode information."""
    if cbf_data.is_empty():
        console.print("[yellow]No CBF data to display[/yellow]")
        return

    console.print("\n[bold yellow]🧪 Enhanced Test Mode - CloudZero AnyCost API Payloads[/bold yellow]")
    console.print(f"[blue]Mode: {mode} | Operation: {operation} | Records: {len(cbf_data)}[/blue]")
    console.print("[dim]These payloads would be batched by day and sent to the CloudZero AnyCost API:[/dim]\n")

    # Show first few records as examples
    records = cbf_data.to_dicts()

    # Group by date for demonstration
    from collections import defaultdict
    daily_groups = defaultdict(list)

    for record in records:
        timestamp = record.get('time/usage_start', '')
        if timestamp:
            try:
                date_part = timestamp.split('T')[0] if 'T' in timestamp else timestamp[:10]
                daily_groups[date_part].append(record)
            except:
                daily_groups['unknown'].append(record)

    shown_batches = 0
    for date_key, batch_records in daily_groups.items():
        if shown_batches >= 3:  # Show max 3 batches
            break

        console.print(f"[bold cyan]Daily Batch for {date_key} ({len(batch_records)} records):[/bold cyan]")

        # Show the API payload structure
        month_str = date_key[:7] if len(date_key) >= 7 else datetime.now().strftime('%Y-%m')

        # Convert records to API format for display
        temp_streamer = CloudZeroStreamer("test", "test")
        api_records = []
        for record in batch_records[:2]:  # Show first 2 records as example
            api_record = temp_streamer._convert_cbf_to_api_format(record)
            if api_record:
                api_records.append(api_record)

        batch_payload = {
            'month': month_str,
            'operation': operation,
            'data': api_records
        }

        json_payload = json.dumps(batch_payload, indent=2, default=str)
        console.print(f"[white]{json_payload}[/white]")

        if len(batch_records) > 2:
            console.print(f"[dim]... and {len(batch_records) - 2} more records in this batch[/dim]")

        console.print()  # Add spacing
        shown_batches += 1

    if len(daily_groups) > 3:
        console.print(f"[dim]... and {len(daily_groups) - 3} more daily batches[/dim]")

    # Show summary
    total_records = len(records)
    total_batches = len(daily_groups)

    console.print("[bold green]📋 Test Summary[/bold green]")
    console.print(f"  Total records: {total_records}")
    console.print(f"  Daily batches: {total_batches}")
    console.print(f"  Operation: {operation}")
    console.print(f"  Mode: {mode}")
    console.print("\n[dim]💡 Remove --test flag to actually transmit these batches to CloudZero AnyCost API[/dim]")


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option("--version", callback=version_callback, help="Show version and exit")] = None,
) -> None:
    """LiteLLM to CloudZero ETL tool."""
    pass


if __name__ == '__main__':
    app()

