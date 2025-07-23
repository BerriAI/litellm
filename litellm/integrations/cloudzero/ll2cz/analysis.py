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
# CHANGELOG: 2025-01-19 - Added CBF transformation examples to analysis output (erik.peterson)
# CHANGELOG: 2025-01-19 - Updated analysis for daily spend tables and enhanced insights (erik.peterson)
# CHANGELOG: 2025-01-19 - Added rich formatting for enhanced terminal output (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars for data analysis (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial data analysis module for LiteLLM inspection (erik.peterson)

"""Data analysis module for LiteLLM database inspection."""

from typing import Any, Union

import polars as pl
from rich.console import Console
from rich.table import Table

from .cached_database import CachedLiteLLMDatabase
from .czrn import CZRNGenerator
from .database import LiteLLMDatabase
from .transform import CBFTransformer


class DataAnalyzer:
    """Analyze LiteLLM database data for inspection and validation."""

    def __init__(self, database: Union[LiteLLMDatabase, CachedLiteLLMDatabase]):
        """Initialize analyzer with database connection."""
        self.database = database
        self.console = Console()

    def analyze(self, limit: int = 10000, force_refresh: bool = False) -> dict[str, Any]:
        """Perform comprehensive analysis of LiteLLM data."""
        if isinstance(self.database, CachedLiteLLMDatabase):
            raw_data = self.database.get_usage_data(limit=limit, force_refresh=force_refresh)
        else:
            raw_data = self.database.get_usage_data(limit=limit)
        table_info = self.database.get_table_info()

        # Filter data and show filtering summary
        data, filter_summary = self._filter_successful_requests(raw_data)

        # Generate CBF transformation examples
        cbf_examples = []
        if not data.is_empty():
            transformer = CBFTransformer()
            sample_data = data.head(3)  # Transform first 3 records as examples
            cbf_data = transformer.transform(sample_data)
            cbf_examples = cbf_data.to_dicts() if not cbf_data.is_empty() else []

        return {
            'table_info': table_info,
            'data_summary': self._analyze_data_summary(data),
            'column_analysis': self._analyze_columns(data),
            'sample_records': data.head(5).to_dicts() if not data.is_empty() else [],
            'cbf_examples': cbf_examples,
            'filter_summary': filter_summary
        }

    def _analyze_data_summary(self, data: pl.DataFrame) -> dict[str, Any]:
        """Analyze basic data summary statistics for daily spend data."""
        if data.is_empty():
            return {'message': 'No data available'}

        columns = data.columns

        # Calculate total tokens from prompt + completion tokens
        total_tokens = 0
        if 'prompt_tokens' in columns and 'completion_tokens' in columns:
            total_tokens = int(data['prompt_tokens'].sum() + data['completion_tokens'].sum())

        # Get date range
        date_range = {}
        if 'date' in columns:
            date_range = {
                'start': str(data['date'].min()),
                'end': str(data['date'].max())
            }

        # Entity type breakdown
        entity_breakdown = {}
        if 'entity_type' in columns:
            entity_counts = data['entity_type'].value_counts()
            entity_breakdown = {
                row['entity_type']: row['count']
                for row in entity_counts.to_dicts()
            }

        return {
            'total_records_analyzed': len(data),
            'date_range': date_range,
            'total_spend': (
                float(data['spend'].sum())
                if 'spend' in columns else None
            ),
            'total_tokens': total_tokens,
            'total_api_requests': (
                int(data['api_requests'].sum())
                if 'api_requests' in columns else None
            ),
            'entity_breakdown': entity_breakdown,
            'data_types': {
                col: str(dtype)
                for col, dtype in zip(columns, data.dtypes, strict=False)
            }
        }

    def _analyze_columns(self, data: pl.DataFrame) -> dict[str, dict[str, Any]]:
        """Analyze each column for unique values and statistics."""
        column_analysis = {}

        for column in data.columns:
            series = data[column]
            dtype = series.dtype

            analysis = {
                'unique_count': series.n_unique(),
                'null_count': series.null_count(),
                'data_type': str(dtype)
            }

            if dtype in [pl.String, pl.Utf8]:
                value_counts = series.value_counts().limit(10)
                if not value_counts.is_empty():
                    analysis['top_values'] = {
                        row[column]: row['count']
                        for row in value_counts.to_dicts()
                    }
            elif dtype.is_numeric():
                if not data.is_empty():
                    analysis['stats'] = {
                        'min': (
                            float(series.min())
                            if series.min() is not None else None
                        ),
                        'max': (
                            float(series.max())
                            if series.max() is not None else None
                        ),
                        'mean': (
                            float(series.mean())
                            if series.mean() is not None else None
                        ),
                        'median': (
                            float(series.median())
                            if series.median() is not None else None
                        ),
                    }

            column_analysis[column] = analysis

        return column_analysis

    def _filter_successful_requests(self, data: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Filter data to only include records with successful_requests > 0."""
        if data.is_empty():
            return data, {'original_count': 0, 'filtered_count': 0, 'removed_count': 0}

        original_count = len(data)

        # Filter for successful requests only
        if 'successful_requests' in data.columns:
            filtered_data = data.filter(pl.col('successful_requests') > 0)
        else:
            # If column doesn't exist, assume all records are valid
            filtered_data = data

        filtered_count = len(filtered_data)
        removed_count = original_count - filtered_count

        filter_summary = {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count
        }

        return filtered_data, filter_summary

    def print_results(self, analysis: dict[str, Any]) -> None:
        """Print analysis results to console using rich formatting."""
        table_info = analysis['table_info']
        data_summary = analysis['data_summary']
        column_analysis = analysis['column_analysis']

        # Table Structure - compact format
        self.console.print("\n[bold blue]📊 Database Overview[/bold blue]")
        rows = f"{table_info['row_count']:,}"
        cols = str(len(table_info['columns']))

        if 'table_breakdown' in table_info:
            breakdown = table_info['table_breakdown']
            self.console.print(f"  Rows: {rows} ({breakdown['user_spend']:,} user, {breakdown['team_spend']:,} team, {breakdown['tag_spend']:,} tag)")
        else:
            self.console.print(f"  Rows: {rows}")
        self.console.print(f"  Columns: {cols}")

        # Filter Summary
        if 'filter_summary' in analysis:
            filter_info = analysis['filter_summary']
            if filter_info['removed_count'] > 0:
                self.console.print(f"  Filtered: {filter_info['filtered_count']:,} records (removed {filter_info['removed_count']:,} with 0 successful requests)")
            else:
                self.console.print(f"  Filtered: {filter_info['filtered_count']:,} records (no filtering needed)")

        # Data Summary - compact format
        if 'message' in data_summary:
            self.console.print(f"\n[yellow]⚠️  {data_summary['message']}[/yellow]")
        else:
            self.console.print("\n[bold green]📈 Analysis Results[/bold green]")

            # Create compact summary lines
            summary_parts = []
            summary_parts.append(f"Records: {data_summary['total_records_analyzed']:,}")

            if data_summary['date_range']['start']:
                date_range = f"{data_summary['date_range']['start']} to {data_summary['date_range']['end']}"
                summary_parts.append(f"Dates: {date_range}")

            if data_summary['total_spend']:
                summary_parts.append(f"Spend: ${data_summary['total_spend']:.2f}")

            if data_summary['total_tokens']:
                summary_parts.append(f"Tokens: {data_summary['total_tokens']:,}")

            if data_summary['total_api_requests']:
                summary_parts.append(f"API calls: {data_summary['total_api_requests']:,}")

            # Print summary in compact format
            for part in summary_parts:
                self.console.print(f"  {part}")

            # Entity breakdown if available
            if data_summary['entity_breakdown']:
                entity_parts = []
                for entity_type, count in data_summary['entity_breakdown'].items():
                    entity_parts.append(f"{entity_type}: {count:,}")
                self.console.print(f"  Entities: {', '.join(entity_parts)}")

        # Column Analysis - more compact
        self.console.print("\n[bold cyan]🗂️  Column Analysis[/bold cyan]")

        # Use a compact table with lightweight formatting, no width limits to prevent truncation
        from rich.box import SIMPLE
        columns_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        columns_table.add_column("Column", style="cyan", no_wrap=False)
        columns_table.add_column("Type", style="magenta", no_wrap=False)
        columns_table.add_column("Unique", justify="right", style="blue", no_wrap=False)
        columns_table.add_column("Null", justify="right", style="red", no_wrap=False)
        columns_table.add_column("Sample/Stats", style="dim", no_wrap=False)

        for column, stats in column_analysis.items():
            stats_text = ""

            if 'top_values' in stats:
                top_items = list(stats['top_values'].items())  # Show ALL items, no truncation
                stats_text = ", ".join([f"'{k}': {v}" for k, v in top_items])

            elif 'stats' in stats:
                stats_info = stats['stats']
                stats_text = f"min: {stats_info['min']}, max: {stats_info['max']}, mean: {stats_info['mean']:.4f}"

            # Show full column name - no truncation
            columns_table.add_row(
                column,
                stats['data_type'].replace("String", "Str").replace("Datetime", "Date"),  # Shorter types
                f"{stats['unique_count']:,}",
                f"{stats['null_count']:,}",
                stats_text
            )

        self.console.print(columns_table)

        # CBF Transformation Examples
        if analysis.get('cbf_examples'):
            self._print_cbf_examples(analysis['cbf_examples'])

    def _print_cbf_examples(self, cbf_examples: list[dict[str, Any]]) -> None:
        """Print CloudZero CBF transformation examples in spreadsheet format."""
        self.console.print("\n[bold yellow]💰 CBF Transformation Examples[/bold yellow]")

        # Create compact spreadsheet-like table with lightweight formatting, no width limits to prevent truncation
        from rich.box import SIMPLE_HEAVY
        cbf_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE_HEAVY, padding=(0, 1))
        cbf_table.add_column("Date", style="blue", no_wrap=False)
        cbf_table.add_column("Cost", style="green", justify="right", no_wrap=False)
        cbf_table.add_column("Resource ID", style="magenta", no_wrap=False)
        cbf_table.add_column("Tokens", style="yellow", justify="right", no_wrap=False)
        cbf_table.add_column("Entity", style="cyan", no_wrap=False)
        cbf_table.add_column("Model", style="white", no_wrap=False)
        cbf_table.add_column("Provider", style="dim", no_wrap=False)

        for cbf_record in cbf_examples:
            # Extract key information using current CBF field names
            date_full = str(cbf_record.get('time/usage_start', 'N/A'))
            # Extract just the date part but show it completely
            date = date_full.split('T')[0] if 'T' in date_full else date_full
            cost = f"${cbf_record.get('cost/cost', 0):.6f}"
            resource_id = str(cbf_record.get('resource/id', 'N/A'))  # Show full resource_id

            tokens = f"{cbf_record.get('usage/amount', 0):,}"

            # Extract resource tags (dimensions are now stored as resource/tag: fields)
            entity_type = cbf_record.get('resource/tag:entity_type', 'N/A')
            entity_id = cbf_record.get('resource/tag:entity_id', '')
            entity = f"{entity_type}\n{entity_id}" if entity_id else entity_type

            model = cbf_record.get('resource/tag:model', 'N/A')  # Show full model name
            provider = cbf_record.get('resource/tag:provider', 'N/A')

            cbf_table.add_row(
                date,
                cost,
                resource_id,
                tokens,
                entity,
                model,
                provider
            )

        self.console.print(cbf_table)

        # Summary message - more compact
        total_records = len(cbf_examples)
        self.console.print(f"\n[dim]💡 {total_records} sample CBF record(s) • Use --csv or --cz-api-key to export all data[/dim]")

    def czrn_analysis(self, limit: int | None = 10000, force_refresh: bool = False) -> None:
        """Perform CZRN-focused analysis showing resource ID generation."""
        if limit is None:
            self.console.print("\n[bold yellow]🔗 CZRN Analysis - Processing all records[/bold yellow]")
        else:
            self.console.print(f"\n[bold yellow]🔗 CZRN Analysis - Processing {limit} records[/bold yellow]")

        # Get sample data
        if isinstance(self.database, CachedLiteLLMDatabase):
            raw_data = self.database.get_usage_data(limit=limit, force_refresh=force_refresh)
        else:
            raw_data = self.database.get_usage_data(limit=limit)

        # Filter data and show filtering summary
        data, filter_summary = self._filter_successful_requests(raw_data)

        # Show filter summary
        if filter_summary['removed_count'] > 0:
            self.console.print(f"[dim]Filtered {filter_summary['filtered_count']:,} records (removed {filter_summary['removed_count']:,} with 0 successful requests)[/dim]")
        else:
            self.console.print(f"[dim]Processing {filter_summary['filtered_count']:,} records (no filtering needed)[/dim]")

        if data.is_empty():
            self.console.print("[yellow]No data available for CZRN analysis after filtering[/yellow]")
            return

        # Generate CZRNs for the sample data
        czrn_generator = CZRNGenerator()
        czrn_results = []

        for row in data.to_dicts():
            try:
                czrn = czrn_generator.create_from_litellm_data(row)
                czrn_results.append({
                    'czrn': czrn,
                    'source_data': row
                })
            except Exception as e:
                czrn_results.append({
                    'czrn': f"ERROR: {str(e)}",
                    'source_data': row
                })

        # Show CZRN component breakdown first
        self._print_czrn_component_analysis(czrn_results)

        # Display results as a simple list
        self._print_czrn_list(czrn_results)

        # Show detailed error information if any errors occurred
        self._print_czrn_errors(czrn_results)

    def _print_czrn_list(self, czrn_results: list[dict[str, Any]]) -> None:
        """Print generated CZRNs as a deduplicated table with aligned components."""
        self.console.print("\n[bold green]📝 Generated CZRNs (Deduplicated)[/bold green]")

        # Group results by CZRN for deduplication
        czrn_groups = {}
        for result in czrn_results:
            czrn = result['czrn']
            if czrn not in czrn_groups:
                czrn_groups[czrn] = []
            czrn_groups[czrn].append(result)

        # Separate successful CZRNs from errors
        successful_czrns = {}
        error_czrns = {}

        for czrn, group in czrn_groups.items():
            if czrn.startswith('ERROR:'):
                error_czrns[czrn] = group
            else:
                successful_czrns[czrn] = group

        # Display successful CZRNs in a formatted table
        if successful_czrns:
            from rich.box import SIMPLE
            from rich.table import Table

            # Create table with no width constraints to show all data
            czrn_table = Table(
                show_header=True,
                header_style="bold cyan",
                box=SIMPLE,
                padding=(0, 1),
                expand=False,
                min_width=None,
                width=None,
            )
            czrn_table.add_column("#", style="green", justify="right", no_wrap=True)
            czrn_table.add_column("Service", style="blue", no_wrap=True)
            czrn_table.add_column("Provider", style="yellow", no_wrap=True)
            czrn_table.add_column("Region", style="magenta", no_wrap=True)
            czrn_table.add_column("Owner Account", style="cyan", no_wrap=True)
            czrn_table.add_column("Resource", style="green", no_wrap=True)
            czrn_table.add_column("Local ID", style="white", no_wrap=True)
            czrn_table.add_column("Records", style="dim", justify="right", no_wrap=True)

            czrn_generator = CZRNGenerator()

            for i, (czrn, group) in enumerate(sorted(successful_czrns.items()), 1):
                try:
                    # Parse CZRN components
                    service_type, provider, region, owner_account_id, resource_type, cloud_local_id = czrn_generator.extract_components(czrn)

                    # Display full components without truncation
                    czrn_table.add_row(
                        str(i),
                        service_type,
                        provider,
                        region,
                        owner_account_id,
                        resource_type,
                        cloud_local_id,
                        str(len(group))
                    )
                except Exception:
                    # Fallback for malformed CZRNs
                    czrn_table.add_row(
                        str(i),
                        "[red]MALFORMED[/red]",
                        "",
                        "",
                        "",
                        "",
                        czrn,
                        str(len(group))
                    )

            # Print table with wider console to avoid truncation
            self.console.print(czrn_table)
            # Create a temporary wider console if needed
            from rich.console import Console
            temp_console = Console(width=200, force_terminal=True)
            temp_console.print(czrn_table)

        # Display error CZRNs separately
        if error_czrns:
            self.console.print("\n[bold red]❌ Error CZRNs[/bold red]")
            for i, (czrn, group) in enumerate(error_czrns.items(), 1):
                clean_error = czrn.replace('ERROR: ', '')
                self.console.print(f"[red]{i:3d}. {clean_error}[/red] [dim]({len(group)} records)[/dim]")

        # Summary
        total_records = len(czrn_results)
        errors = len(error_czrns)
        successful_unique = len(successful_czrns)

        self.console.print(f"\n[dim]💡 {successful_unique:,} unique CZRNs from {total_records:,} total records[/dim]")
        if errors > 0:
            self.console.print(f"[dim]❌ {errors:,} error types affecting records[/dim]")

        # Show source records for unknown-account CZRNs
        self._show_unknown_account_details(czrn_groups)

    def _show_unknown_account_details(self, czrn_groups: dict[str, list[dict[str, Any]]]) -> None:
        """Show source records for CZRNs with unknown-account owner account IDs."""
        unknown_account_czrns = {}

        for czrn, group in czrn_groups.items():
            if not czrn.startswith('ERROR:') and 'unknown-account' in czrn:
                unknown_account_czrns[czrn] = group

        if not unknown_account_czrns:
            return

        self.console.print("\n[bold red]⚠️  Unknown Account Details[/bold red]")
        self.console.print(f"[yellow]Found {len(unknown_account_czrns)} CZRN(s) with unknown-account. Showing source records:[/yellow]\n")

        for czrn, group in unknown_account_czrns.items():
            self.console.print(f"[bold white]CZRN:[/bold white] [red]{czrn}[/red]")
            self.console.print(f"[dim]Affected records ({len(group)} total, showing up to 5):[/dim]")

            # Show up to 5 source records that contribute to this CZRN with ALL fields
            from rich.box import SIMPLE
            table = Table(show_header=True, header_style="bold yellow", box=SIMPLE, padding=(0, 1))
            table.add_column("ID", style="dim", no_wrap=False)
            table.add_column("Date", style="green", no_wrap=False)
            table.add_column("Entity Type", style="blue", no_wrap=False)
            table.add_column("Entity ID", style="cyan", no_wrap=False)
            table.add_column("API Key", style="red", no_wrap=False)
            table.add_column("Model", style="magenta", no_wrap=False)
            table.add_column("Model Group", style="purple", no_wrap=False)
            table.add_column("Provider", style="yellow", no_wrap=False)
            table.add_column("Prompt Tokens", style="blue", justify="right", no_wrap=False)
            table.add_column("Completion Tokens", style="blue", justify="right", no_wrap=False)
            table.add_column("Spend", style="green", justify="right", no_wrap=False)
            table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
            table.add_column("Success", style="green", justify="right", no_wrap=False)
            table.add_column("Failed", style="red", justify="right", no_wrap=False)
            table.add_column("Cache Create", style="orange1", justify="right", no_wrap=False)
            table.add_column("Cache Read", style="orange3", justify="right", no_wrap=False)
            table.add_column("Created At", style="dim", no_wrap=False)
            table.add_column("Updated At", style="dim", no_wrap=False)

            for record in group[:5]:  # Limit to 5 records
                source = record['source_data']

                # Show ALL fields from the source record
                record_id = str(source.get('id', 'N/A'))
                date = str(source.get('date', 'N/A'))
                entity_type = str(source.get('entity_type', 'N/A'))
                entity_id = str(source.get('entity_id', 'N/A'))

                api_key = str(source.get('api_key', 'N/A'))
                if len(api_key) > 20:
                    api_key = api_key[:8] + "..." + api_key[-4:]  # Show prefix and suffix

                model = str(source.get('model', 'N/A'))
                model_group = str(source.get('model_group', 'N/A'))
                provider = str(source.get('custom_llm_provider', 'N/A'))

                prompt_tokens = str(source.get('prompt_tokens', 0))
                completion_tokens = str(source.get('completion_tokens', 0))
                spend = f"${source.get('spend', 0):.6f}"
                api_requests = str(source.get('api_requests', 0))
                successful_requests = str(source.get('successful_requests', 0))
                failed_requests = str(source.get('failed_requests', 0))
                cache_creation_tokens = str(source.get('cache_creation_input_tokens', 0))
                cache_read_tokens = str(source.get('cache_read_input_tokens', 0))

                created_at = str(source.get('created_at', 'N/A'))
                if 'T' in created_at:
                    created_at = created_at.split('T')[0]  # Show just date part for brevity

                updated_at = str(source.get('updated_at', 'N/A'))
                if 'T' in updated_at:
                    updated_at = updated_at.split('T')[0]  # Show just date part for brevity

                table.add_row(
                    record_id, date, entity_type, entity_id, api_key, model, model_group,
                    provider, prompt_tokens, completion_tokens, spend, api_requests,
                    successful_requests, failed_requests, cache_creation_tokens,
                    cache_read_tokens, created_at, updated_at
                )

            self.console.print(table)

            if len(group) > 5:
                self.console.print(f"[dim]... and {len(group) - 5} more records[/dim]")

            self.console.print()  # Add spacing between CZRNs

    def _print_czrn_component_analysis(self, czrn_results: list[dict[str, Any]]) -> None:
        """Print analysis of CZRN components."""
        self.console.print("\n[bold yellow]🧩 CZRN Component Analysis[/bold yellow]")

        # Extract components from successful CZRNs
        czrn_generator = CZRNGenerator()
        component_stats = {
            'service_type': {},
            'provider': {},
            'region': {},
            'owner_account_id': {},
            'resource_type': {},
            'cloud_local_id_patterns': {},
            'entity_types': {},
            'models': {}
        }

        valid_czrns = [r for r in czrn_results if not r['czrn'].startswith('ERROR:')]

        for result in valid_czrns:
            try:
                components = czrn_generator.extract_components(result['czrn'])
                service_type, provider, region, owner_account_id, resource_type, cloud_local_id = components

                # Count component frequencies
                component_stats['service_type'][service_type] = component_stats['service_type'].get(service_type, 0) + 1
                component_stats['provider'][provider] = component_stats['provider'].get(provider, 0) + 1
                component_stats['region'][region] = component_stats['region'].get(region, 0) + 1
                component_stats['owner_account_id'][owner_account_id] = component_stats['owner_account_id'].get(owner_account_id, 0) + 1
                component_stats['resource_type'][resource_type] = component_stats['resource_type'].get(resource_type, 0) + 1

                # Analyze cloud_local_id patterns (now just the model)
                component_stats['cloud_local_id_patterns'][cloud_local_id] = component_stats['cloud_local_id_patterns'].get(cloud_local_id, 0) + 1

                # Extract model from cloud_local_id (now just the model itself)
                component_stats['models'][cloud_local_id] = component_stats['models'].get(cloud_local_id, 0) + 1

                # Get entity types from source data instead of cloud_local_id
                source_data = result['source_data']
                entity_type = source_data.get('entity_type', 'unknown')
                component_stats['entity_types'][entity_type] = component_stats['entity_types'].get(entity_type, 0) + 1

            except Exception:
                continue

        # Create component breakdown table - lightweight formatting, no width limits
        from rich.box import SIMPLE
        comp_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        comp_table.add_column("Component", style="bold blue", no_wrap=False)
        comp_table.add_column("Values", style="white", no_wrap=False)
        comp_table.add_column("Count", style="green", justify="right", no_wrap=False)

        # Ensure ALL components are shown, even if empty
        for component_name, values in component_stats.items():
            if values:
                # Show all values for this component, sorted by frequency
                for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True):
                    comp_table.add_row(component_name.replace('_', ' ').title(), value, str(count))
                    component_name = ""  # Only show component name for first row
            else:
                # Show empty components too
                comp_table.add_row(component_name.replace('_', ' ').title(), "[dim]No values found[/dim]", "0")

        self.console.print(comp_table)

        # Summary statistics - compact format
        total_records = len(czrn_results)
        successful_czrns = len(valid_czrns)
        error_count = total_records - successful_czrns

        self.console.print(f"\n[green]✓ {successful_czrns}/{total_records} CZRNs generated successfully[/green]")
        if error_count > 0:
            self.console.print(f"[red]✗ {error_count} generation errors[/red]")

        self.console.print("\n[dim]💡 CZRNs follow format: czrn:service-type:provider:region:owner-account-id:resource-type:cloud-local-id[/dim]")
        self.console.print("[dim]🔍 Use generated CZRNs as resource_id values in CloudZero CBF records[/dim]")

    def _print_czrn_errors(self, czrn_results: list[dict[str, Any]]) -> None:
        """Print detailed error information for failed CZRN generations."""
        # Collect error results
        error_results = [result for result in czrn_results if result['czrn'].startswith('ERROR:')]

        if not error_results:
            return

        self.console.print("\n[bold red]❌ CZRN Generation Errors[/bold red]")
        self.console.print(f"[yellow]Found {len(error_results)} record(s) with generation errors:[/yellow]\n")

        # Group errors by error message
        error_groups = {}
        for result in error_results:
            error_msg = result['czrn']  # Contains "ERROR: <message>"
            if error_msg not in error_groups:
                error_groups[error_msg] = []
            error_groups[error_msg].append(result)

        for i, (error_msg, group) in enumerate(error_groups.items(), 1):
            # Clean up the error message for display
            clean_error = error_msg.replace('ERROR: ', '')
            self.console.print(f"[bold red]Error {i}:[/bold red] [white]{clean_error}[/white]")
            self.console.print(f"[dim]Affects {len(group)} record(s). Sample problematic records:[/dim]")

            # Show up to 3 sample records per error in detailed format
            for j, record in enumerate(group[:3], 1):
                source = record['source_data']

                self.console.print(f"\n[bold]Record {j}:[/bold]")

                # Show all fields from the database record
                for field_name, field_value in source.items():
                    # Format field name and value for display
                    formatted_name = field_name.replace('_', ' ').title()

                    # Handle different field types
                    if field_value is None:
                        formatted_value = "[dim red]NULL[/dim red]"
                    elif field_value == "":
                        formatted_value = "[dim red]EMPTY[/dim red]"
                    elif isinstance(field_value, str):
                        # Truncate very long strings but keep API keys readable
                        if field_name == 'api_key' and len(str(field_value)) > 20:
                            formatted_value = f"[red]{str(field_value)[:8]}...{str(field_value)[-4:]}[/red]"
                        elif len(str(field_value)) > 100:
                            formatted_value = f"[white]{str(field_value)[:97]}...[/white]"
                        else:
                            formatted_value = f"[white]{str(field_value)}[/white]"
                    elif isinstance(field_value, (int, float)):
                        formatted_value = f"[cyan]{field_value}[/cyan]"
                    else:
                        formatted_value = f"[white]{str(field_value)}[/white]"

                    # Color-code problematic fields
                    if field_value in [None, "", 0] and field_name in ['model', 'entity_id', 'entity_type', 'custom_llm_provider']:
                        formatted_name = f"[bold red]{formatted_name}[/bold red]"
                    else:
                        formatted_name = f"[bold blue]{formatted_name}:[/bold blue]"

                    self.console.print(f"  {formatted_name:25} {formatted_value}")

            if len(group) > 3:
                self.console.print(f"\n[dim]... and {len(group) - 3} more record(s) with the same error[/dim]")

            self.console.print()  # Add spacing between error groups

    def spend_analysis(self, limit: int | None = 10000, force_refresh: bool = False) -> None:
        """Perform comprehensive spend analysis based on teams and users."""
        if limit is None:
            self.console.print("\n[bold blue]💰 Spend Analysis - Processing all records[/bold blue]")
        else:
            self.console.print(f"\n[bold blue]💰 Spend Analysis - Processing {limit} records[/bold blue]")

        # Get data from cache/database
        if isinstance(self.database, CachedLiteLLMDatabase):
            raw_data = self.database.get_usage_data(limit=limit, force_refresh=force_refresh)
        else:
            raw_data = self.database.get_usage_data(limit=limit)

        # Filter data and show filtering summary
        data, filter_summary = self._filter_successful_requests(raw_data)

        # Show filter summary
        if filter_summary['removed_count'] > 0:
            self.console.print(f"[dim]Filtered {filter_summary['filtered_count']:,} records (removed {filter_summary['removed_count']:,} with 0 successful requests)[/dim]")
        else:
            self.console.print(f"[dim]Processing {filter_summary['filtered_count']:,} records (no filtering needed)[/dim]")

        if data.is_empty():
            self.console.print("[yellow]No data available for spend analysis after filtering[/yellow]")
            return

        # Perform spend analysis
        self._analyze_spend_by_entity(data)
        self._analyze_spend_by_model(data)
        self._analyze_spend_by_provider(data)
        self._analyze_spend_trends(data)

    def _analyze_spend_by_entity(self, data: pl.DataFrame) -> None:
        """Analyze spending breakdown by entity type (teams vs users)."""
        self.console.print("\n[bold yellow]👥 Entity Spend Analysis[/bold yellow]")

        # Group by entity type and calculate totals
        entity_summary = data.group_by('entity_type').agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('entity_id').n_unique().alias('unique_entities'),
            pl.col('api_requests').sum().alias('total_requests'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.len().alias('record_count')
        ]).sort('total_spend', descending=True)

        # Display entity type summary
        from rich.box import SIMPLE
        from rich.table import Table

        summary_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        summary_table.add_column("Entity Type", style="bold green", no_wrap=False)
        summary_table.add_column("Total Spend", style="green", justify="right", no_wrap=False)
        summary_table.add_column("Entities", style="blue", justify="right", no_wrap=False)
        summary_table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
        summary_table.add_column("Total Tokens", style="yellow", justify="right", no_wrap=False)
        summary_table.add_column("Records", style="dim", justify="right", no_wrap=False)

        total_spend = 0
        for row in entity_summary.to_dicts():
            entity_type = row['entity_type']
            spend = row['total_spend']
            total_spend += spend
            unique_entities = row['unique_entities']
            total_requests = row['total_requests']
            total_tokens = row['total_prompt_tokens'] + row['total_completion_tokens']
            records = row['record_count']

            summary_table.add_row(
                entity_type.title(),
                f"${spend:.2f}",
                f"{unique_entities:,}",
                f"{total_requests:,}",
                f"{total_tokens:,}",
                f"{records:,}"
            )

        self.console.print(summary_table)
        self.console.print(f"[dim]💡 Total spend across all entities: ${total_spend:.2f}[/dim]")

        # Show top spenders within each entity type
        self._show_top_spenders_by_entity_type(data, 'team', 5)
        self._show_top_spenders_by_entity_type(data, 'user', 5)

    def _show_top_spenders_by_entity_type(self, data: pl.DataFrame, entity_type: str, top_n: int) -> None:
        """Show top spenders for a specific entity type."""
        # Filter by entity type
        entity_data = data.filter(pl.col('entity_type') == entity_type)

        if entity_data.is_empty():
            return

        # Group by entity_id and calculate spend
        top_spenders = entity_data.group_by('entity_id').agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('api_requests').sum().alias('total_requests'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.col('model').n_unique().alias('unique_models'),
            pl.len().alias('record_count')
        ]).sort('total_spend', descending=True).head(top_n)

        self.console.print(f"\n[bold cyan]🏆 Top {top_n} {entity_type.title()} Spenders[/bold cyan]")

        from rich.box import SIMPLE
        from rich.table import Table

        spenders_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        spenders_table.add_column(f"{entity_type.title()} ID", style="bold blue", no_wrap=False)
        spenders_table.add_column("Total Spend", style="green", justify="right", no_wrap=False)
        spenders_table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
        spenders_table.add_column("Total Tokens", style="yellow", justify="right", no_wrap=False)
        spenders_table.add_column("Models Used", style="magenta", justify="right", no_wrap=False)
        spenders_table.add_column("Records", style="dim", justify="right", no_wrap=False)

        for row in top_spenders.to_dicts():
            entity_id = row['entity_id']
            spend = row['total_spend']
            requests = row['total_requests']
            total_tokens = row['total_prompt_tokens'] + row['total_completion_tokens']
            unique_models = row['unique_models']
            records = row['record_count']

            spenders_table.add_row(
                entity_id,
                f"${spend:.2f}",
                f"{requests:,}",
                f"{total_tokens:,}",
                f"{unique_models}",
                f"{records:,}"
            )

        self.console.print(spenders_table)

    def _analyze_spend_by_model(self, data: pl.DataFrame) -> None:
        """Analyze spending breakdown by model."""
        self.console.print("\n[bold yellow]🤖 Model Spend Analysis[/bold yellow]")

        # Group by model and calculate totals
        model_summary = data.group_by('model').agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('entity_id').n_unique().alias('unique_users'),
            pl.col('api_requests').sum().alias('total_requests'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.len().alias('record_count')
        ]).sort('total_spend', descending=True).head(10)  # Top 10 models

        from rich.box import SIMPLE
        from rich.table import Table

        model_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        model_table.add_column("Model", style="bold magenta", no_wrap=False)
        model_table.add_column("Total Spend", style="green", justify="right", no_wrap=False)
        model_table.add_column("Users", style="blue", justify="right", no_wrap=False)
        model_table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
        model_table.add_column("Total Tokens", style="yellow", justify="right", no_wrap=False)
        model_table.add_column("Avg Cost/Token", style="red", justify="right", no_wrap=False)

        for row in model_summary.to_dicts():
            model = row['model']
            spend = row['total_spend']
            users = row['unique_users']
            requests = row['total_requests']
            total_tokens = row['total_prompt_tokens'] + row['total_completion_tokens']
            avg_cost_per_token = spend / total_tokens if total_tokens > 0 else 0

            model_table.add_row(
                model,
                f"${spend:.2f}",
                f"{users:,}",
                f"{requests:,}",
                f"{total_tokens:,}",
                f"${avg_cost_per_token:.6f}"
            )

        self.console.print(model_table)

    def _analyze_spend_by_provider(self, data: pl.DataFrame) -> None:
        """Analyze spending breakdown by provider."""
        self.console.print("\n[bold yellow]🏢 Provider Spend Analysis[/bold yellow]")

        # Group by provider and calculate totals
        provider_summary = data.group_by('custom_llm_provider').agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('entity_id').n_unique().alias('unique_users'),
            pl.col('model').n_unique().alias('unique_models'),
            pl.col('api_requests').sum().alias('total_requests'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.len().alias('record_count')
        ]).sort('total_spend', descending=True)

        from rich.box import SIMPLE
        from rich.table import Table

        provider_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        provider_table.add_column("Provider", style="bold yellow", no_wrap=False)
        provider_table.add_column("Total Spend", style="green", justify="right", no_wrap=False)
        provider_table.add_column("Users", style="blue", justify="right", no_wrap=False)
        provider_table.add_column("Models", style="magenta", justify="right", no_wrap=False)
        provider_table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
        provider_table.add_column("Total Tokens", style="yellow", justify="right", no_wrap=False)

        for row in provider_summary.to_dicts():
            provider = row['custom_llm_provider'] or 'Unknown'
            spend = row['total_spend']
            users = row['unique_users']
            models = row['unique_models']
            requests = row['total_requests']
            total_tokens = row['total_prompt_tokens'] + row['total_completion_tokens']

            provider_table.add_row(
                provider,
                f"${spend:.2f}",
                f"{users:,}",
                f"{models:,}",
                f"{requests:,}",
                f"{total_tokens:,}"
            )

        self.console.print(provider_table)

    def _analyze_spend_trends(self, data: pl.DataFrame) -> None:
        """Analyze spending trends over time."""
        self.console.print("\n[bold yellow]📈 Spend Trends Analysis[/bold yellow]")

        # Check if we have date information
        if 'date' not in data.columns:
            self.console.print("[dim]No date information available for trend analysis[/dim]")
            return

        # Group by date and calculate daily totals
        daily_trends = data.group_by('date').agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('entity_id').n_unique().alias('unique_users'),
            pl.col('api_requests').sum().alias('total_requests'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.len().alias('record_count')
        ]).sort('date')

        if daily_trends.is_empty():
            self.console.print("[dim]No trend data available[/dim]")
            return

        # Show summary statistics
        total_days = len(daily_trends)
        avg_daily_spend = daily_trends['total_spend'].mean()
        max_daily_spend = daily_trends['total_spend'].max()
        min_daily_spend = daily_trends['total_spend'].min()

        self.console.print("[green]📊 Trend Summary[/green]")
        self.console.print(f"  Days analyzed: {total_days}")
        self.console.print(f"  Average daily spend: ${avg_daily_spend:.2f}")
        self.console.print(f"  Highest daily spend: ${max_daily_spend:.2f}")
        self.console.print(f"  Lowest daily spend: ${min_daily_spend:.2f}")

        # Show recent days (last 7 days)
        recent_days = daily_trends.tail(7)

        from rich.box import SIMPLE
        from rich.table import Table

        trend_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        trend_table.add_column("Date", style="bold blue", no_wrap=False)
        trend_table.add_column("Daily Spend", style="green", justify="right", no_wrap=False)
        trend_table.add_column("Users", style="blue", justify="right", no_wrap=False)
        trend_table.add_column("API Requests", style="cyan", justify="right", no_wrap=False)
        trend_table.add_column("Total Tokens", style="yellow", justify="right", no_wrap=False)

        for row in recent_days.to_dicts():
            date = row['date']
            spend = row['total_spend']
            users = row['unique_users']
            requests = row['total_requests']
            total_tokens = row['total_prompt_tokens'] + row['total_completion_tokens']

            trend_table.add_row(
                str(date),
                f"${spend:.2f}",
                f"{users:,}",
                f"{requests:,}",
                f"{total_tokens:,}"
            )

        self.console.print(f"\n[bold cyan]📅 Recent Activity (Last {len(recent_days)} Days)[/bold cyan]")
        self.console.print(trend_table)

