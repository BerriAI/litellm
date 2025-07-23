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
# CHANGELOG: 2025-01-19 - Updated CBF transformation for LiteLLM_SpendLogs with hourly aggregation and team_id focus (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars for data transformation (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial CBF transformation module (erik.peterson)

"""Transform LiteLLM data to CloudZero AnyCost CBF format."""

import json
from datetime import datetime
from typing import Any, List, Optional

import polars as pl

from ...types.integrations.cloudzero import CBFRecord
from .cz_resource_names import CZRNGenerator


class CBFTransformer:
    """Transform LiteLLM usage data to CloudZero Billing Format (CBF)."""

    def __init__(self):
        """Initialize transformer with CZRN generator."""
        self.czrn_generator = CZRNGenerator()

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform LiteLLM SpendLogs data to hourly aggregated CBF format."""
        if data.is_empty():
            return pl.DataFrame()

        # Filter out records with zero spend or invalid team_id
        original_count = len(data)
        filtered_data = data.filter(
            (pl.col('spend') > 0) & 
            (pl.col('team_id').is_not_null()) &
            (pl.col('team_id') != "")
        )
        filtered_count = len(filtered_data)
        zero_spend_dropped = original_count - filtered_count

        if filtered_data.is_empty():
            from rich.console import Console
            console = Console()
            console.print(f"[yellow]⚠️  Dropped all {original_count:,} records due to zero spend or missing team_id[/yellow]")
            return pl.DataFrame()

        # Aggregate data to hourly level
        hourly_aggregated = self._aggregate_to_hourly(filtered_data)
        
        # Transform aggregated data to CBF format
        cbf_data = []
        czrn_dropped_count = 0
        
        for row in hourly_aggregated.iter_rows(named=True):
            try:
                cbf_record = self._create_cbf_record(row)
                cbf_data.append(cbf_record)
            except Exception:
                # Skip records that fail CZRN generation
                czrn_dropped_count += 1
                continue

        # Print summary of transformations
        from rich.console import Console
        console = Console()

        if zero_spend_dropped > 0:
            console.print(f"[yellow]⚠️  Dropped {zero_spend_dropped:,} of {original_count:,} records with zero spend or missing team_id[/yellow]")

        if czrn_dropped_count > 0:
            console.print(f"[yellow]⚠️  Dropped {czrn_dropped_count:,} of {len(hourly_aggregated):,} aggregated records due to invalid CZRNs[/yellow]")

        if len(cbf_data) > 0:
            console.print(f"[green]✓ Successfully transformed {len(cbf_data):,} hourly aggregated records[/green]")

        return pl.DataFrame(cbf_data)

    def _aggregate_to_hourly(self, data: pl.DataFrame) -> pl.DataFrame:
        """Aggregate spend logs to hourly level by team_id, key_name, model, and tags."""
        
        # Extract hour from startTime and parse tags
        data_with_hour = data.with_columns([
            pl.col('startTime').dt.truncate('1h').alias('usage_hour'),
            pl.col('request_tags').map_elements(self._parse_tags, return_dtype=pl.List(pl.String)).alias('parsed_tags')
        ])
        
        # Flatten tags for grouping (each tag becomes a separate record)
        data_exploded = data_with_hour.explode('parsed_tags').with_columns([
            pl.col('parsed_tags').alias('tag')
        ])
        
        # Also include records without tags (tag = None)
        data_no_tags = data_with_hour.filter(
            pl.col('parsed_tags').list.len() == 0
        ).with_columns([
            pl.lit(None).alias('tag')
        ])
        
        # Combine data with tags and without tags
        all_data = pl.concat([data_exploded, data_no_tags])
        
        # Group by hour, team_id, key_name, model, provider, and tag
        aggregated = all_data.group_by([
            'usage_hour',
            'team_id', 
            'key_name',
            'model',
            'model_group',
            'custom_llm_provider',
            'tag'
        ]).agg([
            pl.col('spend').sum().alias('total_spend'),
            pl.col('total_tokens').sum().alias('total_tokens'),
            pl.col('prompt_tokens').sum().alias('total_prompt_tokens'),
            pl.col('completion_tokens').sum().alias('total_completion_tokens'),
            pl.col('request_id').count().alias('request_count'),
            pl.col('api_key').first().alias('api_key_sample'),  # Keep one for reference
            pl.col('status').filter(pl.col('status') == 'success').count().alias('successful_requests'),
            pl.col('status').filter(pl.col('status') != 'success').count().alias('failed_requests')
        ])
        
        return aggregated

    def _parse_tags(self, tags_json: str) -> List[str]:
        """Parse request_tags JSON field to extract list of tags."""
        if not tags_json or tags_json in ['[]', '{}', 'null']:
            return []
        
        try:
            if isinstance(tags_json, str):
                tags = json.loads(tags_json)
            else:
                tags = tags_json
                
            if isinstance(tags, list):
                return [str(tag) for tag in tags if tag]
            elif isinstance(tags, dict):
                # If tags is a dict, extract values
                return [str(value) for value in tags.values() if value]
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []

    def _create_cbf_record(self, row: dict[str, Any]) -> CBFRecord:
        """Create a single CBF record from aggregated hourly spend data."""

        # Use the aggregated hour as usage time
        usage_time = self._parse_datetime(row.get('usage_hour'))
        
        # Use team_id as the primary entity_id
        entity_id = str(row.get('team_id', ''))
        key_name = str(row.get('key_name', ''))
        model = str(row.get('model', ''))
        model_group = str(row.get('model_group', ''))
        provider = str(row.get('custom_llm_provider', ''))
        tag = row.get('tag')
        
        # Calculate aggregated metrics
        total_spend = float(row.get('total_spend', 0.0))
        total_tokens = int(row.get('total_tokens', 0))
        total_prompt_tokens = int(row.get('total_prompt_tokens', 0))
        total_completion_tokens = int(row.get('total_completion_tokens', 0))
        request_count = int(row.get('request_count', 0))
        successful_requests = int(row.get('successful_requests', 0))
        failed_requests = int(row.get('failed_requests', 0))

        # Create CloudZero Resource Name (CZRN) as resource_id
        # Create a mock row for CZRN generation with team_id as entity_id
        czrn_row = {
            'entity_id': entity_id,
            'entity_type': 'team',
            'model': model,
            'custom_llm_provider': provider,
            'api_key': row.get('api_key_sample', '')
        }
        resource_id = self.czrn_generator.create_from_litellm_data(czrn_row)

        # Build dimensions for CloudZero tracking
        dimensions = {
            'entity_type': 'team',
            'entity_id': entity_id,
            'key_name': key_name,
            'model': model,
            'model_group': model_group,
            'provider': provider,
            'request_count': str(request_count),
            'successful_requests': str(successful_requests),
            'failed_requests': str(failed_requests),
        }
        
        # Add tag if present
        if tag and tag != 'null':
            dimensions['tag'] = str(tag)

        # Extract CZRN components to populate corresponding CBF columns
        czrn_components = self.czrn_generator.extract_components(resource_id)
        service_type, provider_czrn, region, owner_account_id, resource_type, cloud_local_id = czrn_components

        # CloudZero CBF format with proper column names
        cbf_record = {
            # Required CBF fields
            'time/usage_start': usage_time.isoformat() if usage_time else None,  # Required: ISO-formatted UTC datetime
            'cost/cost': total_spend,  # Required: billed cost
            'resource/id': resource_id,  # Required when resource tags are present

            # Usage metrics for token consumption
            'usage/amount': total_tokens,  # Numeric value of tokens consumed
            'usage/units': 'tokens',  # Description of token units

            # CBF fields that correspond to CZRN components
            'resource/service': service_type,  # Maps to CZRN service-type (litellm)
            'resource/account': owner_account_id,  # Maps to CZRN owner-account-id (entity_id)
            'resource/region': region,  # Maps to CZRN region (cross-region)
            'resource/usage_family': resource_type,  # Maps to CZRN resource-type (llm-usage)

            # Line item details
            'lineitem/type': 'Usage',  # Standard usage line item
        }

        # Add CZRN components that don't have direct CBF column mappings as resource tags
        cbf_record['resource/tag:provider'] = provider_czrn  # CZRN provider component
        cbf_record['resource/tag:model'] = cloud_local_id  # CZRN cloud-local-id component (model)

        # Add resource tags for all dimensions (using resource/tag:<key> format)
        for key, value in dimensions.items():
            if value and value != 'N/A':  # Only add non-empty tags
                cbf_record[f'resource/tag:{key}'] = str(value)

        # Add token breakdown as resource tags for analysis
        if total_prompt_tokens > 0:
            cbf_record['resource/tag:prompt_tokens'] = str(total_prompt_tokens)
        if total_completion_tokens > 0:
            cbf_record['resource/tag:completion_tokens'] = str(total_completion_tokens)
        if total_tokens > 0:
            cbf_record['resource/tag:total_tokens'] = str(total_tokens)

        return CBFRecord(cbf_record)

    def _parse_datetime(self, datetime_obj) -> Optional[datetime]:
        """Parse datetime object to ensure proper format."""
        if datetime_obj is None:
            return None

        if isinstance(datetime_obj, datetime):
            return datetime_obj

        if isinstance(datetime_obj, str):
            try:
                # Try to parse ISO format
                return pl.Series([datetime_obj]).str.to_datetime().item()
            except Exception:
                return None

        return None


