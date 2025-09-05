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
# CHANGELOG: 2025-01-19 - Updated CBF transformation for daily spend tables and proper CloudZero mapping (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars for data transformation (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial CBF transformation module (erik.peterson)

"""Transform LiteLLM data to CloudZero AnyCost CBF format."""

from datetime import datetime
from typing import Any, Optional

import polars as pl

from ...types.integrations.cloudzero import CBFRecord
from .cz_resource_names import CZRNGenerator


class CBFTransformer:
    """Transform LiteLLM usage data to CloudZero Billing Format (CBF)."""

    def __init__(self):
        """Initialize transformer with CZRN generator."""
        self.czrn_generator = CZRNGenerator()

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """Transform LiteLLM data to CBF format, dropping records with zero successful_requests or invalid CZRNs."""
        if data.is_empty():
            return pl.DataFrame()

        # Filter out records with zero successful_requests first
        original_count = len(data)
        if 'successful_requests' in data.columns:
            filtered_data = data.filter(pl.col('successful_requests') > 0)
            zero_requests_dropped = original_count - len(filtered_data)
        else:
            filtered_data = data
            zero_requests_dropped = 0

        cbf_data = []
        czrn_dropped_count = 0
        filtered_count = len(filtered_data)

        for row in filtered_data.iter_rows(named=True):
            try:
                cbf_record = self._create_cbf_record(row)
                # Only include the record if CZRN generation was successful
                cbf_data.append(cbf_record)
            except Exception:
                # Skip records that fail CZRN generation
                czrn_dropped_count += 1
                continue

        # Print summary of dropped records if any
        from rich.console import Console
        console = Console()

        if zero_requests_dropped > 0:
            console.print(f"[yellow]⚠️  Dropped {zero_requests_dropped:,} of {original_count:,} records with zero successful_requests[/yellow]")

        if czrn_dropped_count > 0:
            console.print(f"[yellow]⚠️  Dropped {czrn_dropped_count:,} of {filtered_count:,} filtered records due to invalid CZRNs[/yellow]")

        if len(cbf_data) > 0:
            console.print(f"[green]✓ Successfully transformed {len(cbf_data):,} records[/green]")

        return pl.DataFrame(cbf_data)

    def _create_cbf_record(self, row: dict[str, Any]) -> CBFRecord:
        """Create a single CBF record from LiteLLM daily spend row."""

        # Parse date (daily spend tables use date strings like '2025-04-19')
        usage_date = self._parse_date(row.get('date'))

        # Calculate total tokens
        prompt_tokens = int(row.get('prompt_tokens', 0))
        completion_tokens = int(row.get('completion_tokens', 0))
        total_tokens = prompt_tokens + completion_tokens

        # Create CloudZero Resource Name (CZRN) as resource_id
        resource_id = self.czrn_generator.create_from_litellm_data(row)

        # Build dimensions for CloudZero
        entity_id = str(row.get('entity_id', ''))
        model = str(row.get('model', ''))
        api_key_hash = str(row.get('api_key', ''))[:8]  # First 8 chars for identification

        dimensions = {
            'entity_type': str(row.get('entity_type', '')),  # 'user' or 'team'
            'entity_id': entity_id,
            'model': model,
            'model_group': str(row.get('model_group', '')),
            'provider': str(row.get('custom_llm_provider', '')),
            'api_key_prefix': api_key_hash,
            'api_requests': str(row.get('api_requests', 0)),
            'successful_requests': str(row.get('successful_requests', 0)),
            'failed_requests': str(row.get('failed_requests', 0)),
            'cache_creation_tokens': str(row.get('cache_creation_input_tokens', 0)),
            'cache_read_tokens': str(row.get('cache_read_input_tokens', 0)),
        }

        # Extract CZRN components to populate corresponding CBF columns
        czrn_components = self.czrn_generator.extract_components(resource_id)
        service_type, provider, region, owner_account_id, resource_type, cloud_local_id = czrn_components

        # CloudZero CBF format with proper column names
        cbf_record = {
            # Required CBF fields
            'time/usage_start': usage_date.isoformat() if usage_date else None,  # Required: ISO-formatted UTC datetime
            'cost/cost': float(row.get('spend', 0.0)),  # Required: billed cost
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
        cbf_record['resource/tag:provider'] = provider  # CZRN provider component
        cbf_record['resource/tag:model'] = cloud_local_id  # CZRN cloud-local-id component (model)

        # Add resource tags for all dimensions (using resource/tag:<key> format)
        for key, value in dimensions.items():
            if value and value != 'N/A':  # Only add non-empty tags
                cbf_record[f'resource/tag:{key}'] = str(value)

        # Add token breakdown as resource tags for analysis
        if prompt_tokens > 0:
            cbf_record['resource/tag:prompt_tokens'] = str(prompt_tokens)
        if completion_tokens > 0:
            cbf_record['resource/tag:completion_tokens'] = str(completion_tokens)
        if total_tokens > 0:
            cbf_record['resource/tag:total_tokens'] = str(total_tokens)

        return CBFRecord(cbf_record)

    def _parse_date(self, date_str) -> Optional[datetime]:
        """Parse date string from daily spend tables (e.g., '2025-04-19')."""
        if date_str is None:
            return None

        if isinstance(date_str, datetime):
            return date_str

        if isinstance(date_str, str):
            try:
                # Parse date string and set to midnight UTC for daily aggregation
                return pl.Series([date_str]).str.to_datetime("%Y-%m-%d").item()
            except Exception:
                try:
                    # Fallback: try ISO format parsing
                    return pl.Series([date_str]).str.to_datetime().item()
                except Exception:
                    return None

        return None


