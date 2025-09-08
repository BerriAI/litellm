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
# CHANGELOG: 2025-01-19 - Added pathlib for filesystem operations (erik.peterson)
# CHANGELOG: 2025-01-19 - Migrated from pandas to polars and requests to httpx (erik.peterson)
# CHANGELOG: 2025-01-19 - Initial output module for CSV and CloudZero API (erik.peterson)

"""Output modules for writing CBF data to various destinations."""

import zoneinfo
from datetime import datetime, timezone
from typing import Any, Optional, Union

import httpx
import polars as pl
from rich.console import Console


class CloudZeroStreamer:
    """Stream CBF data to CloudZero AnyCost API with proper batching and timezone handling."""

    def __init__(self, api_key: str, connection_id: str, user_timezone: Optional[str] = None):
        """Initialize CloudZero streamer with credentials."""
        self.api_key = api_key
        self.connection_id = connection_id
        self.base_url = "https://api.cloudzero.com"
        self.console = Console()

        # Set timezone - default to UTC
        self.user_timezone: Union[zoneinfo.ZoneInfo, timezone]
        if user_timezone:
            try:
                self.user_timezone = zoneinfo.ZoneInfo(user_timezone)
            except zoneinfo.ZoneInfoNotFoundError:
                self.console.print(f"[yellow]Warning: Unknown timezone '{user_timezone}', using UTC[/yellow]")
                self.user_timezone = timezone.utc
        else:
            self.user_timezone = timezone.utc

    def send_batched(self, data: pl.DataFrame, operation: str = "replace_hourly") -> None:
        """Send CBF data in daily batches to CloudZero AnyCost API."""
        if data.is_empty():
            self.console.print("[yellow]No data to send to CloudZero[/yellow]")
            return

        # Group data by date and send each day as a batch
        daily_batches = self._group_by_date(data)

        if not daily_batches:
            self.console.print("[yellow]No valid daily batches to send[/yellow]")
            return

        self.console.print(f"[blue]Sending {len(daily_batches)} daily batch(es) with operation '{operation}'[/blue]")

        for batch_date, batch_data in daily_batches.items():
            self._send_daily_batch(batch_date, batch_data, operation)

    def _group_by_date(self, data: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Group data by date, converting to UTC and validating dates."""
        daily_batches: dict[str, list[dict[str, Any]]] = {}

        # Ensure we have the required columns
        if 'time/usage_start' not in data.columns:
            self.console.print("[red]Error: Missing 'time/usage_start' column for date grouping[/red]")
            return {}
        
        timestamp_str: Optional[str] = None
        for row in data.iter_rows(named=True):
            try:
                # Parse the timestamp and convert to UTC
                timestamp_str = row.get('time/usage_start')
                if not timestamp_str:
                    continue

                # Parse timestamp and handle timezone conversion
                dt = self._parse_and_convert_timestamp(timestamp_str)
                batch_date = dt.strftime('%Y-%m-%d')

                if batch_date not in daily_batches:
                    daily_batches[batch_date] = []

                daily_batches[batch_date].append(row)

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not process timestamp '{timestamp_str}': {e}[/yellow]")
                continue

        # Convert lists back to DataFrames
        return {date_key: pl.DataFrame(records) for date_key, records in daily_batches.items() if records}

    def _parse_and_convert_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string and convert to UTC."""
        # Try to parse the timestamp string
        try:
            # Handle various ISO 8601 formats
            if timestamp_str.endswith('Z'):
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            elif '+' in timestamp_str or timestamp_str.endswith(('-00:00', '-01:00', '-02:00', '-03:00',
                                                                   '-04:00', '-05:00', '-06:00', '-07:00',
                                                                   '-08:00', '-09:00', '-10:00', '-11:00',
                                                                   '-12:00', '+01:00', '+02:00', '+03:00',
                                                                   '+04:00', '+05:00', '+06:00', '+07:00',
                                                                   '+08:00', '+09:00', '+10:00', '+11:00', '+12:00')):
                dt = datetime.fromisoformat(timestamp_str)
            else:
                # Assume user timezone if no timezone info
                dt = datetime.fromisoformat(timestamp_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.user_timezone)

            # Convert to UTC
            return dt.astimezone(timezone.utc)

        except ValueError as e:
            raise ValueError(f"Could not parse timestamp '{timestamp_str}': {e}")

    def _send_daily_batch(self, batch_date: str, batch_data: pl.DataFrame, operation: str) -> None:
        """Send a single daily batch to CloudZero API."""
        if batch_data.is_empty():
            return

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # Use the correct API endpoint format from documentation
        url = f"{self.base_url}/v2/connections/billing/anycost/{self.connection_id}/billing_drops"

        # Prepare the batch payload according to AnyCost API format
        payload = self._prepare_batch_payload(batch_date, batch_data, operation)

        try:
            with httpx.Client(timeout=30.0) as client:
                self.console.print(f"[blue]Sending batch for {batch_date} ({len(batch_data)} records)[/blue]")

                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                self.console.print(f"[green]✓ Successfully sent batch for {batch_date} ({len(batch_data)} records)[/green]")

        except httpx.RequestError as e:
            self.console.print(f"[red]✗ Network error sending batch for {batch_date}: {e}[/red]")
            raise
        except httpx.HTTPStatusError as e:
            self.console.print(f"[red]✗ HTTP error sending batch for {batch_date}: {e.response.status_code} {e.response.text}[/red]")
            raise

    def _prepare_batch_payload(self, batch_date: str, batch_data: pl.DataFrame, operation: str) -> dict[str, Any]:
        """Prepare batch payload according to CloudZero AnyCost API format."""
        # Convert batch_date to month for the API (YYYY-MM format)
        try:
            date_obj = datetime.strptime(batch_date, '%Y-%m-%d')
            month_str = date_obj.strftime('%Y-%m')
        except ValueError:
            # Fallback to current month
            month_str = datetime.now().strftime('%Y-%m')

        # Convert DataFrame rows to API format
        data_records = []
        for row in batch_data.iter_rows(named=True):
            record = self._convert_cbf_to_api_format(row)
            if record:
                data_records.append(record)

        payload = {
            'month': month_str,
            'operation': operation,
            'data': data_records
        }

        return payload

    def _convert_cbf_to_api_format(self, row: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Convert CBF row to CloudZero API format - keeping CBF field names as CloudZero expects them."""
        try:
            # CloudZero expects CBF format field names directly, not converted names
            api_record = {}

            # Copy all CBF fields, converting numeric values to strings as required by CloudZero
            for key, value in row.items():
                if value is not None:
                    # CloudZero requires numeric values to be strings, but NOT in scientific notation
                    if isinstance(value, (int, float)):
                        # Format floats to avoid scientific notation
                        if isinstance(value, float):
                            # Use a reasonable precision that avoids scientific notation
                            api_record[key] = f"{value:.10f}".rstrip('0').rstrip('.')
                        else:
                            api_record[key] = str(value)
                    else:
                        api_record[key] = value

            # Ensure timestamp is in UTC format
            if 'time/usage_start' in api_record:
                api_record['time/usage_start'] = self._ensure_utc_timestamp(api_record['time/usage_start'])

            return api_record

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not convert record to API format: {e}[/yellow]")
            return None

    def _ensure_utc_timestamp(self, timestamp_str: str) -> str:
        """Ensure timestamp is in UTC format for API."""
        if not timestamp_str:
            return datetime.now(timezone.utc).isoformat()

        try:
            dt = self._parse_and_convert_timestamp(timestamp_str)
            return dt.isoformat().replace('+00:00', 'Z')
        except Exception:
            # Fallback to current time in UTC
            return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


