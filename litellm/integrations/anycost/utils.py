"""
AnyCost Utilities for CloudZero Integration

Utility functions for handling data aggregation, transformation, and sending
to CloudZero's AnyCost platform.
"""

import gzip
from datetime import datetime
from typing import Any, Dict, List

import boto3
import httpx

from litellm._logging import verbose_logger
from litellm.types.integrations.anycost import (
    AnyCostConfig,
    AnyCostMetrics,
    CBFBillingPeriod,
    CBFFile,
    CBFLineItem,
    CBFUsageInfo,
    TelemetryRecord,
)


class AnyCostUtils:
    """Utility class for AnyCost CloudZero integration"""

    def __init__(self, config: AnyCostConfig):
        """
        Initialize AnyCost utilities

        Args:
            config: AnyCost configuration object
        """
        self.config = config

        # Initialize S3 client if CBF format is enabled
        if config.cbf_file_format:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None

        # Initialize HTTP client for API calls
        self.http_client = httpx.AsyncClient()

        verbose_logger.debug("AnyCost utilities initialized")

    async def get_daily_spend_data(
        self,
        prisma_client,
        date: datetime,
        entity_type: str = "teams"
    ) -> List[Any]:
        """
        Fetch daily spend data from LiteLLM database

        Args:
            prisma_client: LiteLLM Prisma client
            date: Date to fetch data for
            entity_type: Type of entity to fetch (teams, users, etc.)

        Returns:
            List of spend records
        """
        try:
            # Define date range for the day
            start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = date.replace(hour=23, minute=59, second=59, microsecond=999999)

            # Select appropriate table based on charge_by configuration
            table_name = self._get_table_name(entity_type)

            verbose_logger.debug(
                f"Fetching data from {table_name} for {start_date.date()}"
            )

            # Query the appropriate daily spend table
            if table_name == "litellm_dailyteamspend":
                data = await prisma_client.db.litellm_dailyteamspend.find_many(
                    where={
                        "updated_at": {
                            "gte": start_date,
                            "lte": end_date
                        }
                    }
                )
            elif table_name == "litellm_dailyuserspend":
                data = await prisma_client.db.litellm_dailyuserspend.find_many(
                    where={
                        "updated_at": {
                            "gte": start_date,
                            "lte": end_date
                        }
                    }
                )
            else:
                verbose_logger.warning(f"Unknown table name: {table_name}")
                return []

            verbose_logger.debug(f"Retrieved {len(data)} records from {table_name}")
            return data

        except Exception as e:
            verbose_logger.error(f"Error fetching daily spend data: {e}")
            return []

    async def get_key_metadata(self, prisma_client, api_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch key metadata including key_alias for service mapping

        Args:
            prisma_client: LiteLLM Prisma client
            api_keys: List of API keys to fetch metadata for

        Returns:
            Dict mapping api_key to metadata dict
        """
        try:
            if not api_keys:
                return {}

            # Fetch key metadata from LiteLLM_VerificationToken table
            keys_data = await prisma_client.db.litellm_verificationtoken.find_many(
                where={
                    "token": {"in": api_keys}
                },
                select={
                    "token": True,
                    "key_alias": True,
                    "key_name": True,
                    "team_id": True,
                    "user_id": True,
                    "metadata": True
                }
            )

            # Create mapping
            key_metadata = {}
            for key_data in keys_data:
                key_metadata[key_data.token] = {
                    "key_alias": key_data.key_alias,
                    "key_name": key_data.key_name,
                    "team_id": getattr(key_data, 'team_id', None),
                    "user_id": getattr(key_data, 'user_id', None),
                    "metadata": getattr(key_data, 'metadata', {}),
                }

            verbose_logger.debug(f"Retrieved metadata for {len(key_metadata)} keys")
            return key_metadata

        except Exception as e:
            verbose_logger.error(f"Error fetching key metadata: {e}")
            return {}

    def _get_table_name(self, entity_type: str) -> str:
        """Get the appropriate table name based on entity type"""
        if entity_type == "teams":
            return "litellm_dailyteamspend"
        elif entity_type == "users":
            return "litellm_dailyuserspend"
        else:
            return "litellm_dailyuserspend"  # Default to user table

    def aggregate_spend_data(self, raw_data: List[Any]) -> AnyCostMetrics:
        """
        Aggregate raw spend data into AnyCost metrics

        Args:
            raw_data: Raw spend records from database

        Returns:
            Aggregated metrics
        """
        metrics = AnyCostMetrics()

        for record in raw_data:
            # Update main metrics
            metrics.spend += getattr(record, 'spend', 0)
            metrics.prompt_tokens += getattr(record, 'prompt_tokens', 0)
            metrics.completion_tokens += getattr(record, 'completion_tokens', 0)
            metrics.cache_read_input_tokens += getattr(
                record, 'cache_read_input_tokens', 0
            )
            metrics.cache_creation_input_tokens += getattr(
                record, 'cache_creation_input_tokens', 0
            )
            metrics.api_requests += getattr(record, 'api_requests', 0)
            metrics.successful_requests += getattr(record, 'successful_requests', 0)
            metrics.failed_requests += getattr(record, 'failed_requests', 0)

            # Update provider breakdown
            provider = getattr(record, 'custom_llm_provider', 'unknown')
            if provider not in metrics.providers:
                metrics.providers[provider] = 0
            metrics.providers[provider] += getattr(record, 'spend', 0)

            # Update model breakdown
            model = getattr(record, 'model', 'unknown')
            if model not in metrics.models:
                metrics.models[model] = 0
            metrics.models[model] += getattr(record, 'spend', 0)

        # Calculate total tokens
        metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens

        verbose_logger.debug(
            f"Aggregated metrics: ${metrics.spend:.2f} spend, "
            f"{metrics.total_tokens} tokens, {metrics.api_requests} requests"
        )

        return metrics

    async def transform_to_cbf(
        self,
        raw_data: List[Any],
        key_metadata: Dict[str, Dict[str, Any]],
        billing_period_start: datetime,
        billing_period_end: datetime
    ) -> CBFFile:
        """
        Transform raw spend data to CBF format with detailed breakdowns

        Args:
            raw_data: Raw spend records from database
            key_metadata: Metadata for API keys including aliases
            billing_period_start: Start of billing period
            billing_period_end: End of billing period

        Returns:
            CBF file structure
        """
        # Create billing period
        billing_period = CBFBillingPeriod(
            start_date=billing_period_start.isoformat(),
            end_date=billing_period_end.isoformat()
        )

        line_items = []

        # Group data by team, service (key), and provider/model
        grouped_data = self._group_data_for_cbf(raw_data, key_metadata)

        # Create line items for each group
        for group_key, group_records in grouped_data.items():
            team_id, service_name, provider, model = group_key

            # Calculate aggregated metrics for this group
            group_metrics = self._calculate_group_metrics(group_records)

            # Create CBF line item
            line_item = self._create_cbf_line_item(
                team_id=team_id,
                service_name=service_name,
                provider=provider,
                model=model,
                metrics=group_metrics,
                billing_period=billing_period
            )

            line_items.append(line_item)

        verbose_logger.debug(f"Created {len(line_items)} CBF line items")

        return CBFFile(
            billing_period=billing_period,
            line_items=line_items
        )

    def _group_data_for_cbf(
        self, raw_data: List[Any], key_metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[tuple, List[Any]]:
        """Group raw data by team, service, provider, and model"""
        grouped = {}

        for record in raw_data:
            # Extract dimensions
            team_id = getattr(record, 'team_id', 'unknown') or 'unknown'
            api_key = getattr(record, 'api_key', 'unknown')
            provider = getattr(record, 'custom_llm_provider', 'unknown') or 'unknown'
            model = getattr(record, 'model', 'unknown') or 'unknown'

            # Get service name from key metadata
            key_info = key_metadata.get(api_key, {})
            key_alias = key_info.get('key_alias') or key_info.get('key_name')
            service_name = f"{api_key[:8]}...{f' ({key_alias})' if key_alias else ''}"

            # Create group key
            group_key = (team_id, service_name, provider, model)

            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(record)

        return grouped

    def _calculate_group_metrics(self, records: List[Any]) -> Dict[str, Any]:
        """Calculate aggregated metrics for a group of records"""
        metrics = {
            'spend': 0.0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0,
            'api_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
        }

        for record in records:
            metrics['spend'] += getattr(record, 'spend', 0)
            metrics['prompt_tokens'] += getattr(record, 'prompt_tokens', 0)
            metrics['completion_tokens'] += getattr(record, 'completion_tokens', 0)
            metrics['cache_read_input_tokens'] += getattr(
                record, 'cache_read_input_tokens', 0
            )
            metrics['cache_creation_input_tokens'] += getattr(
                record, 'cache_creation_input_tokens', 0
            )
            metrics['api_requests'] += getattr(record, 'api_requests', 0)
            metrics['successful_requests'] += getattr(record, 'successful_requests', 0)
            metrics['failed_requests'] += getattr(record, 'failed_requests', 0)

        metrics['total_tokens'] = metrics['prompt_tokens'] + metrics['completion_tokens']
        return metrics

    def _create_cbf_line_item(
        self,
        team_id: str,
        service_name: str,
        provider: str,
        model: str,
        metrics: Dict[str, Any],
        billing_period: CBFBillingPeriod
    ) -> CBFLineItem:
        """Create a CBF line item for a specific breakdown"""
        usage_info = [
            CBFUsageInfo(
                metric_name="API Requests",
                usage_amount=float(metrics['api_requests']),
                usage_unit="requests"
            ),
            CBFUsageInfo(
                metric_name="Total Tokens",
                usage_amount=float(metrics['total_tokens']),
                usage_unit="tokens"
            ),
            CBFUsageInfo(
                metric_name="Prompt Tokens",
                usage_amount=float(metrics['prompt_tokens']),
                usage_unit="tokens"
            ),
            CBFUsageInfo(
                metric_name="Completion Tokens",
                usage_amount=float(metrics['completion_tokens']),
                usage_unit="tokens"
            )
        ]

        # Create unique line item ID
        date_str = billing_period.start_date[:10]
        line_item_id = f"litellm-{team_id}-{service_name[:16]}-{provider}-{model}-{date_str}".replace(' ', '-').replace('(', '').replace(')', '')

        return CBFLineItem(
            invoice_id=f"litellm-{date_str}",
            billing_period_start_date=billing_period.start_date,
            billing_period_end_date=billing_period.end_date,
            line_item_id=line_item_id,
            usage_info=usage_info,
            resource_id=f"{provider}/{model}",
            resource_name=f"{provider} - {model}",
            list_cost=metrics['spend'],
            unblended_cost=metrics['spend'],
            blended_cost=metrics['spend'],
            tags={
                "team_id": team_id,
                "service": service_name,
                "provider": provider,
                "model": model,
                "source": "litellm",
                "successful_requests": str(metrics['successful_requests']),
                "failed_requests": str(metrics['failed_requests'])
            }
        )

    async def upload_cbf_to_s3(self, cbf_file: CBFFile, filename: str) -> bool:
        """
        Upload CBF file to S3

        Args:
            cbf_file: CBF file structure
            filename: S3 filename

        Returns:
            True if successful, False otherwise
        """
        if not self.s3_client:
            verbose_logger.error("S3 client not initialized")
            return False

        try:
            # Serialize CBF file to JSON
            cbf_json = cbf_file.model_dump_json(indent=2)

            # Compress the JSON
            compressed_data = gzip.compress(cbf_json.encode('utf-8'))

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}{filename}"

            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=compressed_data,
                ContentType='application/json',
                ContentEncoding='gzip'
            )

            verbose_logger.info(
                f"Successfully uploaded CBF file to "
                f"s3://{self.config.s3_bucket}/{s3_key}"
            )
            return True

        except Exception as e:
            verbose_logger.error(f"Error uploading CBF file to S3: {e}")
            return False

    async def send_telemetry_data(self, date: datetime) -> bool:
        """
        Send data via CloudZero Telemetry API

        Args:
            date: Date to send data for

        Returns:
            True if successful, False otherwise
        """
        # Get Prisma client
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_logger.warning("No database connection available")
            return False

        try:
            # Fetch and process data
            raw_data = await self.get_daily_spend_data(
                prisma_client, date, "teams"
            )

            if not raw_data:
                verbose_logger.info(f"No data found for {date.date()}")
                return True  # Success - no data to send

            # Get key metadata
            api_keys = list(set(getattr(record, 'api_key', '') for record in raw_data))
            key_metadata = await self.get_key_metadata(prisma_client, api_keys)

            # Create telemetry records with detailed breakdowns
            records = self._create_detailed_telemetry_records(raw_data, key_metadata, date)

            # Send to CloudZero API
            return await self._send_telemetry_records(records)

        except Exception as e:
            verbose_logger.error(f"Error sending telemetry data: {e}")
            return False

    def _create_detailed_telemetry_records(
        self, raw_data: List[Any], key_metadata: Dict[str, Dict[str, Any]], timestamp: datetime
    ) -> List[TelemetryRecord]:
        """Create detailed telemetry records with team/service/provider breakdowns"""
        records = []

        # Group data same way as CBF
        grouped_data = self._group_data_for_cbf(raw_data, key_metadata)

        for group_key, group_records in grouped_data.items():
            team_id, service_name, provider, model = group_key
            metrics = self._calculate_group_metrics(group_records)

            # Create spend record
            records.append(TelemetryRecord(
                timestamp=timestamp.isoformat(),
                metric_name="litellm_spend",
                value=metrics['spend'],
                dimensions={
                    "team_id": team_id,
                    "service": service_name,
                    "provider": provider,
                    "model": model,
                    "charge_by": self.config.charge_by.value
                }
            ))

            # Create token usage record
            records.append(TelemetryRecord(
                timestamp=timestamp.isoformat(),
                metric_name="litellm_tokens",
                value=float(metrics['total_tokens']),
                unit="tokens",
                dimensions={
                    "team_id": team_id,
                    "service": service_name,
                    "provider": provider,
                    "model": model,
                    "charge_by": self.config.charge_by.value
                }
            ))

            # Create request count record
            records.append(TelemetryRecord(
                timestamp=timestamp.isoformat(),
                metric_name="litellm_requests",
                value=float(metrics['api_requests']),
                unit="requests",
                dimensions={
                    "team_id": team_id,
                    "service": service_name,
                    "provider": provider,
                    "model": model,
                    "charge_by": self.config.charge_by.value
                }
            ))

        verbose_logger.debug(f"Created {len(records)} telemetry records")
        return records

    async def _send_telemetry_records(self, records: List[TelemetryRecord]) -> bool:
        """Send telemetry records to CloudZero API"""
        try:
            url = f"{self.config.base_url}/v2/telemetry"
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }

            # Send records in batches
            batch_size = 100
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                payload = [record.model_dump() for record in batch]

                response = await self.http_client.post(
                    url, headers=headers, json=payload
                )

                if response.status_code != 200:
                    verbose_logger.error(
                        f"Failed to send telemetry batch: {response.status_code} "
                        f"{response.text}"
                    )
                    return False

            return True

        except Exception as e:
            verbose_logger.error(f"Error sending telemetry records: {e}")
            return False

    async def send_cbf_data(self, date: datetime) -> bool:
        """
        Send CBF data for a specific date

        Args:
            date: Date to send data for

        Returns:
            True if successful, False otherwise
        """
        # Get Prisma client
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_logger.warning("No database connection available")
            return False

        try:
            # Fetch and process data
            raw_data = await self.get_daily_spend_data(
                prisma_client, date, "teams"
            )

            if not raw_data:
                verbose_logger.info(f"No data found for {date.date()}")
                return True  # Success - no data to send

            # Get key metadata
            api_keys = list(set(getattr(record, 'api_key', '') for record in raw_data))
            key_metadata = await self.get_key_metadata(prisma_client, api_keys)

            # Transform to CBF with detailed breakdowns
            billing_period_start = date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            billing_period_end = date.replace(
                hour=23, minute=59, second=59, microsecond=999999
            )

            cbf_file = await self.transform_to_cbf(
                raw_data, key_metadata, billing_period_start, billing_period_end
            )

            # Generate filename
            filename = f"litellm-cbf-{date.strftime('%Y-%m-%d')}.json.gz"

            # Upload to S3
            return await self.upload_cbf_to_s3(cbf_file, filename)

        except Exception as e:
            verbose_logger.error(f"Error sending CBF data: {e}")
            return False

    def generate_cbf_filename(
        self, start_date: datetime, end_date: datetime
    ) -> str:
        """Generate CBF filename based on date range"""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        return f"litellm-cbf-{start_str}-to-{end_str}.json.gz"

    async def close(self):
        """Close HTTP client connections"""
        if self.http_client:
            await self.http_client.aclose()
