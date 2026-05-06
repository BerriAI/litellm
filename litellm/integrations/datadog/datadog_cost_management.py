import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.datadog.datadog_handler import (
    get_datadog_env,
    get_datadog_hostname,
    get_datadog_pod_name,
    get_datadog_service,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.datadog_cost_management import (
    DatadogFOCUSCostEntry,
)
from litellm.types.utils import StandardLoggingPayload


class DatadogCostManagementLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        self.dd_api_key = os.getenv("DD_API_KEY")
        self.dd_app_key = os.getenv("DD_APP_KEY")
        self.dd_site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.dd_api_key or not self.dd_app_key:
            verbose_logger.warning(
                "Datadog Cost Management: DD_API_KEY and DD_APP_KEY are required. Integration will not work."
            )

        self.upload_url = f"https://api.{self.dd_site}/api/v2/cost/custom_costs"

        self.async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

        # Initialize lock and start periodic flush task
        self.flush_lock = asyncio.Lock()
        asyncio.create_task(self.periodic_flush())

        # Check if flush_lock is already in kwargs to avoid double passing (unlikely but safe)
        if "flush_lock" not in kwargs:
            kwargs["flush_lock"] = self.flush_lock

        super().__init__(**kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if standard_logging_object is None:
                return

            # Only log if there is a cost associated
            if standard_logging_object.get("response_cost", 0) > 0:
                self.log_queue.append(standard_logging_object)

                if len(self.log_queue) >= self.batch_size:
                    await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(
                f"Datadog Cost Management: Error in async_log_success_event: {str(e)}"
            )

    async def async_send_batch(self):
        if not self.log_queue:
            return

        batch_to_send = self.log_queue[:]
        self.log_queue = []

        try:
            # Aggregate costs from the batch
            aggregated_entries = self._aggregate_costs(batch_to_send)

            if not aggregated_entries:
                return

            # Send to Datadog
            await self._upload_to_datadog(aggregated_entries)

        except Exception as e:
            self.log_queue = batch_to_send + self.log_queue
            verbose_logger.exception(
                f"Datadog Cost Management: Error in async_send_batch: {str(e)}"
            )

    def _aggregate_costs(
        self, logs: List[StandardLoggingPayload]
    ) -> List[DatadogFOCUSCostEntry]:
        """
        Aggregates costs by Provider, Model, and Date.
        Returns a list of DatadogFOCUSCostEntry.
        """
        aggregator: Dict[
            Tuple[str, str, str, Tuple[Tuple[str, str], ...]], DatadogFOCUSCostEntry
        ] = {}

        for log in logs:
            try:
                # Extract keys for aggregation
                provider = log.get("custom_llm_provider") or "unknown"
                model = log.get("model") or "unknown"
                cost = log.get("response_cost", 0)

                if cost == 0:
                    continue

                # Get date strings (FOCUS format requires specific keys, but for aggregation we group by Day)
                # UTC date
                # We interpret "ChargePeriod" as the day of the request.
                ts = log.get("startTime") or time.time()
                dt = datetime.fromtimestamp(ts)
                date_str = dt.strftime("%Y-%m-%d")

                # ChargePeriodStart and End
                # If we want daily granularity, end date is usually same day or next day?
                # Datadog Custom Costs usually expects periods.
                # "ChargePeriodStart": "2023-01-01", "ChargePeriodEnd": "2023-12-31" in example.
                # If we send daily, we can say Start=Date, End=Date.

                # Grouping Key: Provider + Model + Date + Tags?
                # For simplicity, let's aggregate by Provider + Model + Date first.
                # If we handle tags, we need to include them in the key.

                tags = self._extract_tags(log)
                tags_key = tuple(sorted(tags.items())) if tags else ()

                key = (provider, model, date_str, tags_key)

                if key not in aggregator:
                    aggregator[key] = {
                        "ProviderName": provider,
                        "ChargeDescription": f"LLM Usage for {model}",
                        "ChargePeriodStart": date_str,
                        "ChargePeriodEnd": date_str,
                        "BilledCost": 0.0,
                        "BillingCurrency": "USD",
                        "Tags": tags if tags else None,
                    }

                aggregator[key]["BilledCost"] += cost

            except Exception as e:
                verbose_logger.warning(
                    f"Error processing log for cost aggregation: {e}"
                )
                continue

        return list(aggregator.values())

    def _extract_tags(self, log: StandardLoggingPayload) -> Dict[str, str]:
        tags = {
            "env": get_datadog_env(),
            "service": get_datadog_service(),
            "host": get_datadog_hostname(),
            "pod_name": get_datadog_pod_name(),
        }

        self._add_tag_if_present(
            tags=tags, key="provider", value=log.get("custom_llm_provider")
        )
        self._add_tag_if_present(tags=tags, key="model", value=log.get("model"))
        self._add_tag_if_present(
            tags=tags, key="model_group", value=log.get("model_group")
        )
        self._add_tag_if_present(tags=tags, key="model_id", value=log.get("model_id"))
        self._add_request_tags(
            tags=tags, request_tags=log.get("request_tags", []) or []
        )

        # Add metadata as tags
        metadata = log.get("metadata", {})
        if metadata:
            if metadata.get("user_api_key_alias"):
                tags["user"] = str(metadata["user_api_key_alias"])

            # Add Team Tag
            team_tag = (
                metadata.get("user_api_key_team_alias")
                or metadata.get("team_alias")  # type: ignore
                or metadata.get("user_api_key_team_id")
                or metadata.get("team_id")  # type: ignore
            )

            if team_tag:
                tags["team"] = str(team_tag)
            model_group = metadata.get("model_group")  # type: ignore[misc]
            if model_group:
                tags["model_group"] = str(model_group)

            self._add_metadata_tags(tags=tags, metadata=metadata)

        return tags

    @staticmethod
    def _add_tag_if_present(tags: Dict[str, str], key: str, value: Any) -> None:
        if key and isinstance(value, (str, int, float, bool)) and str(value):
            tags[key] = str(value)

    def _add_request_tags(self, tags: Dict[str, str], request_tags: List[Any]) -> None:
        for tag in request_tags:
            if not isinstance(tag, str) or not tag:
                continue

            if ":" in tag:
                key, value = tag.split(":", 1)
                self._add_tag_if_present(tags=tags, key=key, value=value)
            else:
                self._add_tag_if_present(tags=tags, key="request_tag", value=tag)

    def _add_metadata_tags(
        self, tags: Dict[str, str], metadata: Dict[str, Any]
    ) -> None:
        excluded_metadata_keys = {
            "user_api_key_alias",
            "user_api_key_team_alias",
            "team_alias",
            "user_api_key_team_id",
            "team_id",
            "model_group",
            "prompt_management_metadata",
            "mcp_tool_call_metadata",
            "vector_store_request_metadata",
            "usage_object",
            "cold_storage_object_key",
            "requester_custom_headers",
        }
        nested_tag_metadata_keys = {"spend_logs_metadata", "requester_metadata"}

        for key, value in metadata.items():
            if key in excluded_metadata_keys:
                continue

            if key in nested_tag_metadata_keys and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    self._add_tag_if_present(
                        tags=tags,
                        key=str(nested_key),
                        value=nested_value,
                    )
                continue

            self._add_tag_if_present(tags=tags, key=key, value=value)

    async def _upload_to_datadog(self, payload: List[Dict]):
        if not self.dd_api_key or not self.dd_app_key:
            return

        headers = {
            "Content-Type": "application/json",
            "DD-API-KEY": self.dd_api_key,
            "DD-APPLICATION-KEY": self.dd_app_key,
        }

        # The API endpoint expects a list of objects directly in the body (file content behavior)
        data_json = safe_dumps(payload)

        response = await self.async_client.put(
            self.upload_url, content=data_json, headers=headers
        )

        response.raise_for_status()

        verbose_logger.debug(
            f"Datadog Cost Management: Uploaded {len(payload)} cost entries. Status: {response.status_code}"
        )
