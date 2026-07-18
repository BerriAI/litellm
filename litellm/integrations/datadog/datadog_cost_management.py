import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

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

# Reserved tag keys whose values come from trusted sources (infra env, LiteLLM
# core payload fields, or proxy-controlled auth metadata). User-supplied
# request_tags / metadata cannot overwrite these, even when the key is
# allowlisted via cost_tag_keys, because that would let an authenticated caller
# spoof cost attribution (e.g. request_tags=["team:victim-team"]).
_RESERVED_TAG_KEYS: frozenset = frozenset(
    {
        "env",
        "service",
        "host",
        "pod_name",
        "provider",
        "model",
        "model_id",
        "team",
        "user",
        "model_group",
    }
)


class DatadogCostManagementLogger(CustomBatchLogger):
    def __init__(self, cost_tag_keys: Optional[List[str]] = None, **kwargs):
        self.cost_tag_keys: List[str] = list(cost_tag_keys) if cost_tag_keys else []
        self.dd_api_key = os.getenv("DD_API_KEY")
        self.dd_app_key = os.getenv("DD_APP_KEY")
        self.dd_site = os.getenv("DD_SITE", "datadoghq.com")

        if not self.dd_api_key or not self.dd_app_key:
            verbose_logger.warning(
                "Datadog Cost Management: DD_API_KEY and DD_APP_KEY are required. Integration will not work."
            )

        self.upload_url = f"https://api.{self.dd_site}/api/v2/cost/custom_costs"

        self.async_client = get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)

        # Initialize lock and start periodic flush task
        self.flush_lock = asyncio.Lock()
        asyncio.create_task(self.periodic_flush())

        # Check if flush_lock is already in kwargs to avoid double passing (unlikely but safe)
        if "flush_lock" not in kwargs:
            kwargs["flush_lock"] = self.flush_lock

        super().__init__(**kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_logging_object: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)

            if standard_logging_object is None:
                return

            # Only log if there is a cost associated
            if standard_logging_object.get("response_cost", 0) > 0:
                self.log_queue.append(standard_logging_object)

                if len(self.log_queue) >= self.batch_size:
                    await self.async_send_batch()

        except Exception as e:
            verbose_logger.exception(f"Datadog Cost Management: Error in async_log_success_event: {str(e)}")

    async def async_send_batch(self):
        if not self.log_queue:
            return

        batch_to_send = self.log_queue[:]
        self.log_queue = []

        try:
            aggregated_entries = self._aggregate_costs(batch_to_send)
            if not aggregated_entries:
                verbose_logger.debug(
                    "Datadog Cost Management: batch produced no aggregable entries; dropping %d log(s) from queue.",
                    len(batch_to_send),
                )
                return
            await self._upload_to_datadog(aggregated_entries)
        except Exception as e:
            self.log_queue = batch_to_send + self.log_queue
            verbose_logger.exception(f"Datadog Cost Management: Error in async_send_batch: {str(e)}")

    def _aggregate_costs(self, logs: List[StandardLoggingPayload]) -> List[DatadogFOCUSCostEntry]:
        """
        Aggregates costs by Provider, Model, and Date.
        Returns a list of DatadogFOCUSCostEntry.
        """
        aggregator: Dict[Tuple[str, str, str, Tuple[Tuple[str, str], ...]], DatadogFOCUSCostEntry] = {}

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
                verbose_logger.warning(f"Error processing log for cost aggregation: {e}")
                continue

        return list(aggregator.values())

    def _extract_tags(self, log: StandardLoggingPayload) -> Dict[str, str]:
        tags: Dict[str, str] = {
            "env": get_datadog_env(),
            "service": get_datadog_service(),
            "host": get_datadog_hostname(),
            "pod_name": get_datadog_pod_name(),
        }

        # Always-on canonical FOCUS dimensions from top-level payload fields.
        # Non-sensitive and required for Datadog Custom Costs per-model attribution.
        self._add_tag(tags, "provider", log.get("custom_llm_provider"))
        self._add_tag(tags, "model", log.get("model"))
        self._add_tag(tags, "model_id", log.get("model_id"))

        # cast because StandardLoggingMetadata is a TypedDict; we iterate it
        # as a generic mapping below.
        metadata: Dict[str, Any] = cast(Dict[str, Any], log.get("metadata") or {})

        # Backwards-compat: team/user/model_group preserved regardless of allowlist.
        if metadata.get("user_api_key_alias"):
            tags["user"] = str(metadata["user_api_key_alias"])
        team_tag = (
            metadata.get("user_api_key_team_alias")
            or metadata.get("team_alias")
            or metadata.get("user_api_key_team_id")
            or metadata.get("team_id")
        )
        if team_tag:
            tags["team"] = str(team_tag)
        if metadata.get("model_group"):
            tags["model_group"] = str(metadata["model_group"])

        # Allowlist-gated: request_tags (split on `:`) and arbitrary metadata.*.
        # Reserved keys are hard-blocked here regardless of allowlist membership —
        # see _RESERVED_TAG_KEYS for the rationale.
        if self.cost_tag_keys:
            allow = set(self.cost_tag_keys)
            for rt in log.get("request_tags") or []:
                if not isinstance(rt, str) or ":" not in rt:
                    continue
                k, _, v = rt.partition(":")
                if k in allow and v:
                    self._set_custom_tag(tags, k, v)
            for k, v in metadata.items():
                if k in allow and v is not None and not isinstance(v, (dict, list)):
                    self._set_custom_tag(tags, k, str(v))
            for nested_key in ("spend_logs_metadata", "requester_metadata"):
                nested = metadata.get(nested_key)
                if isinstance(nested, dict):
                    for k, v in nested.items():
                        if k in allow and v is not None and not isinstance(v, (dict, list)):
                            self._set_custom_tag(tags, k, str(v))

        return tags

    @staticmethod
    def _set_custom_tag(tags: Dict[str, str], key: str, value: str) -> None:
        if key in _RESERVED_TAG_KEYS:
            verbose_logger.debug(
                "Datadog Cost Management: dropping user-supplied tag %r=%r — "
                "key is reserved for trusted cost attribution.",
                key,
                value,
            )
            return
        tags[key] = value

    @staticmethod
    def _add_tag(tags: Dict[str, str], key: str, value: Any) -> None:
        if value:
            tags[key] = str(value)

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

        response = await self.async_client.put(self.upload_url, content=data_json, headers=headers)

        response.raise_for_status()

        verbose_logger.debug(
            f"Datadog Cost Management: Uploaded {len(payload)} cost entries. Status: {response.status_code}"
        )
