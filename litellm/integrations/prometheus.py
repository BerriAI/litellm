# used for /metrics endpoint on LiteLLM Proxy
#### What this does ####
#    On success, log events to Prometheus
import sys
from datetime import datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    cast,
)

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.types.integrations.prometheus import *
from litellm.types.utils import StandardLoggingPayload
from litellm.utils import get_end_user_id_for_cost_tracking

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any


class PrometheusLogger(CustomLogger):
    # Class variables or attributes
    def __init__(
        self,
        **kwargs,
    ):
        try:
            from prometheus_client import Counter, Gauge, Histogram

            from litellm.proxy.proxy_server import CommonProxyErrors, premium_user

            # Always initialize label_filters, even for non-premium users
            self.label_filters = self._parse_prometheus_config()

            if premium_user is not True:
                verbose_logger.warning(
                    f"🚨🚨🚨 Prometheus Metrics is on LiteLLM Enterprise\n🚨 {CommonProxyErrors.not_premium_user.value}"
                )
                self.litellm_not_a_premium_user_metric = Counter(
                    name="litellm_not_a_premium_user_metric",
                    documentation=f"🚨🚨🚨 Prometheus Metrics is on LiteLLM Enterprise. 🚨 {CommonProxyErrors.not_premium_user.value}",
                )
                return

            # Create metric factory functions
            self._counter_factory = self._create_metric_factory(Counter)
            self._gauge_factory = self._create_metric_factory(Gauge)
            self._histogram_factory = self._create_metric_factory(Histogram)

            self.litellm_proxy_failed_requests_metric = self._counter_factory(
                name="litellm_proxy_failed_requests_metric",
                documentation="Total number of failed responses from proxy - the client did not get a success response from litellm proxy",
                labelnames=self.get_labels_for_metric(
                    "litellm_proxy_failed_requests_metric"
                ),
            )

            self.litellm_proxy_total_requests_metric = self._counter_factory(
                name="litellm_proxy_total_requests_metric",
                documentation="Total number of requests made to the proxy server - track number of client side requests",
                labelnames=self.get_labels_for_metric(
                    "litellm_proxy_total_requests_metric"
                ),
            )

            # request latency metrics
            self.litellm_request_total_latency_metric = self._histogram_factory(
                "litellm_request_total_latency_metric",
                "Total latency (seconds) for a request to LiteLLM",
                labelnames=self.get_labels_for_metric(
                    "litellm_request_total_latency_metric"
                ),
                buckets=LATENCY_BUCKETS,
            )

            self.litellm_llm_api_latency_metric = self._histogram_factory(
                "litellm_llm_api_latency_metric",
                "Total latency (seconds) for a models LLM API call",
                labelnames=self.get_labels_for_metric("litellm_llm_api_latency_metric"),
                buckets=LATENCY_BUCKETS,
            )

            self.litellm_llm_api_time_to_first_token_metric = self._histogram_factory(
                "litellm_llm_api_time_to_first_token_metric",
                "Time to first token for a models LLM API call",
                labelnames=[
                    "model",
                    "hashed_api_key",
                    "api_key_alias",
                    "team",
                    "team_alias",
                ],
                buckets=LATENCY_BUCKETS,
            )

            # Counter for spend
            self.litellm_spend_metric = self._counter_factory(
                "litellm_spend_metric",
                "Total spend on LLM requests",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            # Counter for total_output_tokens
            self.litellm_tokens_metric = self._counter_factory(
                "litellm_total_tokens",
                "Total number of input + output tokens from LLM requests",
                labelnames=self.get_labels_for_metric("litellm_total_tokens_metric"),
            )

            self.litellm_input_tokens_metric = self._counter_factory(
                "litellm_input_tokens",
                "Total number of input tokens from LLM requests",
                labelnames=self.get_labels_for_metric("litellm_input_tokens_metric"),
            )

            self.litellm_output_tokens_metric = self._counter_factory(
                "litellm_output_tokens",
                "Total number of output tokens from LLM requests",
                labelnames=self.get_labels_for_metric("litellm_output_tokens_metric"),
            )

            # Remaining Budget for Team
            self.litellm_remaining_team_budget_metric = self._gauge_factory(
                "litellm_remaining_team_budget_metric",
                "Remaining budget for team",
                labelnames=self.get_labels_for_metric(
                    "litellm_remaining_team_budget_metric"
                ),
            )

            # Max Budget for Team
            self.litellm_team_max_budget_metric = self._gauge_factory(
                "litellm_team_max_budget_metric",
                "Maximum budget set for team",
                labelnames=self.get_labels_for_metric("litellm_team_max_budget_metric"),
            )

            # Team Budget Reset At
            self.litellm_team_budget_remaining_hours_metric = self._gauge_factory(
                "litellm_team_budget_remaining_hours_metric",
                "Remaining days for team budget to be reset",
                labelnames=self.get_labels_for_metric(
                    "litellm_team_budget_remaining_hours_metric"
                ),
            )

            # Remaining Budget for API Key
            self.litellm_remaining_api_key_budget_metric = self._gauge_factory(
                "litellm_remaining_api_key_budget_metric",
                "Remaining budget for api key",
                labelnames=self.get_labels_for_metric(
                    "litellm_remaining_api_key_budget_metric"
                ),
            )

            # Max Budget for API Key
            self.litellm_api_key_max_budget_metric = self._gauge_factory(
                "litellm_api_key_max_budget_metric",
                "Maximum budget set for api key",
                labelnames=self.get_labels_for_metric(
                    "litellm_api_key_max_budget_metric"
                ),
            )

            self.litellm_api_key_budget_remaining_hours_metric = self._gauge_factory(
                "litellm_api_key_budget_remaining_hours_metric",
                "Remaining hours for api key budget to be reset",
                labelnames=self.get_labels_for_metric(
                    "litellm_api_key_budget_remaining_hours_metric"
                ),
            )

            ########################################
            # LiteLLM Virtual API KEY metrics
            ########################################
            # Remaining MODEL RPM limit for API Key
            self.litellm_remaining_api_key_requests_for_model = self._gauge_factory(
                "litellm_remaining_api_key_requests_for_model",
                "Remaining Requests API Key can make for model (model based rpm limit on key)",
                labelnames=["hashed_api_key", "api_key_alias", "model"],
            )

            # Remaining MODEL TPM limit for API Key
            self.litellm_remaining_api_key_tokens_for_model = self._gauge_factory(
                "litellm_remaining_api_key_tokens_for_model",
                "Remaining Tokens API Key can make for model (model based tpm limit on key)",
                labelnames=["hashed_api_key", "api_key_alias", "model"],
            )

            ########################################
            # LLM API Deployment Metrics / analytics
            ########################################

            # Remaining Rate Limit for model
            self.litellm_remaining_requests_metric = self._gauge_factory(
                "litellm_remaining_requests",
                "LLM Deployment Analytics - remaining requests for model, returned from LLM API Provider",
                labelnames=self.get_labels_for_metric(
                    "litellm_remaining_requests_metric"
                ),
            )

            self.litellm_remaining_tokens_metric = self._gauge_factory(
                "litellm_remaining_tokens",
                "remaining tokens for model, returned from LLM API Provider",
                labelnames=self.get_labels_for_metric(
                    "litellm_remaining_tokens_metric"
                ),
            )

            self.litellm_overhead_latency_metric = self._histogram_factory(
                "litellm_overhead_latency_metric",
                "Latency overhead (milliseconds) added by LiteLLM processing",
                labelnames=self.get_labels_for_metric(
                    "litellm_overhead_latency_metric"
                ),
                buckets=LATENCY_BUCKETS,
            )
            # llm api provider budget metrics
            self.litellm_provider_remaining_budget_metric = self._gauge_factory(
                "litellm_provider_remaining_budget_metric",
                "Remaining budget for provider - used when you set provider budget limits",
                labelnames=["api_provider"],
            )

            # Get all keys
            _logged_llm_labels = [
                UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value,
                UserAPIKeyLabelNames.MODEL_ID.value,
                UserAPIKeyLabelNames.API_BASE.value,
                UserAPIKeyLabelNames.API_PROVIDER.value,
            ]

            # Metric for deployment state
            self.litellm_deployment_state = self._gauge_factory(
                "litellm_deployment_state",
                "LLM Deployment Analytics - The state of the deployment: 0 = healthy, 1 = partial outage, 2 = complete outage",
                labelnames=_logged_llm_labels,
            )

            self.litellm_deployment_cooled_down = self._counter_factory(
                "litellm_deployment_cooled_down",
                "LLM Deployment Analytics - Number of times a deployment has been cooled down by LiteLLM load balancing logic. exception_status is the status of the exception that caused the deployment to be cooled down",
                labelnames=_logged_llm_labels + [EXCEPTION_STATUS],
            )

            self.litellm_deployment_success_responses = self._counter_factory(
                name="litellm_deployment_success_responses",
                documentation="LLM Deployment Analytics - Total number of successful LLM API calls via litellm",
                labelnames=self.get_labels_for_metric(
                    "litellm_deployment_success_responses"
                ),
            )
            self.litellm_deployment_failure_responses = self._counter_factory(
                name="litellm_deployment_failure_responses",
                documentation="LLM Deployment Analytics - Total number of failed LLM API calls for a specific LLM deploymeny. exception_status is the status of the exception from the llm api",
                labelnames=self.get_labels_for_metric(
                    "litellm_deployment_failure_responses"
                ),
            )
            self.litellm_deployment_failure_by_tag_responses = self._counter_factory(
                "litellm_deployment_failure_by_tag_responses",
                "Total number of failed LLM API calls for a specific LLM deploymeny by custom metadata tags",
                labelnames=[
                    UserAPIKeyLabelNames.REQUESTED_MODEL.value,
                    UserAPIKeyLabelNames.TAG.value,
                ]
                + _logged_llm_labels
                + EXCEPTION_LABELS,
            )
            self.litellm_deployment_total_requests = self._counter_factory(
                name="litellm_deployment_total_requests",
                documentation="LLM Deployment Analytics - Total number of LLM API calls via litellm - success + failure",
                labelnames=self.get_labels_for_metric(
                    "litellm_deployment_total_requests"
                ),
            )

            # Deployment Latency tracking
            self.litellm_deployment_latency_per_output_token = self._histogram_factory(
                name="litellm_deployment_latency_per_output_token",
                documentation="LLM Deployment Analytics - Latency per output token",
                labelnames=self.get_labels_for_metric(
                    "litellm_deployment_latency_per_output_token"
                ),
            )

            self.litellm_deployment_successful_fallbacks = self._counter_factory(
                "litellm_deployment_successful_fallbacks",
                "LLM Deployment Analytics - Number of successful fallback requests from primary model -> fallback model",
                self.get_labels_for_metric("litellm_deployment_successful_fallbacks"),
            )

            self.litellm_deployment_failed_fallbacks = self._counter_factory(
                "litellm_deployment_failed_fallbacks",
                "LLM Deployment Analytics - Number of failed fallback requests from primary model -> fallback model",
                self.get_labels_for_metric("litellm_deployment_failed_fallbacks"),
            )

            self.litellm_llm_api_failed_requests_metric = self._counter_factory(
                name="litellm_llm_api_failed_requests_metric",
                documentation="deprecated - use litellm_proxy_failed_requests_metric",
                labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    "model",
                    "team",
                    "team_alias",
                    "user",
                ],
            )

            self.litellm_requests_metric = self._counter_factory(
                name="litellm_requests_metric",
                documentation="deprecated - use litellm_proxy_total_requests_metric. Total number of LLM calls to litellm - track total per API Key, team, user",
                labelnames=self.get_labels_for_metric("litellm_requests_metric"),
            )
        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    def _parse_prometheus_config(self) -> Dict[str, List[str]]:
        """Parse prometheus metrics configuration for label filtering and enabled metrics"""
        import litellm
        from litellm.types.integrations.prometheus import PrometheusMetricsConfig

        config = litellm.prometheus_metrics_config

        # If no config is provided, return empty dict (no filtering)
        if not config:
            return {}

        verbose_logger.debug(f"prometheus config: {config}")

        label_filters = {}
        self.enabled_metrics = set()

        # Parse each configuration group
        for group_config in config:
            # Validate configuration using Pydantic
            if isinstance(group_config, dict):
                parsed_config = PrometheusMetricsConfig(**group_config)
            else:
                parsed_config = group_config

            # Add enabled metrics to the set
            self.enabled_metrics.update(parsed_config.metrics)

            # Set label filters for each metric in this group
            for metric_name in parsed_config.metrics:
                if parsed_config.include_labels:
                    label_filters[metric_name] = parsed_config.include_labels

        # Pretty print the processed configuration
        self._pretty_print_prometheus_config(label_filters)

        return label_filters

    def _pretty_print_prometheus_config(
        self, label_filters: Dict[str, List[str]]
    ) -> None:
        """Pretty print the processed prometheus configuration using rich"""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            console = Console()

            # Create main panel title
            title = Text("Prometheus Configuration Processed", style="bold blue")

            # Create enabled metrics table
            metrics_table = Table(
                title="📊 Enabled Metrics",
                show_header=True,
                header_style="bold magenta",
                title_justify="left",
            )
            metrics_table.add_column("Metric Name", style="cyan", no_wrap=True)

            if hasattr(self, "enabled_metrics") and self.enabled_metrics:
                for metric in sorted(self.enabled_metrics):
                    metrics_table.add_row(metric)
            else:
                metrics_table.add_row(
                    "[yellow]All metrics enabled (no filter applied)[/yellow]"
                )

            # Create label filters table
            labels_table = Table(
                title="🏷️  Label Filters",
                show_header=True,
                header_style="bold green",
                title_justify="left",
            )
            labels_table.add_column("Metric Name", style="cyan", no_wrap=True)
            labels_table.add_column("Allowed Labels", style="yellow")

            if label_filters:
                for metric_name, labels in sorted(label_filters.items()):
                    labels_str = (
                        ", ".join(labels)
                        if labels
                        else "[dim]No labels specified[/dim]"
                    )
                    labels_table.add_row(metric_name, labels_str)
            else:
                labels_table.add_row(
                    "[yellow]No label filtering applied[/yellow]",
                    "[dim]All default labels will be used[/dim]",
                )

            # Print everything in a nice panel
            console.print("\n")
            console.print(Panel(title, border_style="blue"))
            console.print(metrics_table)
            console.print(labels_table)
            console.print("\n")

        except ImportError:
            # Fallback to simple logging if rich is not available
            verbose_logger.info(
                f"Enabled metrics: {sorted(self.enabled_metrics) if hasattr(self, 'enabled_metrics') else 'All metrics'}"
            )
            verbose_logger.info(f"Label filters: {label_filters}")

    def _is_metric_enabled(self, metric_name: str) -> bool:
        """Check if a metric is enabled based on configuration"""
        # If no specific configuration is provided, enable all metrics (default behavior)
        if not hasattr(self, "enabled_metrics"):
            return True

        # If enabled_metrics is empty, enable all metrics
        if not self.enabled_metrics:
            return True

        return metric_name in self.enabled_metrics

    def _create_metric_factory(self, metric_class):
        """Create a factory function that returns either a real metric or a no-op metric"""

        def factory(*args, **kwargs):
            # Extract metric name from the first argument or 'name' keyword argument
            metric_name = args[0] if args else kwargs.get("name", "")

            if self._is_metric_enabled(metric_name):
                return metric_class(*args, **kwargs)
            else:
                return NoOpMetric()

        return factory

    def get_labels_for_metric(
        self, metric_name: DEFINED_PROMETHEUS_METRICS
    ) -> List[str]:
        """
        Get the labels for a metric, filtered if configured
        """
        # Get default labels for this metric from PrometheusMetricLabels
        default_labels = PrometheusMetricLabels.get_labels(metric_name)

        # If no label filtering is configured for this metric, use default labels
        if metric_name not in self.label_filters:
            return default_labels

        # Get configured labels for this metric
        configured_labels = self.label_filters[metric_name]

        # Return intersection of configured and default labels to ensure we only use valid labels
        filtered_labels = [
            label for label in default_labels if label in configured_labels
        ]

        return filtered_labels

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Define prometheus client
        from litellm.types.utils import StandardLoggingPayload

        verbose_logger.debug(
            f"prometheus Logging - Enters success logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )

        if standard_logging_payload is None or not isinstance(
            standard_logging_payload, dict
        ):
            raise ValueError(
                f"standard_logging_object is required, got={standard_logging_payload}"
            )

        model = kwargs.get("model", "")
        litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = litellm_params.get("metadata", {})
        end_user_id = get_end_user_id_for_cost_tracking(
            litellm_params, service_type="prometheus"
        )
        user_id = standard_logging_payload["metadata"]["user_api_key_user_id"]
        user_api_key = standard_logging_payload["metadata"]["user_api_key_hash"]
        user_api_key_alias = standard_logging_payload["metadata"]["user_api_key_alias"]
        user_api_team = standard_logging_payload["metadata"]["user_api_key_team_id"]
        user_api_team_alias = standard_logging_payload["metadata"][
            "user_api_key_team_alias"
        ]
        output_tokens = standard_logging_payload["completion_tokens"]
        tokens_used = standard_logging_payload["total_tokens"]
        response_cost = standard_logging_payload["response_cost"]
        _requester_metadata = standard_logging_payload["metadata"].get(
            "requester_metadata"
        )
        if standard_logging_payload is not None and isinstance(
            standard_logging_payload, dict
        ):
            _tags = standard_logging_payload["request_tags"]
        else:
            _tags = []

        print_verbose(
            f"inside track_prometheus_metrics, model {model}, response_cost {response_cost}, tokens_used {tokens_used}, end_user_id {end_user_id}, user_api_key {user_api_key}"
        )

        enum_values = UserAPIKeyLabelValues(
            end_user=end_user_id,
            hashed_api_key=user_api_key,
            api_key_alias=user_api_key_alias,
            requested_model=standard_logging_payload["model_group"],
            model_group=standard_logging_payload["model_group"],
            team=user_api_team,
            team_alias=user_api_team_alias,
            user=user_id,
            user_email=standard_logging_payload["metadata"]["user_api_key_user_email"],
            status_code="200",
            model=model,
            litellm_model_name=model,
            tags=_tags,
            model_id=standard_logging_payload["model_id"],
            api_base=standard_logging_payload["api_base"],
            api_provider=standard_logging_payload["custom_llm_provider"],
            exception_status=None,
            exception_class=None,
            custom_metadata_labels=get_custom_labels_from_metadata(
                metadata=standard_logging_payload["metadata"].get("requester_metadata")
                or {}
            ),
            route=standard_logging_payload["metadata"].get(
                "user_api_key_request_route"
            ),
        )

        if (
            user_api_key is not None
            and isinstance(user_api_key, str)
            and user_api_key.startswith("sk-")
        ):
            from litellm.proxy.utils import hash_token

            user_api_key = hash_token(user_api_key)

        # increment total LLM requests and spend metric
        self._increment_top_level_request_and_spend_metrics(
            end_user_id=end_user_id,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            model=model,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_id=user_id,
            response_cost=response_cost,
            enum_values=enum_values,
        )

        # input, output, total token metrics
        self._increment_token_metrics(
            # why type ignore below?
            # 1. We just checked if isinstance(standard_logging_payload, dict). Pyright complains.
            # 2. Pyright does not allow us to run isinstance(standard_logging_payload, StandardLoggingPayload) <- this would be ideal
            standard_logging_payload=standard_logging_payload,  # type: ignore
            end_user_id=end_user_id,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            model=model,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_id=user_id,
            enum_values=enum_values,
        )

        # remaining budget metrics
        await self._increment_remaining_budget_metrics(
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            litellm_params=litellm_params,
            response_cost=response_cost,
        )

        # set proxy virtual key rpm/tpm metrics
        self._set_virtual_key_rate_limit_metrics(
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            kwargs=kwargs,
            metadata=_metadata,
        )

        # set latency metrics
        self._set_latency_metrics(
            kwargs=kwargs,
            model=model,
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            # why type ignore below?
            # 1. We just checked if isinstance(standard_logging_payload, dict). Pyright complains.
            # 2. Pyright does not allow us to run isinstance(standard_logging_payload, StandardLoggingPayload) <- this would be ideal
            enum_values=enum_values,
        )

        # set x-ratelimit headers
        self.set_llm_deployment_success_metrics(
            kwargs, start_time, end_time, enum_values, output_tokens
        )

        if (
            standard_logging_payload["stream"] is True
        ):  # log successful streaming requests from logging event hook.
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_proxy_total_requests_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_proxy_total_requests_metric.labels(**_labels).inc()

    def _increment_token_metrics(
        self,
        standard_logging_payload: StandardLoggingPayload,
        end_user_id: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        model: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_id: Optional[str],
        enum_values: UserAPIKeyLabelValues,
    ):
        verbose_logger.debug("prometheus Logging - Enters token metrics function")
        # token metrics

        if standard_logging_payload is not None and isinstance(
            standard_logging_payload, dict
        ):
            _tags = standard_logging_payload["request_tags"]

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_proxy_total_requests_metric"
            ),
            enum_values=enum_values,
        )

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_total_tokens_metric"
            ),
            enum_values=enum_values,
        )
        self.litellm_tokens_metric.labels(**_labels).inc(
            standard_logging_payload["total_tokens"]
        )

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_input_tokens_metric"
            ),
            enum_values=enum_values,
        )
        self.litellm_input_tokens_metric.labels(**_labels).inc(
            standard_logging_payload["prompt_tokens"]
        )

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_output_tokens_metric"
            ),
            enum_values=enum_values,
        )

        self.litellm_output_tokens_metric.labels(**_labels).inc(
            standard_logging_payload["completion_tokens"]
        )

    async def _increment_remaining_budget_metrics(
        self,
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        litellm_params: dict,
        response_cost: float,
    ):
        _team_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_team_spend", None
        )
        _team_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_team_max_budget", None
        )

        _api_key_spend = litellm_params.get("metadata", {}).get(
            "user_api_key_spend", None
        )
        _api_key_max_budget = litellm_params.get("metadata", {}).get(
            "user_api_key_max_budget", None
        )
        await self._set_api_key_budget_metrics_after_api_request(
            user_api_key=user_api_key,
            user_api_key_alias=user_api_key_alias,
            response_cost=response_cost,
            key_max_budget=_api_key_max_budget,
            key_spend=_api_key_spend,
        )

        await self._set_team_budget_metrics_after_api_request(
            user_api_team=user_api_team,
            user_api_team_alias=user_api_team_alias,
            team_spend=_team_spend,
            team_max_budget=_team_max_budget,
            response_cost=response_cost,
        )

    def _increment_top_level_request_and_spend_metrics(
        self,
        end_user_id: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        model: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        user_id: Optional[str],
        response_cost: float,
        enum_values: UserAPIKeyLabelValues,
    ):
        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_requests_metric"
            ),
            enum_values=enum_values,
        )

        self.litellm_requests_metric.labels(**_labels).inc()

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_proxy_total_requests_metric"
            ),
            enum_values=enum_values,
        )

        self.litellm_spend_metric.labels(
            end_user_id,
            user_api_key,
            user_api_key_alias,
            model,
            user_api_team,
            user_api_team_alias,
            user_id,
        ).inc(response_cost)

    def _set_virtual_key_rate_limit_metrics(
        self,
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        kwargs: dict,
        metadata: dict,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        # Set remaining rpm/tpm for API Key + model
        # see parallel_request_limiter.py - variables are set there
        model_group = get_model_group_from_litellm_kwargs(kwargs)
        remaining_requests_variable_name = (
            f"litellm-key-remaining-requests-{model_group}"
        )
        remaining_tokens_variable_name = f"litellm-key-remaining-tokens-{model_group}"

        remaining_requests = (
            metadata.get(remaining_requests_variable_name, sys.maxsize) or sys.maxsize
        )
        remaining_tokens = (
            metadata.get(remaining_tokens_variable_name, sys.maxsize) or sys.maxsize
        )

        self.litellm_remaining_api_key_requests_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_requests)

        self.litellm_remaining_api_key_tokens_for_model.labels(
            user_api_key, user_api_key_alias, model_group
        ).set(remaining_tokens)

    def _set_latency_metrics(
        self,
        kwargs: dict,
        model: Optional[str],
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        enum_values: UserAPIKeyLabelValues,
    ):
        # latency metrics
        end_time: datetime = kwargs.get("end_time") or datetime.now()
        start_time: Optional[datetime] = kwargs.get("start_time")
        api_call_start_time = kwargs.get("api_call_start_time", None)
        completion_start_time = kwargs.get("completion_start_time", None)
        time_to_first_token_seconds = self._safe_duration_seconds(
            start_time=api_call_start_time,
            end_time=completion_start_time,
        )
        if (
            time_to_first_token_seconds is not None
            and kwargs.get("stream", False) is True  # only emit for streaming requests
        ):
            self.litellm_llm_api_time_to_first_token_metric.labels(
                model,
                user_api_key,
                user_api_key_alias,
                user_api_team,
                user_api_team_alias,
            ).observe(time_to_first_token_seconds)
        else:
            verbose_logger.debug(
                "Time to first token metric not emitted, stream option in model_parameters is not True"
            )

        api_call_total_time_seconds = self._safe_duration_seconds(
            start_time=api_call_start_time,
            end_time=end_time,
        )
        if api_call_total_time_seconds is not None:
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_llm_api_latency_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_llm_api_latency_metric.labels(**_labels).observe(
                api_call_total_time_seconds
            )

        # total request latency
        total_time_seconds = self._safe_duration_seconds(
            start_time=start_time,
            end_time=end_time,
        )
        if total_time_seconds is not None:
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_request_total_latency_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_request_total_latency_metric.labels(**_labels).observe(
                total_time_seconds
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        from litellm.types.utils import StandardLoggingPayload

        verbose_logger.debug(
            f"prometheus Logging - Enters failure logging function for kwargs {kwargs}"
        )

        # unpack kwargs
        model = kwargs.get("model", "")
        standard_logging_payload: StandardLoggingPayload = kwargs.get(
            "standard_logging_object", {}
        )
        litellm_params = kwargs.get("litellm_params", {}) or {}
        end_user_id = get_end_user_id_for_cost_tracking(
            litellm_params, service_type="prometheus"
        )
        user_id = standard_logging_payload["metadata"]["user_api_key_user_id"]
        user_api_key = standard_logging_payload["metadata"]["user_api_key_hash"]
        user_api_key_alias = standard_logging_payload["metadata"]["user_api_key_alias"]
        user_api_team = standard_logging_payload["metadata"]["user_api_key_team_id"]
        user_api_team_alias = standard_logging_payload["metadata"][
            "user_api_key_team_alias"
        ]
        kwargs.get("exception", None)

        try:
            self.litellm_llm_api_failed_requests_metric.labels(
                end_user_id,
                user_api_key,
                user_api_key_alias,
                model,
                user_api_team,
                user_api_team_alias,
                user_id,
            ).inc()
            self.set_llm_deployment_failure_metrics(kwargs)
        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        """
        Track client side failures

        Proxy level tracking - failed client side requests

        labelnames=[
                    "end_user",
                    "hashed_api_key",
                    "api_key_alias",
                    REQUESTED_MODEL,
                    "team",
                    "team_alias",
                ] + EXCEPTION_LABELS,
        """
        try:
            _tags = cast(List[str], request_data.get("tags") or [])
            enum_values = UserAPIKeyLabelValues(
                end_user=user_api_key_dict.end_user_id,
                user=user_api_key_dict.user_id,
                user_email=user_api_key_dict.user_email,
                hashed_api_key=user_api_key_dict.api_key,
                api_key_alias=user_api_key_dict.key_alias,
                team=user_api_key_dict.team_id,
                team_alias=user_api_key_dict.team_alias,
                requested_model=request_data.get("model", ""),
                status_code=str(getattr(original_exception, "status_code", None)),
                exception_status=str(getattr(original_exception, "status_code", None)),
                exception_class=self._get_exception_class_name(original_exception),
                tags=_tags,
                route=user_api_key_dict.request_route,
            )
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_proxy_failed_requests_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_proxy_failed_requests_metric.labels(**_labels).inc()

            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_proxy_total_requests_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_proxy_total_requests_metric.labels(**_labels).inc()

        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Proxy level tracking - triggered when the proxy responds with a success response to the client
        """
        try:
            enum_values = UserAPIKeyLabelValues(
                end_user=user_api_key_dict.end_user_id,
                hashed_api_key=user_api_key_dict.api_key,
                api_key_alias=user_api_key_dict.key_alias,
                requested_model=data.get("model", ""),
                team=user_api_key_dict.team_id,
                team_alias=user_api_key_dict.team_alias,
                user=user_api_key_dict.user_id,
                user_email=user_api_key_dict.user_email,
                status_code="200",
                route=user_api_key_dict.request_route,
            )
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_proxy_total_requests_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_proxy_total_requests_metric.labels(**_labels).inc()

        except Exception as e:
            verbose_logger.exception(
                "prometheus Layer Error(): Exception occured - {}".format(str(e))
            )
            pass

    def set_llm_deployment_failure_metrics(self, request_kwargs: dict):
        """
        Sets Failure metrics when an LLM API call fails

        - mark the deployment as partial outage
        - increment deployment failure responses metric
        - increment deployment total requests metric

        Args:
            request_kwargs: dict

        """
        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            standard_logging_payload: StandardLoggingPayload = request_kwargs.get(
                "standard_logging_object", {}
            )
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            litellm_model_name = request_kwargs.get("model", None)
            model_group = standard_logging_payload.get("model_group", None)
            api_base = standard_logging_payload.get("api_base", None)
            model_id = standard_logging_payload.get("model_id", None)
            exception = request_kwargs.get("exception", None)

            llm_provider = _litellm_params.get("custom_llm_provider", None)

            # Create enum_values for the label factory (always create for use in different metrics)
            enum_values = UserAPIKeyLabelValues(
                litellm_model_name=litellm_model_name,
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider,
                exception_status=(
                    str(getattr(exception, "status_code", None)) if exception else None
                ),
                exception_class=(
                    self._get_exception_class_name(exception) if exception else None
                ),
                requested_model=model_group,
                hashed_api_key=standard_logging_payload["metadata"][
                    "user_api_key_hash"
                ],
                api_key_alias=standard_logging_payload["metadata"][
                    "user_api_key_alias"
                ],
                team=standard_logging_payload["metadata"]["user_api_key_team_id"],
                team_alias=standard_logging_payload["metadata"][
                    "user_api_key_team_alias"
                ],
            )

            """
            log these labels
            ["litellm_model_name", "model_id", "api_base", "api_provider"]
            """
            self.set_deployment_partial_outage(
                litellm_model_name=litellm_model_name or "",
                model_id=model_id,
                api_base=api_base,
                api_provider=llm_provider or "",
            )
            if exception is not None:

                _labels = prometheus_label_factory(
                    supported_enum_labels=self.get_labels_for_metric(
                        metric_name="litellm_deployment_failure_responses"
                    ),
                    enum_values=enum_values,
                )
                self.litellm_deployment_failure_responses.labels(**_labels).inc()

                # tag based tracking
                if standard_logging_payload is not None and isinstance(
                    standard_logging_payload, dict
                ):
                    _tags = standard_logging_payload["request_tags"]
                    for tag in _tags:
                        self.litellm_deployment_failure_by_tag_responses.labels(
                            **{
                                UserAPIKeyLabelNames.REQUESTED_MODEL.value: model_group,
                                UserAPIKeyLabelNames.TAG.value: tag,
                                UserAPIKeyLabelNames.v2_LITELLM_MODEL_NAME.value: litellm_model_name,
                                UserAPIKeyLabelNames.MODEL_ID.value: model_id,
                                UserAPIKeyLabelNames.API_BASE.value: api_base,
                                UserAPIKeyLabelNames.API_PROVIDER.value: llm_provider,
                                UserAPIKeyLabelNames.EXCEPTION_CLASS.value: exception.__class__.__name__,
                                UserAPIKeyLabelNames.EXCEPTION_STATUS.value: str(
                                    getattr(exception, "status_code", None)
                                ),
                            }
                        ).inc()

            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_deployment_total_requests"
                ),
                enum_values=enum_values,
            )
            self.litellm_deployment_total_requests.labels(**_labels).inc()

            pass
        except Exception as e:
            verbose_logger.debug(
                "Prometheus Error: set_llm_deployment_failure_metrics. Exception occured - {}".format(
                    str(e)
                )
            )

    def set_llm_deployment_success_metrics(
        self,
        request_kwargs: dict,
        start_time,
        end_time,
        enum_values: UserAPIKeyLabelValues,
        output_tokens: float = 1.0,
    ):

        try:
            verbose_logger.debug("setting remaining tokens requests metric")
            standard_logging_payload: Optional[StandardLoggingPayload] = (
                request_kwargs.get("standard_logging_object")
            )

            if standard_logging_payload is None:
                return

            api_base = standard_logging_payload["api_base"]
            _litellm_params = request_kwargs.get("litellm_params", {}) or {}
            _metadata = _litellm_params.get("metadata", {})
            litellm_model_name = request_kwargs.get("model", None)
            llm_provider = _litellm_params.get("custom_llm_provider", None)
            _model_info = _metadata.get("model_info") or {}
            model_id = _model_info.get("id", None)

            remaining_requests: Optional[int] = None
            remaining_tokens: Optional[int] = None
            if additional_headers := standard_logging_payload["hidden_params"][
                "additional_headers"
            ]:
                # OpenAI / OpenAI Compatible headers
                remaining_requests = additional_headers.get(
                    "x_ratelimit_remaining_requests", None
                )
                remaining_tokens = additional_headers.get(
                    "x_ratelimit_remaining_tokens", None
                )

            if litellm_overhead_time_ms := standard_logging_payload[
                "hidden_params"
            ].get("litellm_overhead_time_ms"):
                _labels = prometheus_label_factory(
                    supported_enum_labels=self.get_labels_for_metric(
                        metric_name="litellm_overhead_latency_metric"
                    ),
                    enum_values=enum_values,
                )
                self.litellm_overhead_latency_metric.labels(**_labels).observe(
                    litellm_overhead_time_ms / 1000
                )  # set as seconds

            if remaining_requests:
                """
                "model_group",
                "api_provider",
                "api_base",
                "litellm_model_name"
                """
                _labels = prometheus_label_factory(
                    supported_enum_labels=self.get_labels_for_metric(
                        metric_name="litellm_remaining_requests_metric"
                    ),
                    enum_values=enum_values,
                )
                self.litellm_remaining_requests_metric.labels(**_labels).set(
                    remaining_requests
                )

            if remaining_tokens:
                _labels = prometheus_label_factory(
                    supported_enum_labels=self.get_labels_for_metric(
                        metric_name="litellm_remaining_tokens_metric"
                    ),
                    enum_values=enum_values,
                )
                self.litellm_remaining_tokens_metric.labels(**_labels).set(
                    remaining_tokens
                )

            """
            log these labels
            ["litellm_model_name", "requested_model", model_id", "api_base", "api_provider"]
            """
            self.set_deployment_healthy(
                litellm_model_name=litellm_model_name or "",
                model_id=model_id or "",
                api_base=api_base or "",
                api_provider=llm_provider or "",
            )

            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_deployment_success_responses"
                ),
                enum_values=enum_values,
            )
            self.litellm_deployment_success_responses.labels(**_labels).inc()

            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_deployment_total_requests"
                ),
                enum_values=enum_values,
            )
            self.litellm_deployment_total_requests.labels(**_labels).inc()

            # Track deployment Latency
            response_ms: timedelta = end_time - start_time
            time_to_first_token_response_time: Optional[timedelta] = None

            if (
                request_kwargs.get("stream", None) is not None
                and request_kwargs["stream"] is True
            ):
                # only log ttft for streaming request
                time_to_first_token_response_time = (
                    request_kwargs.get("completion_start_time", end_time) - start_time
                )

            # use the metric that is not None
            # if streaming - use time_to_first_token_response
            # if not streaming - use response_ms
            _latency: timedelta = time_to_first_token_response_time or response_ms
            _latency_seconds = _latency.total_seconds()

            # latency per output token
            latency_per_token = None
            if output_tokens is not None and output_tokens > 0:
                latency_per_token = _latency_seconds / output_tokens
                _labels = prometheus_label_factory(
                    supported_enum_labels=self.get_labels_for_metric(
                        metric_name="litellm_deployment_latency_per_output_token"
                    ),
                    enum_values=enum_values,
                )
                self.litellm_deployment_latency_per_output_token.labels(
                    **_labels
                ).observe(latency_per_token)

        except Exception as e:
            verbose_logger.exception(
                "Prometheus Error: set_llm_deployment_success_metrics. Exception occured - {}".format(
                    str(e)
                )
            )
            return

    @staticmethod
    def _get_exception_class_name(exception: Exception) -> str:
        exception_class_name = ""
        if hasattr(exception, "llm_provider"):
            exception_class_name = getattr(exception, "llm_provider") or ""

        # pretty print the provider name on prometheus
        # eg. `openai` -> `Openai.`
        if len(exception_class_name) >= 1:
            exception_class_name = (
                exception_class_name[0].upper() + exception_class_name[1:] + "."
            )

        exception_class_name += exception.__class__.__name__
        return exception_class_name

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        """

        Logs a successful LLM fallback event on prometheus

        """
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingMetadata,
            StandardLoggingPayloadSetup,
        )

        verbose_logger.debug(
            "Prometheus: log_success_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _metadata = kwargs.get("metadata", {})
        standard_metadata: StandardLoggingMetadata = (
            StandardLoggingPayloadSetup.get_standard_logging_metadata(
                metadata=_metadata
            )
        )
        _new_model = kwargs.get("model")
        _tags = cast(List[str], kwargs.get("tags") or [])

        enum_values = UserAPIKeyLabelValues(
            requested_model=original_model_group,
            fallback_model=_new_model,
            hashed_api_key=standard_metadata["user_api_key_hash"],
            api_key_alias=standard_metadata["user_api_key_alias"],
            team=standard_metadata["user_api_key_team_id"],
            team_alias=standard_metadata["user_api_key_team_alias"],
            exception_status=str(getattr(original_exception, "status_code", None)),
            exception_class=self._get_exception_class_name(original_exception),
            tags=_tags,
        )
        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_deployment_successful_fallbacks"
            ),
            enum_values=enum_values,
        )
        self.litellm_deployment_successful_fallbacks.labels(**_labels).inc()

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        """
        Logs a failed LLM fallback event on prometheus
        """
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingMetadata,
            StandardLoggingPayloadSetup,
        )

        verbose_logger.debug(
            "Prometheus: log_failure_fallback_event, original_model_group: %s, kwargs: %s",
            original_model_group,
            kwargs,
        )
        _new_model = kwargs.get("model")
        _metadata = kwargs.get("metadata", {})
        _tags = cast(List[str], kwargs.get("tags") or [])
        standard_metadata: StandardLoggingMetadata = (
            StandardLoggingPayloadSetup.get_standard_logging_metadata(
                metadata=_metadata
            )
        )

        enum_values = UserAPIKeyLabelValues(
            requested_model=original_model_group,
            fallback_model=_new_model,
            hashed_api_key=standard_metadata["user_api_key_hash"],
            api_key_alias=standard_metadata["user_api_key_alias"],
            team=standard_metadata["user_api_key_team_id"],
            team_alias=standard_metadata["user_api_key_team_alias"],
            exception_status=str(getattr(original_exception, "status_code", None)),
            exception_class=self._get_exception_class_name(original_exception),
            tags=_tags,
        )

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_deployment_failed_fallbacks"
            ),
            enum_values=enum_values,
        )
        self.litellm_deployment_failed_fallbacks.labels(**_labels).inc()

    def set_litellm_deployment_state(
        self,
        state: int,
        litellm_model_name: str,
        model_id: Optional[str],
        api_base: Optional[str],
        api_provider: str,
    ):
        self.litellm_deployment_state.labels(
            litellm_model_name, model_id, api_base, api_provider
        ).set(state)

    def set_deployment_healthy(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            0, litellm_model_name, model_id, api_base, api_provider
        )

    def set_deployment_partial_outage(
        self,
        litellm_model_name: str,
        model_id: Optional[str],
        api_base: Optional[str],
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            1, litellm_model_name, model_id, api_base, api_provider
        )

    def set_deployment_complete_outage(
        self,
        litellm_model_name: str,
        model_id: Optional[str],
        api_base: Optional[str],
        api_provider: str,
    ):
        self.set_litellm_deployment_state(
            2, litellm_model_name, model_id, api_base, api_provider
        )

    def increment_deployment_cooled_down(
        self,
        litellm_model_name: str,
        model_id: str,
        api_base: str,
        api_provider: str,
        exception_status: str,
    ):
        """
        increment metric when litellm.Router / load balancing logic places a deployment in cool down
        """
        self.litellm_deployment_cooled_down.labels(
            litellm_model_name, model_id, api_base, api_provider, exception_status
        ).inc()

    def track_provider_remaining_budget(
        self, provider: str, spend: float, budget_limit: float
    ):
        """
        Track provider remaining budget in Prometheus
        """
        self.litellm_provider_remaining_budget_metric.labels(provider).set(
            self._safe_get_remaining_budget(
                max_budget=budget_limit,
                spend=spend,
            )
        )

    def _safe_get_remaining_budget(
        self, max_budget: Optional[float], spend: Optional[float]
    ) -> float:
        if max_budget is None:
            return float("inf")

        if spend is None:
            return max_budget

        return max_budget - spend

    async def _initialize_budget_metrics(
        self,
        data_fetch_function: Callable[..., Awaitable[Tuple[List[Any], Optional[int]]]],
        set_metrics_function: Callable[[List[Any]], Awaitable[None]],
        data_type: Literal["teams", "keys"],
    ):
        """
        Generic method to initialize budget metrics for teams or API keys.

        Args:
            data_fetch_function: Function to fetch data with pagination.
            set_metrics_function: Function to set metrics for the fetched data.
            data_type: String representing the type of data ("teams" or "keys") for logging purposes.
        """
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return

        try:
            page = 1
            page_size = 50
            data, total_count = await data_fetch_function(
                page_size=page_size, page=page
            )

            if total_count is None:
                total_count = len(data)

            # Calculate total pages needed
            total_pages = (total_count + page_size - 1) // page_size

            # Set metrics for first page of data
            await set_metrics_function(data)

            # Get and set metrics for remaining pages
            for page in range(2, total_pages + 1):
                data, _ = await data_fetch_function(page_size=page_size, page=page)
                await set_metrics_function(data)

        except Exception as e:
            verbose_logger.exception(
                f"Error initializing {data_type} budget metrics: {str(e)}"
            )

    async def _initialize_team_budget_metrics(self):
        """
        Initialize team budget metrics by reusing the generic pagination logic.
        """
        from litellm.proxy.management_endpoints.team_endpoints import (
            get_paginated_teams,
        )
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_logger.debug(
                "Prometheus: skipping team metrics initialization, DB not initialized"
            )
            return

        async def fetch_teams(
            page_size: int, page: int
        ) -> Tuple[List[LiteLLM_TeamTable], Optional[int]]:
            teams, total_count = await get_paginated_teams(
                prisma_client=prisma_client, page_size=page_size, page=page
            )
            if total_count is None:
                total_count = len(teams)
            return teams, total_count

        await self._initialize_budget_metrics(
            data_fetch_function=fetch_teams,
            set_metrics_function=self._set_team_list_budget_metrics,
            data_type="teams",
        )

    async def _initialize_api_key_budget_metrics(self):
        """
        Initialize API key budget metrics by reusing the generic pagination logic.
        """
        from typing import Union

        from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _list_key_helper,
        )
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            verbose_logger.debug(
                "Prometheus: skipping key metrics initialization, DB not initialized"
            )
            return

        async def fetch_keys(
            page_size: int, page: int
        ) -> Tuple[List[Union[str, UserAPIKeyAuth]], Optional[int]]:
            key_list_response = await _list_key_helper(
                prisma_client=prisma_client,
                page=page,
                size=page_size,
                user_id=None,
                team_id=None,
                key_alias=None,
                key_hash=None,
                exclude_team_id=UI_SESSION_TOKEN_TEAM_ID,
                return_full_object=True,
                organization_id=None,
            )
            keys = key_list_response.get("keys", [])
            total_count = key_list_response.get("total_count")
            if total_count is None:
                total_count = len(keys)
            return keys, total_count

        await self._initialize_budget_metrics(
            data_fetch_function=fetch_keys,
            set_metrics_function=self._set_key_list_budget_metrics,
            data_type="keys",
        )

    async def initialize_remaining_budget_metrics(self):
        """
        Handler for initializing remaining budget metrics for all teams to avoid metric discrepancies.

        Runs when prometheus logger starts up.

        - If redis cache is available, we use the pod lock manager to acquire a lock and initialize the metrics.
            - Ensures only one pod emits the metrics at a time.
        - If redis cache is not available, we initialize the metrics directly.
        """
        from litellm.constants import PROMETHEUS_EMIT_BUDGET_METRICS_JOB_NAME
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager

        # if using redis, ensure only one pod emits the metrics at a time
        if pod_lock_manager and pod_lock_manager.redis_cache:
            if await pod_lock_manager.acquire_lock(
                cronjob_id=PROMETHEUS_EMIT_BUDGET_METRICS_JOB_NAME
            ):
                try:
                    await self._initialize_remaining_budget_metrics()
                finally:
                    await pod_lock_manager.release_lock(
                        cronjob_id=PROMETHEUS_EMIT_BUDGET_METRICS_JOB_NAME
                    )
        else:
            # if not using redis, initialize the metrics directly
            await self._initialize_remaining_budget_metrics()

    async def _initialize_remaining_budget_metrics(self):
        """
        Helper to initialize remaining budget metrics for all teams and API keys.
        """
        verbose_logger.debug("Emitting key, team budget metrics....")
        await self._initialize_team_budget_metrics()
        await self._initialize_api_key_budget_metrics()

    async def _set_key_list_budget_metrics(
        self, keys: List[Union[str, UserAPIKeyAuth]]
    ):
        """Helper function to set budget metrics for a list of keys"""
        for key in keys:
            if isinstance(key, UserAPIKeyAuth):
                self._set_key_budget_metrics(key)

    async def _set_team_list_budget_metrics(self, teams: List[LiteLLM_TeamTable]):
        """Helper function to set budget metrics for a list of teams"""
        for team in teams:
            self._set_team_budget_metrics(team)

    async def _set_team_budget_metrics_after_api_request(
        self,
        user_api_team: Optional[str],
        user_api_team_alias: Optional[str],
        team_spend: float,
        team_max_budget: float,
        response_cost: float,
    ):
        """
        Set team budget metrics after an LLM API request

        - Assemble a LiteLLM_TeamTable object
            - looks up team info from db if not available in metadata
        - Set team budget metrics
        """
        if user_api_team:
            team_object = await self._assemble_team_object(
                team_id=user_api_team,
                team_alias=user_api_team_alias or "",
                spend=team_spend,
                max_budget=team_max_budget,
                response_cost=response_cost,
            )

            self._set_team_budget_metrics(team_object)

    async def _assemble_team_object(
        self,
        team_id: str,
        team_alias: str,
        spend: Optional[float],
        max_budget: Optional[float],
        response_cost: float,
    ) -> LiteLLM_TeamTable:
        """
        Assemble a LiteLLM_TeamTable object

        for fields not available in metadata, we fetch from db
        Fields not available in metadata:
        - `budget_reset_at`
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        _total_team_spend = (spend or 0) + response_cost
        team_object = LiteLLM_TeamTable(
            team_id=team_id,
            team_alias=team_alias,
            spend=_total_team_spend,
            max_budget=max_budget,
        )
        try:
            team_info = await get_team_object(
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
            )
        except Exception as e:
            verbose_logger.debug(
                f"[Non-Blocking] Prometheus: Error getting team info: {str(e)}"
            )
            return team_object

        if team_info:
            team_object.budget_reset_at = team_info.budget_reset_at

        return team_object

    def _set_team_budget_metrics(
        self,
        team: LiteLLM_TeamTable,
    ):
        """
        Set team budget metrics for a single team

        - Remaining Budget
        - Max Budget
        - Budget Reset At
        """
        enum_values = UserAPIKeyLabelValues(
            team=team.team_id,
            team_alias=team.team_alias or "",
        )

        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_remaining_team_budget_metric"
            ),
            enum_values=enum_values,
        )
        self.litellm_remaining_team_budget_metric.labels(**_labels).set(
            self._safe_get_remaining_budget(
                max_budget=team.max_budget,
                spend=team.spend,
            )
        )

        if team.max_budget is not None:
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_team_max_budget_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_team_max_budget_metric.labels(**_labels).set(team.max_budget)

        if team.budget_reset_at is not None:
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_team_budget_remaining_hours_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_team_budget_remaining_hours_metric.labels(**_labels).set(
                self._get_remaining_hours_for_budget_reset(
                    budget_reset_at=team.budget_reset_at
                )
            )

    def _set_key_budget_metrics(self, user_api_key_dict: UserAPIKeyAuth):
        """
        Set virtual key budget metrics

        - Remaining Budget
        - Max Budget
        - Budget Reset At
        """
        enum_values = UserAPIKeyLabelValues(
            hashed_api_key=user_api_key_dict.token,
            api_key_alias=user_api_key_dict.key_alias or "",
        )
        _labels = prometheus_label_factory(
            supported_enum_labels=self.get_labels_for_metric(
                metric_name="litellm_remaining_api_key_budget_metric"
            ),
            enum_values=enum_values,
        )
        self.litellm_remaining_api_key_budget_metric.labels(**_labels).set(
            self._safe_get_remaining_budget(
                max_budget=user_api_key_dict.max_budget,
                spend=user_api_key_dict.spend,
            )
        )

        if user_api_key_dict.max_budget is not None:
            _labels = prometheus_label_factory(
                supported_enum_labels=self.get_labels_for_metric(
                    metric_name="litellm_api_key_max_budget_metric"
                ),
                enum_values=enum_values,
            )
            self.litellm_api_key_max_budget_metric.labels(**_labels).set(
                user_api_key_dict.max_budget
            )

        if user_api_key_dict.budget_reset_at is not None:
            self.litellm_api_key_budget_remaining_hours_metric.labels(**_labels).set(
                self._get_remaining_hours_for_budget_reset(
                    budget_reset_at=user_api_key_dict.budget_reset_at
                )
            )

    async def _set_api_key_budget_metrics_after_api_request(
        self,
        user_api_key: Optional[str],
        user_api_key_alias: Optional[str],
        response_cost: float,
        key_max_budget: float,
        key_spend: Optional[float],
    ):
        if user_api_key:
            user_api_key_dict = await self._assemble_key_object(
                user_api_key=user_api_key,
                user_api_key_alias=user_api_key_alias or "",
                key_max_budget=key_max_budget,
                key_spend=key_spend,
                response_cost=response_cost,
            )
            self._set_key_budget_metrics(user_api_key_dict)

    async def _assemble_key_object(
        self,
        user_api_key: str,
        user_api_key_alias: str,
        key_max_budget: float,
        key_spend: Optional[float],
        response_cost: float,
    ) -> UserAPIKeyAuth:
        """
        Assemble a UserAPIKeyAuth object
        """
        from litellm.proxy.auth.auth_checks import get_key_object
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        _total_key_spend = (key_spend or 0) + response_cost
        user_api_key_dict = UserAPIKeyAuth(
            token=user_api_key,
            key_alias=user_api_key_alias,
            max_budget=key_max_budget,
            spend=_total_key_spend,
        )
        try:
            if user_api_key_dict.token:
                key_object = await get_key_object(
                    hashed_token=user_api_key_dict.token,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                )
                if key_object:
                    user_api_key_dict.budget_reset_at = key_object.budget_reset_at
        except Exception as e:
            verbose_logger.debug(
                f"[Non-Blocking] Prometheus: Error getting key info: {str(e)}"
            )

        return user_api_key_dict

    def _get_remaining_hours_for_budget_reset(self, budget_reset_at: datetime) -> float:
        """
        Get remaining hours for budget reset
        """
        return (
            budget_reset_at - datetime.now(budget_reset_at.tzinfo)
        ).total_seconds() / 3600

    def _safe_duration_seconds(
        self,
        start_time: Any,
        end_time: Any,
    ) -> Optional[float]:
        """
        Compute the duration in seconds between two objects.

        Returns the duration as a float if both start and end are instances of datetime,
        otherwise returns None.
        """
        if isinstance(start_time, datetime) and isinstance(end_time, datetime):
            return (end_time - start_time).total_seconds()
        return None

    @staticmethod
    def initialize_budget_metrics_cron_job(scheduler: AsyncIOScheduler):
        """
        Initialize budget metrics as a cron job. This job runs every `PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES` minutes.

        It emits the current remaining budget metrics for all Keys and Teams.
        """
        from litellm.constants import PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
        from litellm.integrations.custom_logger import CustomLogger
        from litellm.integrations.prometheus import PrometheusLogger

        prometheus_loggers: List[CustomLogger] = (
            litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=PrometheusLogger
            )
        )
        # we need to get the initialized prometheus logger instance(s) and call logger.initialize_remaining_budget_metrics() on them
        verbose_logger.debug("found %s prometheus loggers", len(prometheus_loggers))
        if len(prometheus_loggers) > 0:
            prometheus_logger = cast(PrometheusLogger, prometheus_loggers[0])
            verbose_logger.debug(
                "Initializing remaining budget metrics as a cron job executing every %s minutes"
                % PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
            )
            scheduler.add_job(
                prometheus_logger.initialize_remaining_budget_metrics,
                "interval",
                minutes=PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES,
            )

    @staticmethod
    def _mount_metrics_endpoint(premium_user: bool):
        """
        Mount the Prometheus metrics endpoint with optional authentication.

        Args:
            premium_user (bool): Whether the user is a premium user
            require_auth (bool, optional): Whether to require authentication for the metrics endpoint.
                                        Defaults to False.
        """
        from prometheus_client import make_asgi_app

        from litellm._logging import verbose_proxy_logger
        from litellm.proxy._types import CommonProxyErrors
        from litellm.proxy.proxy_server import app

        if premium_user is not True:
            verbose_proxy_logger.warning(
                f"Prometheus metrics are only available for premium users. {CommonProxyErrors.not_premium_user.value}"
            )

        # Create metrics ASGI app
        metrics_app = make_asgi_app()

        # Mount the metrics app to the app
        app.mount("/metrics", metrics_app)
        verbose_proxy_logger.debug(
            "Starting Prometheus Metrics on /metrics (no authentication)"
        )


def prometheus_label_factory(
    supported_enum_labels: List[str],
    enum_values: UserAPIKeyLabelValues,
    tag: Optional[str] = None,
) -> dict:
    """
    Returns a dictionary of label + values for prometheus.

    Ensures end_user param is not sent to prometheus if it is not supported.
    """
    # Extract dictionary from Pydantic object
    enum_dict = enum_values.model_dump()

    # Filter supported labels
    filtered_labels = {
        label: value
        for label, value in enum_dict.items()
        if label in supported_enum_labels
    }

    if UserAPIKeyLabelNames.END_USER.value in filtered_labels:
        filtered_labels["end_user"] = get_end_user_id_for_cost_tracking(
            litellm_params={"user_api_key_end_user_id": enum_values.end_user},
            service_type="prometheus",
        )

    if enum_values.custom_metadata_labels is not None:
        for key, value in enum_values.custom_metadata_labels.items():
            if key in supported_enum_labels:
                filtered_labels[key] = value

    for label in supported_enum_labels:
        if label not in filtered_labels:
            filtered_labels[label] = None

    return filtered_labels


def get_custom_labels_from_metadata(metadata: dict) -> Dict[str, str]:
    """
    Get custom labels from metadata
    """
    keys = litellm.custom_prometheus_metadata_labels
    if keys is None or len(keys) == 0:
        return {}

    result: Dict[str, str] = {}

    for key in keys:
        # Split the dot notation key into parts
        original_key = key
        key = key.replace("metadata.", "", 1) if key.startswith("metadata.") else key

        keys_parts = key.split(".")
        # Traverse through the dictionary using the parts
        value = metadata
        for part in keys_parts:
            value = value.get(part, None)  # Get the value, return None if not found
            if value is None:
                break

        if value is not None and isinstance(value, str):
            result[original_key.replace(".", "_")] = value

    return result
