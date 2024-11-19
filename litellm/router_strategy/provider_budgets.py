import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.types.router import (
    LiteLLM_Params,
    ProviderBudgetConfigType,
    ProviderBudgetInfo,
)
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class ProviderSpend(TypedDict, total=False):
    """
    Provider spend data

    {
        "openai": 300.0,
        "anthropic": 100.0
    }
    """

    provider: str
    spend: float


class ProviderBudgetLimiting(CustomLogger):
    def __init__(self, router_cache: DualCache, provider_budget_config: dict):
        self.router_cache = router_cache
        self.provider_budget_config: ProviderBudgetConfigType = provider_budget_config
        verbose_router_logger.debug(
            f"Initalized Provider budget config: {self.provider_budget_config}"
        )

    async def async_get_available_deployments(
        self,
        model_group: str,
        healthy_deployments: List[Dict],
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List]] = None,
        request_kwargs: Optional[Dict] = None,
    ):
        """
        Filter list of healthy deployments based on provider budget
        """
        potential_deployments: List[Dict] = []

        parent_otel_span: Optional[Span] = _get_parent_otel_span_from_kwargs(
            request_kwargs
        )

        for deployment in healthy_deployments:
            provider = self._get_llm_provider_for_deployment(deployment)
            budget_config = self._get_budget_config_for_provider(provider)
            if budget_config is None:
                verbose_router_logger.debug(
                    f"No budget config found for provider {provider}, skipping"
                )
                continue

            budget_limit = budget_config.budget_limit
            current_spend: float = (
                await self.router_cache.async_get_cache(
                    key=f"provider_spend:{provider}:{budget_config.time_period}",
                    parent_otel_span=parent_otel_span,
                )
                or 0.0
            )

            verbose_router_logger.debug(
                f"Current spend for {provider}: {current_spend}, budget limit: {budget_limit}"
            )

            if current_spend >= budget_limit:
                verbose_router_logger.debug(
                    f"Skipping deployment {deployment} for provider {provider} as spend limit exceeded"
                )
                continue

            potential_deployments.append(deployment)
        # randomly pick one deployment from potential_deployments
        if potential_deployments:
            return random.choice(potential_deployments)
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Increment provider spend in DualCache (InMemory + Redis)
        """
        verbose_router_logger.debug(
            f"in ProviderBudgetLimiting.async_log_success_event"
        )
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_payload is required")

        response_cost: float = standard_logging_payload.get("response_cost", 0)

        custom_llm_provider: str = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", None
        )
        if custom_llm_provider is None:
            raise ValueError("custom_llm_provider is required")

        budget_config = self._get_budget_config_for_provider(custom_llm_provider)
        if budget_config is None:
            raise ValueError(
                f"No budget config found for provider {custom_llm_provider}, self.provider_budget_config: {self.provider_budget_config}"
            )

        spend_key = f"provider_spend:{custom_llm_provider}:{budget_config.time_period}"
        ttl_seconds = self.get_ttl_seconds(budget_config.time_period)
        verbose_router_logger.debug(
            f"Incrementing spend for {spend_key} by {response_cost}, ttl: {ttl_seconds}"
        )
        # Increment the spend in Redis and set TTL
        await self.router_cache.async_increment_cache(
            key=spend_key,
            value=response_cost,
            ttl=ttl_seconds,
        )
        verbose_router_logger.debug(
            f"Incremented spend for {spend_key} by {response_cost}, ttl: {ttl_seconds}"
        )

    def _get_budget_config_for_provider(
        self, provider: str
    ) -> Optional[ProviderBudgetInfo]:
        return self.provider_budget_config.get(provider, None)

    def _get_llm_provider_for_deployment(self, deployment: Dict) -> str:
        try:
            _litellm_params: LiteLLM_Params = LiteLLM_Params(
                **deployment["litellm_params"]
            )
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=_litellm_params.model,
                litellm_params=_litellm_params,
            )
        except Exception as e:
            raise e
        return custom_llm_provider

    def _get_unique_custom_llm_providers_in_deployments(
        self, deployments: List[Dict]
    ) -> list:
        """
        Get unique custom LLM providers in deployments
        """
        unique_providers = set()
        for deployment in deployments:
            provider = self._get_llm_provider_for_deployment(deployment)
            unique_providers.add(provider)
        return list(unique_providers)

    def get_ttl_seconds(self, time_period: str) -> int:
        """
        Convert time period (e.g., '1d', '30d') to seconds for Redis TTL
        """
        if time_period.endswith("d"):
            days = int(time_period[:-1])
            return days * 24 * 60 * 60
        raise ValueError(f"Unsupported time period format: {time_period}")

    def get_budget_limit(self, custom_llm_provider: str, time_period: str) -> float:
        """
        Fetch the budget limit for a given provider and time period.
        This can be fetched from a config or database.
        """
        _provider_budget_settings = self.provider_budget_config.get(
            custom_llm_provider, None
        )
        if _provider_budget_settings is None:
            return float("inf")

        verbose_router_logger.debug(
            f"Provider budget settings: {_provider_budget_settings}"
        )
        return _provider_budget_settings.budget_limit
