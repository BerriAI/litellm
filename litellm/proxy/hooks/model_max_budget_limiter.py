import json
from typing import Dict, List, Optional, Set

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import Span
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    BudgetConfig,
    GenericBudgetConfigType,
    StandardLoggingPayload,
)

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX = "virtual_key_spend"


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue = []

    async def is_key_within_model_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> bool:
        """
        Check if the user_api_key_dict is within the model budget

        Raises:
            BudgetExceededError: If the user_api_key_dict has exceeded the model budget
        """
        _model_max_budget = user_api_key_dict.model_max_budget
        internal_model_max_budget: GenericBudgetConfigType = {}

        for _model, _budget_info in _model_max_budget.items():
            internal_model_max_budget[_model] = BudgetConfig(**_budget_info)

        verbose_proxy_logger.debug(
            "internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        # check if current model is in internal_model_max_budget
        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(
                f"Model {model} not found in internal_model_max_budget"
            )
            return True

        # check if current model is within budget
        if (
            _current_model_budget_info.max_budget
            and _current_model_budget_info.max_budget > 0
        ):
            _current_spend = await self._get_virtual_key_spend_for_model(
                user_api_key_hash=user_api_key_dict.token,
                model=model,
                key_budget_config=_current_model_budget_info,
            )
            if (
                _current_spend is not None
                and _current_model_budget_info.max_budget is not None
                and _current_spend > _current_model_budget_info.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, exceeded budget for model={model}",
                    current_cost=_current_spend,
                    max_budget=_current_model_budget_info.max_budget,
                )

        return True

    async def _get_virtual_key_spend_for_model(
        self,
        user_api_key_hash: Optional[str],
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        """
        Get the current spend for a virtual key for a model

        Lookup model in this order:
            1. model: directly look up `model`
            2. If 1, does not exist, check if passed as {custom_llm_provider}/model
        """

        # 1. model: directly look up `model`
        virtual_key_model_spend_cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{model}:{key_budget_config.budget_duration}"
        _current_spend = await self.dual_cache.async_get_cache(
            key=virtual_key_model_spend_cache_key,
        )

        if _current_spend is None:
            # 2. If 1, does not exist, check if passed as {custom_llm_provider}/model
            # if "/" in model, remove first part before "/" - eg. openai/o1-preview -> o1-preview
            virtual_key_model_spend_cache_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=virtual_key_model_spend_cache_key,
            )
        return _current_spend

    def _get_request_model_budget_config(
        self, model: str, internal_model_max_budget: GenericBudgetConfigType
    ) -> Optional[BudgetConfig]:
        """
        Get the budget config for the request model

        1. Check if `model` is in `internal_model_max_budget`
        2. If not, check if `model` without custom llm provider is in `internal_model_max_budget`
        """
        return internal_model_max_budget.get(
            model, None
        ) or internal_model_max_budget.get(
            self._get_model_without_custom_llm_provider(model), None
        )

    def _get_model_without_custom_llm_provider(self, model: str) -> str:
        if "/" in model:
            return model.split("/")[-1]
        return model

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,  # type: ignore
    ) -> List[dict]:
        if len(healthy_deployments) == 0:
            return healthy_deployments

        request_metadata: Dict = {}
        if isinstance(request_kwargs, dict):
            if isinstance(request_kwargs.get("metadata"), dict):
                request_metadata = request_kwargs.get("metadata", {})
            elif isinstance(request_kwargs.get("litellm_metadata"), dict):
                request_metadata = request_kwargs.get("litellm_metadata", {})

        user_api_key_model_max_budget = request_metadata.get(
            "user_api_key_model_max_budget"
        )
        virtual_key_hash: Optional[str] = request_metadata.get("user_api_key_hash")

        if (
            not user_api_key_model_max_budget
            or not isinstance(user_api_key_model_max_budget, dict)
            or virtual_key_hash is None
        ):
            return healthy_deployments

        internal_model_max_budget: Dict[str, BudgetConfig] = {}
        for budget_model, budget_info in user_api_key_model_max_budget.items():
            try:
                if isinstance(budget_info, BudgetConfig):
                    internal_model_max_budget[budget_model] = budget_info
                elif isinstance(budget_info, dict):
                    internal_model_max_budget[budget_model] = BudgetConfig(
                        **budget_info
                    )
                else:
                    verbose_proxy_logger.debug(
                        "Unsupported budget info type for model %s: %s",
                        budget_model,
                        type(budget_info),
                    )
            except Exception as e:
                verbose_proxy_logger.debug(
                    "Failed to parse budget config for model %s - %s",
                    budget_model,
                    str(e),
                )

        if len(internal_model_max_budget) == 0:
            return healthy_deployments

        filtered_deployments: List[dict] = []
        first_violation: Optional[Dict[str, float]] = None
        request_model_candidates: List[str] = [model]
        if isinstance(request_kwargs, dict):
            request_model = request_kwargs.get("model")
            if isinstance(request_model, str) and request_model not in request_model_candidates:
                request_model_candidates.append(request_model)

        for deployment in healthy_deployments:
            deployment_params: Dict = deployment.get("litellm_params", {})
            deployment_model_name: Optional[str] = deployment.get("model_name")
            deployment_model: Optional[str] = deployment_params.get("model")

            candidate_models: List[str] = []
            for candidate in request_model_candidates:
                if isinstance(candidate, str):
                    candidate_models.append(candidate)
            if isinstance(deployment_model_name, str):
                candidate_models.append(deployment_model_name)
            if isinstance(deployment_model, str):
                candidate_models.append(deployment_model)

            matched_budget: Optional[BudgetConfig] = None
            resolved_model_name: Optional[str] = None

            for candidate in candidate_models:
                if not isinstance(candidate, str):
                    continue
                budget_config = self._get_request_model_budget_config(
                    model=candidate,
                    internal_model_max_budget=internal_model_max_budget,
                )
                if budget_config is None and candidate != "":
                    sanitized_candidate = self._get_model_without_custom_llm_provider(
                        candidate
                    )
                    budget_config = self._get_request_model_budget_config(
                        model=sanitized_candidate,
                        internal_model_max_budget=internal_model_max_budget,
                    )
                    if budget_config is not None:
                        candidate = sanitized_candidate

                if budget_config is not None:
                    matched_budget = budget_config
                    resolved_model_name = candidate
                    break

            if matched_budget is None or resolved_model_name is None:
                filtered_deployments.append(deployment)
                continue

            if not matched_budget.max_budget or matched_budget.max_budget <= 0:
                filtered_deployments.append(deployment)
                continue

            current_spend = await self._get_virtual_key_spend_for_model(
                user_api_key_hash=virtual_key_hash,
                model=resolved_model_name,
                key_budget_config=matched_budget,
            )

            if (
                current_spend is not None
                and matched_budget.max_budget is not None
                and current_spend >= matched_budget.max_budget
            ):
                verbose_proxy_logger.debug(
                    "Filtered deployment %s for virtual key %s due to budget exceed. Spend=%s, Max Budget=%s",
                    deployment_model,
                    virtual_key_hash,
                    current_spend,
                    matched_budget.max_budget,
                )
                if first_violation is None:
                    first_violation = {
                        "model": resolved_model_name,
                        "current_spend": current_spend,
                        "max_budget": matched_budget.max_budget,
                        "key_alias": request_metadata.get("user_api_key_alias"),
                    }
                continue

            filtered_deployments.append(deployment)

        if len(filtered_deployments) > 0:
            return filtered_deployments

        if first_violation is not None:
            key_alias = first_violation.get("key_alias")
            if key_alias is not None:
                message = (
                    "LiteLLM Virtual Key: {}, key_alias: {} exceeded budget for model={}".format(
                        virtual_key_hash,
                        key_alias,
                        first_violation["model"],
                    )
                )
            else:
                message = "LiteLLM Virtual Key: {} exceeded budget for model={}".format(
                    virtual_key_hash,
                    first_violation["model"],
                )
            raise litellm.BudgetExceededError(
                message=message,
                current_cost=first_violation["current_spend"],
                max_budget=first_violation["max_budget"],
            )

        return filtered_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track spend for virtual key + model in DualCache

        Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object", None
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_payload is required")

        _litellm_params: dict = kwargs.get("litellm_params", {}) or {}
        _metadata: dict = _litellm_params.get("metadata", {}) or {}
        user_api_key_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_model_max_budget", None
        )
        if (
            user_api_key_model_max_budget is None
            or len(user_api_key_model_max_budget) == 0
        ):
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event because user_api_key_model_max_budget is None or empty. `user_api_key_model_max_budget`=%s",
                user_api_key_model_max_budget,
            )
            return
        response_cost: float = float(standard_logging_payload.get("response_cost", 0) or 0)
        standard_logging_metadata = standard_logging_payload.get("metadata") or {}
        virtual_key = standard_logging_metadata.get("user_api_key_hash")
        if virtual_key is None:
            verbose_proxy_logger.debug(
                "Skipping virtual key budget tracking - `user_api_key_hash` missing in standard logging payload metadata."
            )
            return

        proxy_request_model: Optional[str] = None
        proxy_request = kwargs.get("proxy_server_request") or {}
        if isinstance(proxy_request, dict):
            request_body = proxy_request.get("body")
            if isinstance(request_body, dict):
                proxy_request_model = request_body.get("model")

        internal_model_max_budget: Dict[str, BudgetConfig] = {}
        for budget_model, budget_info in user_api_key_model_max_budget.items():
            try:
                if isinstance(budget_info, BudgetConfig):
                    internal_model_max_budget[budget_model] = budget_info
                elif isinstance(budget_info, dict):
                    internal_model_max_budget[budget_model] = BudgetConfig(**budget_info)
                else:
                    verbose_proxy_logger.debug(
                        "Unsupported budget info type for model %s: %s",
                        budget_model,
                        type(budget_info),
                    )
            except Exception as e:
                verbose_proxy_logger.debug(
                    "Failed to parse budget config for model %s - %s",
                    budget_model,
                    str(e),
                )

        if len(internal_model_max_budget) == 0:
            verbose_proxy_logger.debug(
                "Not tracking virtual key spend - no parsable budget configs."
            )
            return

        model_candidates: List[str] = []
        if proxy_request_model:
            model_candidates.append(proxy_request_model)
        response_model = standard_logging_payload.get("model")
        if response_model:
            model_candidates.append(response_model)

        seen_models: Set[str] = set()
        spend_tracked = False

        for candidate in model_candidates:
            if not candidate or candidate in seen_models:
                continue
            seen_models.add(candidate)

            resolved_model_name = candidate
            budget_config = internal_model_max_budget.get(candidate)
            if budget_config is None:
                sanitized_model = self._get_model_without_custom_llm_provider(candidate)
                if sanitized_model != candidate:
                    budget_config = internal_model_max_budget.get(sanitized_model)
                    if budget_config is not None:
                        resolved_model_name = sanitized_model

            if budget_config is None:
                verbose_proxy_logger.debug(
                    "No budget config matched for candidate model %s",
                    candidate,
                )
                continue

            if budget_config.budget_duration is None:
                verbose_proxy_logger.debug(
                    "Budget config for model %s missing `budget_duration`, skipping spend tracking.",
                    resolved_model_name,
                )
                continue

            spend_key = (
                f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{resolved_model_name}:{budget_config.budget_duration}"
            )
            start_time_key = (
                f"virtual_key_budget_start_time:{virtual_key}:{resolved_model_name}"
            )

            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )
            spend_tracked = True

        if not spend_tracked:
            verbose_proxy_logger.debug(
                "Virtual key spend not tracked - no candidate models matched the configured budgets."
            )
            return

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(
                self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str
            ),
        )
