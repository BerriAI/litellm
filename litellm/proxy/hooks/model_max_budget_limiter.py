import json
from typing import List, Literal, Optional, Tuple

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
END_USER_SPEND_CACHE_KEY_PREFIX = "end_user_model_spend"
TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX = "team_member_model_spend"

ModelBudgetSpendScope = Literal["key", "team_member", "team"]


async def build_effective_model_max_budget_usage(
    limiter: "_PROXY_VirtualKeyModelMaxBudgetLimiter",
    *,
    api_key_hash: str,
    team_id: Optional[str],
    user_id: Optional[str],
    key_model_max_budget: Optional[dict],
    team_model_max_budget: Optional[dict],
    team_member_model_max_budget: Optional[dict],
) -> dict[str, dict[str, object]]:
    model_names: set[str] = set()
    for budget_map in (key_model_max_budget, team_member_model_max_budget, team_model_max_budget):
        if isinstance(budget_map, dict):
            model_names.update(budget_map.keys())

    if not model_names:
        return {}

    result: dict[str, dict[str, object]] = {}
    for model in sorted(model_names):
        budget_config, scope = limiter._resolve_model_budget_config_for_scope(
            model=model,
            key_model_max_budget=key_model_max_budget if isinstance(key_model_max_budget, dict) else None,
            team_member_model_max_budget=(
                team_member_model_max_budget if isinstance(team_member_model_max_budget, dict) else None
            ),
            team_model_max_budget=team_model_max_budget if isinstance(team_model_max_budget, dict) else None,
        )
        if budget_config is None or budget_config.max_budget is None or budget_config.max_budget <= 0:
            continue

        display_scope = scope or "key"
        if scope == "key":
            current_spend = await limiter._get_virtual_key_spend_for_model(
                user_api_key_hash=api_key_hash,
                model=model,
                key_budget_config=budget_config,
            )
        elif scope in ("team_member", "team") and team_id is not None and user_id is not None:
            current_spend = await limiter._get_team_member_model_spend_for_model(
                team_id=team_id,
                user_id=user_id,
                model=model,
                key_budget_config=budget_config,
            )
        else:
            current_spend = await limiter._get_virtual_key_spend_for_model(
                user_api_key_hash=api_key_hash,
                model=model,
                key_budget_config=budget_config,
            )
            display_scope = "key"

        budget_limit = float(budget_config.max_budget)
        spend_value = float(current_spend or 0.0)
        percent_used = min(round((spend_value / budget_limit) * 100, 1), 999.9) if budget_limit > 0 else 0.0
        result[model] = {
            "current_spend": round(spend_value, 4),
            "budget_limit": budget_limit,
            "time_period": budget_config.budget_duration,
            "scope": display_scope,
            "percent_used": percent_used,
        }
    return result


def resolve_effective_model_max_budget(
    key_model_max_budget: Optional[dict],
    team_model_max_budget: Optional[dict] = None,
    team_member_model_max_budget: Optional[dict] = None,
) -> Optional[dict]:
    if not team_model_max_budget and not team_member_model_max_budget and not key_model_max_budget:
        return None
    merged: dict = {}
    if team_model_max_budget:
        merged.update(team_model_max_budget)
    if team_member_model_max_budget:
        merged.update(team_member_model_max_budget)
    if key_model_max_budget:
        merged.update(key_model_max_budget)
    return merged if merged else None


def _to_internal_model_max_budget(model_max_budget: Optional[dict]) -> GenericBudgetConfigType:
    if not model_max_budget:
        return {}
    return {model: BudgetConfig(**budget_info) for model, budget_info in model_max_budget.items()}


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue = []
        self.deployment_budget_config = None

    def _resolve_model_budget_config_for_scope(
        self,
        model: str,
        *,
        key_model_max_budget: Optional[dict],
        team_member_model_max_budget: Optional[dict],
        team_model_max_budget: Optional[dict],
    ) -> Tuple[Optional[BudgetConfig], Optional[ModelBudgetSpendScope]]:
        for scope, model_max_budget in (
            ("key", key_model_max_budget),
            ("team_member", team_member_model_max_budget),
            ("team", team_model_max_budget),
        ):
            if not model_max_budget:
                continue
            internal_model_max_budget = _to_internal_model_max_budget(model_max_budget)
            budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if budget_config is not None:
                return budget_config, scope
        return None, None

    async def is_key_within_model_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
        model_max_budget: Optional[dict] = None,
        *,
        team_model_max_budget: Optional[dict] = None,
        team_member_model_max_budget: Optional[dict] = None,
        key_model_max_budget: Optional[dict] = None,
    ) -> bool:
        """
        Check if the user_api_key_dict is within the model budget

        Raises:
            BudgetExceededError: If the user_api_key_dict has exceeded the model budget
        """
        use_layered_spend = (
            team_model_max_budget is not None
            or team_member_model_max_budget is not None
            or key_model_max_budget is not None
        )

        if use_layered_spend:
            _key_model_max_budget = (
                key_model_max_budget
                if key_model_max_budget is not None
                else user_api_key_dict.model_max_budget
            )
            budget_config, scope = self._resolve_model_budget_config_for_scope(
                model=model,
                key_model_max_budget=_key_model_max_budget,
                team_member_model_max_budget=team_member_model_max_budget,
                team_model_max_budget=team_model_max_budget,
            )
        else:
            _model_max_budget = (
                model_max_budget if model_max_budget is not None else user_api_key_dict.model_max_budget
            )
            if _model_max_budget is None:
                return True
            internal_model_max_budget = _to_internal_model_max_budget(_model_max_budget)
            verbose_proxy_logger.debug(
                "internal_model_max_budget %s",
                json.dumps(internal_model_max_budget, indent=4, default=str),
            )
            budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            scope = "key"

        if budget_config is None:
            verbose_proxy_logger.debug(f"Model {model} not found in model max budget config")
            return True

        if budget_config.max_budget and budget_config.max_budget > 0:
            if scope == "key":
                current_spend = await self._get_virtual_key_spend_for_model(
                    user_api_key_hash=user_api_key_dict.token,
                    model=model,
                    key_budget_config=budget_config,
                )
            elif (
                scope in ("team_member", "team")
                and user_api_key_dict.team_id is not None
                and user_api_key_dict.user_id is not None
            ):
                current_spend = await self._get_team_member_model_spend_for_model(
                    team_id=user_api_key_dict.team_id,
                    user_id=user_api_key_dict.user_id,
                    model=model,
                    key_budget_config=budget_config,
                )
            else:
                current_spend = await self._get_virtual_key_spend_for_model(
                    user_api_key_hash=user_api_key_dict.token,
                    model=model,
                    key_budget_config=budget_config,
                )
            if (
                current_spend is not None
                and budget_config.max_budget is not None
                and current_spend > budget_config.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, exceeded budget for model={model}",
                    current_cost=current_spend,
                    max_budget=budget_config.max_budget,
                )

        return True

    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: dict,
        model: str,
    ) -> bool:
        """
        Check if the end_user is within the model budget

        Raises:
            BudgetExceededError: If the end_user has exceeded the model budget
        """
        internal_model_max_budget = _to_internal_model_max_budget(end_user_model_max_budget)

        verbose_proxy_logger.debug(
            "end_user internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(f"Model {model} not found in end_user_model_max_budget")
            return True

        if _current_model_budget_info.max_budget and _current_model_budget_info.max_budget > 0:
            _current_spend = await self._get_end_user_spend_for_model(
                end_user_id=end_user_id,
                model=model,
                key_budget_config=_current_model_budget_info,
            )
            if (
                _current_spend is not None
                and _current_model_budget_info.max_budget is not None
                and _current_spend > _current_model_budget_info.max_budget
            ):
                raise litellm.BudgetExceededError(
                    message=f"LiteLLM End User: {end_user_id}, exceeded budget for model={model}",
                    current_cost=_current_spend,
                    max_budget=_current_model_budget_info.max_budget,
                )

        return True

    async def _get_end_user_spend_for_model(
        self,
        end_user_id: str,
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        end_user_model_spend_cache_key = (
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
        )
        _current_spend = await self.dual_cache.async_get_cache(
            key=end_user_model_spend_cache_key,
        )

        if _current_spend is None:
            end_user_model_spend_cache_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=end_user_model_spend_cache_key,
            )
        return _current_spend

    async def _get_team_member_model_spend_for_model(
        self,
        team_id: str,
        user_id: str,
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        team_member_model_spend_cache_key = (
            f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{model}:{key_budget_config.budget_duration}"
        )
        current_spend = await self.dual_cache.async_get_cache(
            key=team_member_model_spend_cache_key,
        )

        if current_spend is None:
            team_member_model_spend_cache_key = f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            current_spend = await self.dual_cache.async_get_cache(
                key=team_member_model_spend_cache_key,
            )
        return current_spend

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

        virtual_key_model_spend_cache_key = (
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{model}:{key_budget_config.budget_duration}"
        )
        _current_spend = await self.dual_cache.async_get_cache(
            key=virtual_key_model_spend_cache_key,
        )

        if _current_spend is None:
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
        return internal_model_max_budget.get(model, None) or internal_model_max_budget.get(
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
        return healthy_deployments

    async def _increment_model_budget_spend(
        self,
        *,
        response_cost: float,
        model: str,
        budget_config: BudgetConfig,
        scope: ModelBudgetSpendScope,
        virtual_key: Optional[str],
        team_id: Optional[str],
        user_id: Optional[str],
        end_user_id: Optional[str],
    ) -> None:
        if not budget_config.budget_duration:
            return

        if scope == "key" and virtual_key is not None:
            spend_key = (
                f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_config.budget_duration}"
            )
            start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )
            return

        if scope in ("team_member", "team") and team_id is not None and user_id is not None:
            spend_key = (
                f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{model}:{budget_config.budget_duration}"
            )
            start_time_key = f"team_member_model_budget_start_time:{team_id}:{user_id}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )
            return

        if scope in ("team_member", "team") and virtual_key is not None:
            spend_key = (
                f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_config.budget_duration}"
            )
            start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )
            return

        if end_user_id is not None:
            spend_key = (
                f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{budget_config.budget_duration}"
            )
            start_time_key = f"end_user_budget_start_time:{end_user_id}"
            await self._increment_spend_for_key(
                budget_config=budget_config,
                spend_key=spend_key,
                start_time_key=start_time_key,
                response_cost=response_cost,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track spend for virtual key + model in DualCache

        Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)
        if standard_logging_payload is None:
            verbose_proxy_logger.debug(
                "Skipping _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: standard_logging_payload is None"
            )
            return

        _litellm_params: dict = kwargs.get("litellm_params", {}) or {}
        _metadata: dict = _litellm_params.get("metadata", {}) or {}
        user_api_key_model_max_budget: Optional[dict] = _metadata.get("user_api_key_model_max_budget", None)
        user_api_key_team_model_max_budget: Optional[dict] = _metadata.get("user_api_key_team_model_max_budget", None)
        user_api_key_team_member_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_team_member_model_max_budget", None
        )
        user_api_key_end_user_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_end_user_model_max_budget", None
        )
        has_layered_budget = any(
            budget is not None and len(budget) > 0
            for budget in (
                user_api_key_model_max_budget,
                user_api_key_team_model_max_budget,
                user_api_key_team_member_model_max_budget,
            )
        )
        if not has_layered_budget and (
            user_api_key_end_user_model_max_budget is None or len(user_api_key_end_user_model_max_budget) == 0
        ):
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event because model max budgets are None or empty."
            )
            return

        response_cost: float = standard_logging_payload.get("response_cost", 0)
        model = standard_logging_payload.get("model_group") or standard_logging_payload.get("model")
        virtual_key = standard_logging_payload.get("metadata", {}).get("user_api_key_hash")
        end_user_id = standard_logging_payload.get("end_user") or standard_logging_payload.get("metadata", {}).get(
            "user_api_key_end_user_id"
        )
        team_id = _metadata.get("user_api_key_team_id")
        user_id = _metadata.get("user_api_key_user_id")

        if model is None:
            return

        if has_layered_budget:
            budget_config, scope = self._resolve_model_budget_config_for_scope(
                model=model,
                key_model_max_budget=user_api_key_model_max_budget,
                team_member_model_max_budget=user_api_key_team_member_model_max_budget,
                team_model_max_budget=user_api_key_team_model_max_budget,
            )
            if budget_config is not None and scope is not None:
                await self._increment_model_budget_spend(
                    response_cost=response_cost,
                    model=model,
                    budget_config=budget_config,
                    scope=scope,
                    virtual_key=virtual_key,
                    team_id=team_id,
                    user_id=user_id,
                    end_user_id=end_user_id,
                )

        if (
            end_user_id is not None
            and user_api_key_end_user_model_max_budget is not None
            and len(user_api_key_end_user_model_max_budget) > 0
        ):
            internal_model_max_budget = _to_internal_model_max_budget(user_api_key_end_user_model_max_budget)
            key_budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if key_budget_config is not None and key_budget_config.budget_duration:
                end_user_spend_key = (
                    f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
                )
                end_user_start_time_key = f"end_user_budget_start_time:{end_user_id}"
                await self._increment_spend_for_key(
                    budget_config=key_budget_config,
                    spend_key=end_user_spend_key,
                    start_time_key=end_user_start_time_key,
                    response_cost=response_cost,
                )

        if self.dual_cache.redis_cache is not None:
            await self._push_in_memory_increments_to_redis()

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str),
        )
