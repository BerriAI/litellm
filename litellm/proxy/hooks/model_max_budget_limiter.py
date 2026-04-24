import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import Span
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
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
        internal_model_max_budget: GenericBudgetConfigType = {}

        for _model, _budget_info in end_user_model_max_budget.items():
            internal_model_max_budget[_model] = BudgetConfig(**_budget_info)

        verbose_proxy_logger.debug(
            "end_user internal_model_max_budget %s",
            json.dumps(internal_model_max_budget, indent=4, default=str),
        )

        # check if current model is in internal_model_max_budget
        _current_model_budget_info = self._get_request_model_budget_config(
            model=model, internal_model_max_budget=internal_model_max_budget
        )
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(
                f"Model {model} not found in end_user_model_max_budget"
            )
            return True

        # check if current model is within budget
        if (
            _current_model_budget_info.max_budget
            and _current_model_budget_info.max_budget > 0
        ):
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
        """
        Get the current spend for an end-user for a model.

        Lookup order:
            1. Cache keyed by exact `model`
            2. Cache keyed by model without custom_llm_provider prefix
            3. DB fallback (cold-cache guard) — sums LiteLLM_SpendLogs within
               the current budget window so budget limits are enforced even on
               the first request after a cache flush or proxy restart.
        """
        # 1. model: directly look up `model`
        end_user_model_spend_cache_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
        _current_spend = await self.dual_cache.async_get_cache(
            key=end_user_model_spend_cache_key,
        )

        if _current_spend is None:
            # 2. If 1, does not exist, check if passed as {custom_llm_provider}/model
            end_user_model_spend_cache_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{self._get_model_without_custom_llm_provider(model)}:{key_budget_config.budget_duration}"
            _current_spend = await self.dual_cache.async_get_cache(
                key=end_user_model_spend_cache_key,
            )

        if _current_spend is None and key_budget_config.budget_duration:
            # 3. Both cache tiers cold — query DB so we don't accidentally pass
            #    through a request that has already exceeded its budget.
            _current_spend = await self._get_spend_from_db(
                model=model,
                budget_duration=key_budget_config.budget_duration,
                budget_start_time_key=f"end_user_budget_start_time:{end_user_id}",
                entity_filter={"end_user": end_user_id},
            )
            # Seed the canonical cache key so subsequent requests don't hit DB
            # again until async_log_success_event writes the first increment.
            if _current_spend is not None and key_budget_config.budget_duration:
                ttl = duration_in_seconds(key_budget_config.budget_duration)
                await self.dual_cache.async_set_cache(
                    key=end_user_model_spend_cache_key,
                    value=_current_spend,
                    ttl=ttl,
                )

        return _current_spend

    async def _get_virtual_key_spend_for_model(
        self,
        user_api_key_hash: Optional[str],
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        """
        Get the current spend for a virtual key for a model.

        Lookup order:
            1. Cache keyed by exact `model`
            2. Cache keyed by model without custom_llm_provider prefix
            3. DB fallback (cold-cache guard) — sums LiteLLM_SpendLogs within
               the current budget window so budget limits are enforced even on
               the first request after a cache flush or proxy restart.
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

        if (
            _current_spend is None
            and user_api_key_hash is not None
            and key_budget_config.budget_duration
        ):
            # 3. Both cache tiers cold — query DB so we don't accidentally pass
            #    through a request that has already exceeded its budget.
            _current_spend = await self._get_spend_from_db(
                model=model,
                budget_duration=key_budget_config.budget_duration,
                budget_start_time_key=f"virtual_key_budget_start_time:{user_api_key_hash}",
                entity_filter={"api_key": user_api_key_hash},
            )
            # Seed the canonical cache key so subsequent requests don't hit DB
            # again until async_log_success_event writes the first increment.
            if _current_spend is not None and key_budget_config.budget_duration:
                ttl = duration_in_seconds(key_budget_config.budget_duration)
                await self.dual_cache.async_set_cache(
                    key=virtual_key_model_spend_cache_key,
                    value=_current_spend,
                    ttl=ttl,
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

    async def _get_spend_from_db(
        self,
        model: str,
        budget_duration: str,
        budget_start_time_key: str,
        entity_filter: dict,
    ) -> Optional[float]:
        """
        DB fallback when cache is cold.

        Queries LiteLLM_SpendLogs and sums spend for the given entity + model
        within the current budget window. This prevents requests from slipping
        through unenforced on the very first call after a cache flush or proxy
        restart.

        Returns None if prisma_client is unavailable (budget check is skipped
        gracefully — same behaviour as before this fallback was added).
        """
        from litellm.proxy.proxy_server import prisma_client  # circular-import exception

        if prisma_client is None:
            return None

        if not budget_duration:
            return None

        ttl_seconds = duration_in_seconds(budget_duration)
        budget_start = await self.dual_cache.async_get_cache(budget_start_time_key)
        if budget_start is not None:
            window_start = datetime.fromtimestamp(float(budget_start), tz=timezone.utc)
        else:
            window_start = datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)

        model_without_provider = self._get_model_without_custom_llm_provider(model)
        # Match model_group (preferred — what spend logger writes) with a
        # fallback to the raw model field for rows written before model_group
        # was populated. Also strip provider prefix so a budget keyed on
        # "gpt-4o" matches logs written as "openai/gpt-4o".
        model_candidates = list(dict.fromkeys([model, model_without_provider]))
        model_or_clauses = [{"model_group": m} for m in model_candidates] + [
            {"model": m} for m in model_candidates
        ]

        where: dict = {
            **entity_filter,
            "OR": model_or_clauses,
            "startTime": {"gte": window_start},
        }

        rows = await prisma_client.db.litellm_spendlogs.group_by(
            by=["model_group"],
            where=where,
            sum={"spend": True},
        )
        return sum((row.get("_sum") or {}).get("spend") or 0.0 for row in rows)

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,  # type: ignore
    ) -> List[dict]:
        return healthy_deployments

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
            verbose_proxy_logger.debug(
                "Skipping _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: standard_logging_payload is None"
            )
            return

        _litellm_params: dict = kwargs.get("litellm_params", {}) or {}
        _metadata: dict = _litellm_params.get("metadata", {}) or {}
        user_api_key_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_model_max_budget", None
        )
        user_api_key_end_user_model_max_budget: Optional[dict] = _metadata.get(
            "user_api_key_end_user_model_max_budget", None
        )
        if (
            user_api_key_model_max_budget is None
            or len(user_api_key_model_max_budget) == 0
        ) and (
            user_api_key_end_user_model_max_budget is None
            or len(user_api_key_end_user_model_max_budget) == 0
        ):
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event because user_api_key_model_max_budget and user_api_key_end_user_model_max_budget are None or empty."
            )
            return

        response_cost: float = standard_logging_payload.get("response_cost", 0)
        # Use model_group (the user-facing model alias, e.g. "gpt-4o") when
        # available.  The enforcement path (is_key_within_model_budget) receives
        # the model name from request_data["model"] which is the model group
        # alias, so the spend tracking cache key must use the same name.
        # Falling back to the deployment-level "model" field preserves
        # behaviour for non-proxy or non-router deployments where model_group
        # is None.
        model = standard_logging_payload.get(
            "model_group"
        ) or standard_logging_payload.get("model")
        virtual_key = standard_logging_payload.get("metadata", {}).get(
            "user_api_key_hash"
        )
        end_user_id = standard_logging_payload.get(
            "end_user"
        ) or standard_logging_payload.get("metadata", {}).get(
            "user_api_key_end_user_id"
        )

        if model is None:
            return

        if (
            virtual_key is not None
            and user_api_key_model_max_budget is not None
            and len(user_api_key_model_max_budget) > 0
        ):
            internal_model_max_budget: GenericBudgetConfigType = {}
            for _model, _budget_info in user_api_key_model_max_budget.items():
                internal_model_max_budget[_model] = BudgetConfig(**_budget_info)
            key_budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if key_budget_config is not None and key_budget_config.budget_duration:
                virtual_spend_key = f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{key_budget_config.budget_duration}"
                virtual_start_time_key = f"virtual_key_budget_start_time:{virtual_key}"
                await self._increment_spend_for_key(
                    budget_config=key_budget_config,
                    spend_key=virtual_spend_key,
                    start_time_key=virtual_start_time_key,
                    response_cost=response_cost,
                )

        if (
            end_user_id is not None
            and user_api_key_end_user_model_max_budget is not None
            and len(user_api_key_end_user_model_max_budget) > 0
        ):
            internal_model_max_budget: GenericBudgetConfigType = {}
            for _model, _budget_info in user_api_key_end_user_model_max_budget.items():
                internal_model_max_budget[_model] = BudgetConfig(**_budget_info)
            key_budget_config = self._get_request_model_budget_config(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if key_budget_config is not None and key_budget_config.budget_duration:
                end_user_spend_key = f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"
                end_user_start_time_key = f"end_user_budget_start_time:{end_user_id}"
                await self._increment_spend_for_key(
                    budget_config=key_budget_config,
                    spend_key=end_user_spend_key,
                    start_time_key=end_user_start_time_key,
                    response_cost=response_cost,
                )

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(
                self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str
            ),
        )
