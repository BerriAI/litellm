import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import Span
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.llms.bedrock.common_utils import get_bedrock_base_model
from litellm.proxy._types import Litellm_EntityType, UserAPIKeyAuth
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import BudgetConfig, StandardLoggingPayload

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX: Final = "virtual_key_spend"
END_USER_SPEND_CACHE_KEY_PREFIX: Final = "end_user_model_spend"
USER_SPEND_CACHE_KEY_PREFIX: Final = "user_model_spend"

_SPEND_CACHE_KEY_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.USER: USER_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.END_USER: END_USER_SPEND_CACHE_KEY_PREFIX,
    }
)

_LEGACY_REQUEST_MODEL_SCOPES: Final = frozenset({Litellm_EntityType.KEY, Litellm_EntityType.END_USER})

_PROCESS_STARTED_AT: Final = time.monotonic()

_BUDGET_START_TIME_KEY_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: "virtual_key_budget_start_time",
        Litellm_EntityType.USER: "user_model_budget_start_time",
        Litellm_EntityType.END_USER: "end_user_budget_start_time",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedModelBudget:
    """The `model_max_budget` entry a request resolved to.

    ``budget_model`` is the key as the operator configured it, not the model
    name on the request. Every counter is keyed on it so enforcement, the
    post-call increment and the `/key/info` + `/user/info` usage reads cannot
    disagree about which counter a request belongs to.
    """

    budget_model: str
    budget_config: BudgetConfig


def model_budget_spend_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    budget_model: str,
    budget_duration: str | None,
) -> str:
    """Sole owner of the per-model spend counter key, shared by its writer and all of its readers."""
    return f"{_SPEND_CACHE_KEY_PREFIXES[entity_type]}:{entity_id}:{budget_model}:{budget_duration}"


def _legacy_request_model_spend_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    model: str,
    resolved: ResolvedModelBudget,
) -> str | None:
    """The counter this request was billed to before the budget model owned the key, or None.

    Upgrading proxies carry live counters keyed on the model as REQUESTED
    (`openai/gpt-4`) rather than as configured (`gpt-4`), and those were the
    counters the previous version enforced on. Nothing writes that spelling once
    this version is running, so the pre-upgrade and post-upgrade counters hold
    disjoint halves of one window and adding them is the window's real spend.

    Only the key and end-user scopes ever had one. The user scope is introduced
    by this change, so it has no counter to carry.

    The carry stops one budget window after start-up, because a legacy counter
    belongs to a window that was already open when this process replaced the one
    writing it. Past that point the lookup could only ever miss.
    """
    budget_duration: Final = resolved.budget_config.budget_duration
    if entity_type not in _LEGACY_REQUEST_MODEL_SCOPES or budget_duration is None:
        return None
    if time.monotonic() - _PROCESS_STARTED_AT >= duration_in_seconds(budget_duration):
        return None
    return model_budget_spend_cache_key(
        entity_type=entity_type,
        entity_id=entity_id,
        budget_model=model,
        budget_duration=budget_duration,
    )


def model_budget_start_time_cache_key(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    budget_model: str,
    budget_duration: str | None,
) -> str:
    """Window start for one (entity, budget model) pair.

    Scoped per budget model because an entity may budget two models over
    different periods, and a shared start time lets the shorter period restart
    the longer one's window.
    """
    return f"{_BUDGET_START_TIME_KEY_PREFIXES[entity_type]}:{entity_id}:{budget_model}:{budget_duration}"


def resolve_model_budget(model: str, model_max_budget: Mapping[str, object]) -> ResolvedModelBudget | None:
    """Find the `model_max_budget` entry that governs `model`, or None."""
    for candidate in _budget_model_candidates(model):
        raw_budget_config = model_max_budget.get(candidate)
        if raw_budget_config is None:
            continue
        if (budget_config := _usable_budget_config(raw_budget_config)) is None:
            # An entry that will not validate cannot be keyed, so it cannot be
            # enforced or incremented. Skip to the next candidate rather than
            # raising: raising would abort every other scope's increment and turn
            # a config typo into a 500, and stopping here would let one malformed
            # specific entry disable a perfectly good bare-family budget beside
            # it. The candidate chain already falls through an ABSENT entry, and
            # an unparseable one is indistinguishable from absent to enforcement.
            # `validate_model_max_budget` rejects these on the write path, so
            # reaching here means config.yaml or a direct DB edit.
            verbose_proxy_logger.warning(
                "Ignoring unusable model_max_budget entry for %s; it cannot be enforced or tracked",
                candidate,
            )
            continue
        return ResolvedModelBudget(budget_model=candidate, budget_config=budget_config)
    return None


def _budget_model_candidates(model: str) -> tuple[str, ...]:
    """Names a budget may be configured under for a request on `model`, most specific first.

    Beyond the model as sent, a budget may be keyed on the model without its
    ``{custom_llm_provider}/`` prefix (``gpt-4o`` governs ``openai/gpt-4o``), on
    the Bedrock base model (``anthropic.claude-opus-4-8`` governs the
    cross-region ``us.anthropic.claude-opus-4-8``), or on the bare family name
    that Bedrock id shares with its direct-provider twin (``claude-opus-4-8``).
    """
    return tuple(dict.fromkeys((model, model.split("/")[-1], *_bedrock_candidates(model))))


def _bedrock_candidates(model: str) -> tuple[str, ...]:
    """Bedrock-only candidates, empty unless litellm prices `model` as a Bedrock model.

    Gating on the cost map rather than on a vendor allowlist is what makes
    splitting the leading dotted segment safe: most dotted model ids are not
    Bedrock ids at all (``azure/gpt-4.1``, ``gpt-image-1.5``), and splitting one
    of those would produce a garbage candidate.
    """
    base_model: Final = get_bedrock_base_model(model)
    cost_entry: Final = litellm.model_cost.get(base_model)
    if not isinstance(cost_entry, dict) or not str(cost_entry.get("litellm_provider", "")).startswith("bedrock"):
        return ()
    _, _, without_vendor = base_model.partition(".")
    return (base_model, without_vendor) if without_vendor else (base_model,)


async def build_model_max_budget_usage(
    entity_type: Litellm_EntityType,
    entity_id: str | None,
    model_max_budget: Mapping[str, object] | None,
    cache: DualCache | None,
) -> dict[str, dict[str, object]]:
    """Current-window spend per configured budget model, as `/key/info` and `/user/info` report it.

    `cache` must be the DualCache the limiter writes the counters to; callers
    read it off the limiter rather than re-deriving it, so a scope that is being
    blocked can never report zero usage.
    """
    if cache is None or entity_id is None or not model_max_budget:
        return {}

    budgets: Final = tuple(
        (budget_model, budget_config)
        for budget_model, raw_budget_config in model_max_budget.items()
        for budget_config in (_usable_budget_config(raw_budget_config),)
        if budget_config is not None
    )
    if not budgets:
        return {}
    spend_keys: Final = tuple(
        model_budget_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=budget_model,
            budget_duration=budget_config.budget_duration,
        )
        for budget_model, budget_config in budgets
    )
    batched: Final = await cache.async_batch_get_cache(
        keys=list(spend_keys)  # mutable-ok: async_batch_get_cache annotates keys as list, so one must exist here
    )
    # async_batch_get_cache returns None if it fails internally, and its result is
    # index-aligned with `keys` otherwise. An unusable result reads as a miss,
    # which is what a never-written counter already reads as.
    current_spends: Final = (
        tuple(batched) if isinstance(batched, list) and len(batched) == len(budgets) else (None,) * len(budgets)
    )
    return {
        budget_model: {
            "current_spend": round(_as_spend(current_spend), 4),
            "budget_limit": budget_config.max_budget,
            "time_period": budget_config.budget_duration,
        }
        for (budget_model, budget_config), current_spend in zip(budgets, current_spends, strict=True)
    }


def _usable_budget_config(raw_budget_config: object) -> BudgetConfig | None:
    try:
        budget_config: Final = BudgetConfig.model_validate(raw_budget_config)
        if budget_config.budget_duration is None:
            return None
        duration_in_seconds(budget_config.budget_duration)
    except Exception:  # noqa: BLE001  # a malformed entry must not fail the whole report
        return None
    return budget_config


def _as_spend(current_spend: object) -> float:
    try:
        return float(current_spend or 0.0)  # pyright: ignore[reportArgumentType]  # non-numeric falls to the except
    except (TypeError, ValueError):
        return 0.0


def _resolve_entity_model_budgets(
    model: str,
    entity_budgets: Iterable[tuple[Litellm_EntityType, str | None, object]],
) -> tuple[tuple[Litellm_EntityType, str, ResolvedModelBudget], ...]:
    """Drop the scopes that do not budget `model`, keeping only what can be incremented."""
    return tuple(
        (entity_type, entity_id, resolved)
        for entity_type, entity_id, model_max_budget in entity_budgets
        if entity_id is not None and isinstance(model_max_budget, Mapping) and model_max_budget
        for resolved in (resolve_model_budget(model=model, model_max_budget=model_max_budget),)
        if resolved is not None and resolved.budget_config.budget_duration is not None
    )


class _PROXY_VirtualKeyModelMaxBudgetLimiter(RouterBudgetLimiting):
    """
    Handles budgets for model + virtual key

    Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
    """

    def __init__(self, dual_cache: DualCache):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue = []
        self.deployment_budget_config = None

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
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.KEY,
            entity_id=user_api_key_dict.token,
            model_max_budget=user_api_key_dict.model_max_budget,
            model=model,
            exceeded_message=(
                f"LiteLLM Virtual Key: {user_api_key_dict.token}, key_alias: {user_api_key_dict.key_alias}, "
                f"exceeded budget for model={model}"
            ),
        )

    async def get_fallback_model_within_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> str | None:
        budget_fallbacks: Final[dict[str, list[str]]] = user_api_key_dict.budget_fallbacks or {}
        for fallback_model in budget_fallbacks.get(model, []):
            try:
                await self.is_key_within_model_budget(user_api_key_dict=user_api_key_dict, model=fallback_model)
                return fallback_model
            except litellm.BudgetExceededError:
                continue
        return None

    async def is_user_within_model_budget(
        self,
        user_id: str,
        user_model_max_budget: Mapping[str, object],
        model: str,
    ) -> bool:
        """
        Check if the internal user is within the model budget

        Raises:
            BudgetExceededError: If the user has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.USER,
            entity_id=user_id,
            model_max_budget=user_model_max_budget,
            model=model,
            exceeded_message=f"LiteLLM User: {user_id}, exceeded budget for model={model}",
        )

    async def is_end_user_within_model_budget(
        self,
        end_user_id: str,
        end_user_model_max_budget: Mapping[str, object],
        model: str,
    ) -> bool:
        """
        Check if the end_user is within the model budget

        Raises:
            BudgetExceededError: If the end_user has exceeded the model budget
        """
        return await self._is_entity_within_model_budget(
            entity_type=Litellm_EntityType.END_USER,
            entity_id=end_user_id,
            model_max_budget=end_user_model_max_budget,
            model=model,
            exceeded_message=f"LiteLLM End User: {end_user_id}, exceeded budget for model={model}",
        )

    async def _is_entity_within_model_budget(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        model_max_budget: Mapping[str, object] | None,
        model: str,
        exceeded_message: str,
    ) -> bool:
        if not model_max_budget:
            return True
        resolved: Final = resolve_model_budget(model=model, model_max_budget=model_max_budget)
        if resolved is None:
            verbose_proxy_logger.debug("Model %s not found in %s model_max_budget", model, entity_type.value)
            return True

        max_budget: Final = resolved.budget_config.max_budget
        if max_budget is None or max_budget < 0:
            return True

        current_spend: Final = await self._get_spend_for_model_budget(
            entity_type=entity_type,
            entity_id=entity_id,
            model=model,
            resolved=resolved,
        )
        if current_spend >= max_budget:
            raise litellm.BudgetExceededError(
                message=exceeded_message,
                current_cost=current_spend,
                max_budget=max_budget,
                entity_type=entity_type.value,
                entity_id=entity_id,
            )
        return True

    async def _get_spend_for_model_budget(
        self,
        entity_type: Litellm_EntityType,
        entity_id: str | None,
        model: str,
        resolved: ResolvedModelBudget,
    ) -> float:
        """Spend charged to this budget in the current window, legacy counter included.

        A counter that was never written is zero spend, not unknown spend. The
        distinction only shows up at a zero-dollar cap, where skipping the
        comparison would let the strictest possible limit admit every request.
        """
        spend_key: Final = model_budget_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            budget_model=resolved.budget_model,
            budget_duration=resolved.budget_config.budget_duration,
        )
        legacy_spend_key: Final = _legacy_request_model_spend_cache_key(
            entity_type=entity_type,
            entity_id=entity_id,
            model=model,
            resolved=resolved,
        )
        current_spend: Final = _as_spend(await self._cached_spend(spend_key))
        if legacy_spend_key is None or legacy_spend_key == spend_key:
            return current_spend
        return current_spend + _as_spend(await self._cached_spend(legacy_spend_key))

    async def _cached_spend(self, spend_key: str) -> float | None:
        return await self.dual_cache.async_get_cache(key=spend_key)

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        return healthy_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Track spend for virtual key + model in DualCache

        Example: key=sk-1234567890, model=gpt-4o, max_budget=100, time_period=1d
        """
        verbose_proxy_logger.debug("in RouterBudgetLimiting.async_log_success_event")
        standard_logging_payload: Final[StandardLoggingPayload | None] = kwargs.get("standard_logging_object", None)
        if standard_logging_payload is None:
            verbose_proxy_logger.debug(
                "Skipping _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: standard_logging_payload is None"
            )
            return

        _litellm_params: Final[dict] = kwargs.get("litellm_params", {}) or {}
        _metadata: Final[dict] = _litellm_params.get("metadata", {}) or {}
        payload_metadata: Final = standard_logging_payload.get("metadata") or {}

        # Use model_group (the user-facing model alias, e.g. "gpt-4o") when
        # available.  The enforcement path receives the model name from
        # request_data["model"] which is the model group alias, so the spend
        # tracking cache key must resolve from the same name.  Falling back to
        # the deployment-level "model" field preserves behaviour for non-proxy
        # or non-router deployments where model_group is None.
        model: Final = standard_logging_payload.get("model_group") or standard_logging_payload.get("model")
        if model is None:
            return

        response_cost: Final[float] = standard_logging_payload.get("response_cost", 0)
        entity_budgets: Final = (
            (
                Litellm_EntityType.KEY,
                payload_metadata.get("user_api_key_hash"),
                _metadata.get("user_api_key_model_max_budget"),
            ),
            (
                Litellm_EntityType.USER,
                payload_metadata.get("user_api_key_user_id"),
                _metadata.get("user_api_key_user_model_max_budget"),
            ),
            (
                Litellm_EntityType.END_USER,
                standard_logging_payload.get("end_user") or payload_metadata.get("user_api_key_end_user_id"),
                _metadata.get("user_api_key_end_user_model_max_budget"),
            ),
        )

        resolved_budgets: Final = _resolve_entity_model_budgets(model=model, entity_budgets=entity_budgets)
        if not resolved_budgets:
            verbose_proxy_logger.debug(
                "Not running _PROXY_VirtualKeyModelMaxBudgetLimiter.async_log_success_event: "
                "no key, user or end-user model_max_budget covers model=%s",
                model,
            )
            return

        for entity_type, entity_id, resolved in resolved_budgets:
            await self._increment_spend_for_key(
                budget_config=resolved.budget_config,
                spend_key=model_budget_spend_cache_key(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    budget_model=resolved.budget_model,
                    budget_duration=resolved.budget_config.budget_duration,
                ),
                start_time_key=model_budget_start_time_cache_key(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    budget_model=resolved.budget_model,
                    budget_duration=resolved.budget_config.budget_duration,
                ),
                response_cost=response_cost,
            )

        if self.dual_cache.redis_cache is not None:
            await self._push_in_memory_increments_to_redis()

        verbose_proxy_logger.debug(
            "current state of in memory cache %s",
            json.dumps(self.dual_cache.in_memory_cache.cache_dict, indent=4, default=str),
        )
