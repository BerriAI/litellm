import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, List, Literal, Optional, Tuple

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
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

MODEL_BUDGET_WINDOW_TTL_BUFFER_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class ModelBudgetWindow:
    """A calendar-aligned budget window for a given duration.

    ``window_start`` and ``reset_at`` are the inclusive start and exclusive end
    of the current window in the configured budget-reset timezone (UTC by
    default). ``epoch`` is the integer ``window_start`` timestamp used to key the
    spend counter so it rolls over deterministically at each boundary without a
    first-request anchor or a scheduled reset job.
    """

    window_start: datetime
    reset_at: datetime
    epoch: int


def current_model_budget_window(budget_duration: str) -> ModelBudgetWindow:
    from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time

    reset_at = get_budget_reset_time(budget_duration=budget_duration)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    window_start = reset_at - timedelta(seconds=duration_in_seconds(budget_duration))
    return ModelBudgetWindow(window_start=window_start, reset_at=reset_at, epoch=int(window_start.timestamp()))


def _windowed_cache_key(base_key: str, window_epoch: int) -> str:
    return f"{base_key}:w{window_epoch}"


def _window_ttl_seconds(reset_at: datetime) -> int:
    remaining = int((reset_at - datetime.now(timezone.utc)).total_seconds())
    return max(remaining, 1) + MODEL_BUDGET_WINDOW_TTL_BUFFER_SECONDS


async def _sum_model_window_spend_logs(
    *,
    group_field: str,
    entity_where: dict,
    model: str,
    window_start: datetime,
) -> Optional[float]:
    """Authoritative spend for (entity, model) since ``window_start`` from spend logs.

    Returns the summed spend (including 0.0 when there is no matching spend),
    or None when the DB is unavailable or the query fails, so callers can leave
    the counter cold rather than seed a wrong value.
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.repositories.table_repositories import SpendLogsRepository

    if prisma_client is None:
        return None

    where: dict = {
        **entity_where,
        "startTime": {"gte": window_start},
        "OR": [{"model_group": model}, {"model": model}],
    }
    try:
        response = await SpendLogsRepository(prisma_client).table.group_by(
            by=[group_field],
            where=where,  # type: ignore[arg-type]
            sum={"spend": True},
        )
    except Exception:
        verbose_proxy_logger.exception(
            "model_max_budget reconcile: spend-log aggregation failed for %s model=%s",
            entity_where,
            model,
        )
        return None

    if not response:
        return 0.0
    total = 0.0
    for row in response:
        sum_row = row.get("_sum") if isinstance(row, dict) else getattr(row, "_sum", None)
        spend = sum_row.get("spend") if isinstance(sum_row, dict) else getattr(sum_row, "spend", None)
        total += float(spend or 0.0)
    return total

ModelBudgetSpendScope = Literal["key", "team_member", "team"]
ModelMaxBudgetResolutionSource = Literal["flat_metadata", "user_api_key_auth", "db_fallback", "empty"]
ModelMaxBudgetIncrementSource = Literal["callback", "piggyback"]


def _hash_prefix(value: Optional[str], length: int = 8) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    return value[:length]


def _budget_map_model_count(budget_map: Optional[dict]) -> int:
    if not isinstance(budget_map, dict):
        return 0
    return len(budget_map)


def _budget_map_model_names(
    key_budget: Optional[dict],
    team_budget: Optional[dict],
    member_budget: Optional[dict],
) -> list[str]:
    model_names: set[str] = set()
    for budget_map in (key_budget, team_budget, member_budget):
        if isinstance(budget_map, dict):
            model_names.update(budget_map.keys())
    return sorted(model_names)


def _has_layered_model_budget(
    key_budget: Optional[dict],
    team_budget: Optional[dict],
    member_budget: Optional[dict],
) -> bool:
    return any(budget is not None and len(budget) > 0 for budget in (key_budget, team_budget, member_budget))


def _normalize_budget_map(budget_map: Optional[object]) -> Optional[dict]:
    if not isinstance(budget_map, dict) or len(budget_map) == 0:
        return None
    return budget_map


def _get_metadata_source_from_kwargs(kwargs: dict) -> Literal["litellm_metadata", "metadata", "empty"]:
    litellm_params = kwargs.get("litellm_params", {}) or {}
    litellm_metadata = litellm_params.get("litellm_metadata")
    metadata = litellm_params.get("metadata")
    if isinstance(litellm_metadata, dict) and len(litellm_metadata) > 0:
        return "litellm_metadata"
    if isinstance(metadata, dict) and len(metadata) > 0:
        return "metadata"
    return "empty"


def _extract_user_api_key_auth_from_kwargs(kwargs: dict) -> Optional[UserAPIKeyAuth]:
    litellm_params = kwargs.get("litellm_params", {}) or {}
    for bag_key in ("litellm_metadata", "metadata"):
        bag = litellm_params.get(bag_key)
        if isinstance(bag, dict):
            auth = bag.get("user_api_key_auth")
            if isinstance(auth, UserAPIKeyAuth):
                return auth
    metadata = get_litellm_metadata_from_kwargs(kwargs)
    auth = metadata.get("user_api_key_auth")
    if isinstance(auth, UserAPIKeyAuth):
        return auth
    return None


def _log_model_max_budget_spend_trace(event: str, **fields: object) -> None:
    parts = [f"model_max_budget_spend_trace event={event}"]
    for key, value in fields.items():
        if value is not None:
            parts.append(f"{key}={value}")
    verbose_proxy_logger.info(" ".join(parts))


def _compute_spend_cache_key(
    *,
    scope: ModelBudgetSpendScope,
    model: str,
    budget_duration: str,
    virtual_key: Optional[str],
    team_id: Optional[str],
    user_id: Optional[str],
    end_user_id: Optional[str],
    window_epoch: int,
) -> Optional[str]:
    base = _compute_spend_cache_key_base(
        scope=scope,
        model=model,
        budget_duration=budget_duration,
        virtual_key=virtual_key,
        team_id=team_id,
        user_id=user_id,
        end_user_id=end_user_id,
    )
    if base is None:
        return None
    return _windowed_cache_key(base, window_epoch)


def _compute_spend_cache_key_base(
    *,
    scope: ModelBudgetSpendScope,
    model: str,
    budget_duration: str,
    virtual_key: Optional[str],
    team_id: Optional[str],
    user_id: Optional[str],
    end_user_id: Optional[str],
) -> Optional[str]:
    if scope == "key" and virtual_key is not None:
        return f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}"
    if scope in ("team_member", "team") and team_id is not None and user_id is not None:
        return f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{model}:{budget_duration}"
    if scope in ("team_member", "team") and virtual_key is not None:
        return f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{virtual_key}:{model}:{budget_duration}"
    if end_user_id is not None:
        return f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{budget_duration}"
    return None


@dataclass(frozen=True, slots=True)
class ResolvedModelBudgetMaps:
    key_model_max_budget: Optional[dict]
    team_model_max_budget: Optional[dict]
    team_member_model_max_budget: Optional[dict]
    resolution_source: ModelMaxBudgetResolutionSource


@dataclass(frozen=True, slots=True)
class ModelMaxBudgetIncrementContext:
    litellm_call_id: Optional[str]
    call_type: Optional[str]
    route: Optional[str]
    model: str
    model_group: Optional[str]
    response_cost: float
    virtual_key: Optional[str]
    team_id: Optional[str]
    user_id: Optional[str]
    end_user_id: Optional[str]


def _should_skip_model_max_budget_increment(
    *,
    standard_logging_payload: Optional[StandardLoggingPayload],
    response_cost: float,
    cache_hit: bool,
) -> Optional[str]:
    if standard_logging_payload is None:
        return "no_standard_logging"
    if cache_hit or response_cost == 0:
        return "zero_cost"
    model = standard_logging_payload.get("model_group") or standard_logging_payload.get("model")
    if model is None:
        return "no_model"
    return None


def _extract_increment_context_from_kwargs(
    *,
    kwargs: dict,
    response_cost: float,
    source: ModelMaxBudgetIncrementSource,
) -> Optional[ModelMaxBudgetIncrementContext]:
    standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)
    skip_reason = _should_skip_model_max_budget_increment(
        standard_logging_payload=standard_logging_payload,
        response_cost=response_cost,
        cache_hit=bool(kwargs.get("cache_hit", False)),
    )
    litellm_call_id = kwargs.get("litellm_call_id")
    call_type = kwargs.get("call_type")
    if skip_reason is not None:
        _log_model_max_budget_spend_trace(
            "callback_skipped",
            litellm_call_id=litellm_call_id,
            skip_reason=skip_reason,
            increment_source=source,
            call_type=call_type,
            response_cost=response_cost,
        )
        return None

    assert standard_logging_payload is not None
    model = standard_logging_payload.get("model_group") or standard_logging_payload.get("model")
    assert isinstance(model, str)

    metadata: dict = get_litellm_metadata_from_kwargs(kwargs)
    standard_metadata = standard_logging_payload.get("metadata", {}) or {}
    virtual_key = standard_metadata.get("user_api_key_hash") or metadata.get("user_api_key")
    end_user_id = standard_logging_payload.get("end_user") or standard_metadata.get("user_api_key_end_user_id")
    team_id = metadata.get("user_api_key_team_id") or standard_metadata.get("user_api_key_team_id")
    user_id = metadata.get("user_api_key_user_id") or standard_metadata.get("user_api_key_user_id")
    route = metadata.get("user_api_key_request_route")

    return ModelMaxBudgetIncrementContext(
        litellm_call_id=litellm_call_id if isinstance(litellm_call_id, str) else None,
        call_type=call_type if isinstance(call_type, str) else None,
        route=route if isinstance(route, str) else None,
        model=model,
        model_group=standard_logging_payload.get("model_group"),
        response_cost=response_cost,
        virtual_key=virtual_key if isinstance(virtual_key, str) else None,
        team_id=team_id if isinstance(team_id, str) else None,
        user_id=user_id if isinstance(user_id, str) else None,
        end_user_id=end_user_id if isinstance(end_user_id, str) else None,
    )


async def _load_team_member_model_budget_from_db(
    *,
    effective_team_id: str,
    effective_user_id: str,
    team_table: object,
) -> Optional[dict]:
    from litellm.proxy.auth.auth_checks import (
        get_team_member_default_budget,
        get_team_membership,
    )
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None or user_api_key_cache is None:
        return None

    membership = await get_team_membership(
        user_id=str(effective_user_id),
        team_id=str(effective_team_id),
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    if membership is not None and membership.litellm_budget_table is not None:
        return _normalize_budget_map(membership.litellm_budget_table.model_max_budget)

    team_metadata = getattr(team_table, "metadata", None)
    if not isinstance(team_metadata, dict):
        return None
    default_budget_id = team_metadata.get("team_member_budget_id")
    if not isinstance(default_budget_id, str):
        return None
    default_budget = await get_team_member_default_budget(
        budget_id=default_budget_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    if default_budget is None:
        return None
    return _normalize_budget_map(default_budget.model_max_budget)


async def _load_budget_maps_from_db_for_key_hash(
    *,
    api_key_hash: str,
    team_id: Optional[str],
    user_id: Optional[str],
) -> ResolvedModelBudgetMaps:
    from litellm.proxy.auth.auth_checks import (
        get_key_object,
        get_team_object,
    )
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    empty = ResolvedModelBudgetMaps(None, None, None, "empty")
    if prisma_client is None or user_api_key_cache is None or not api_key_hash:
        return empty

    try:
        key_obj = await get_key_object(
            hashed_token=api_key_hash,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
    except Exception:
        verbose_proxy_logger.debug(
            "model_max_budget db_fallback: key lookup failed for hash prefix %s", _hash_prefix(api_key_hash)
        )
        return empty

    key_model_max_budget = _normalize_budget_map(key_obj.model_max_budget)
    team_model_max_budget: Optional[dict] = None
    team_member_model_max_budget: Optional[dict] = None
    effective_team_id = team_id or key_obj.team_id
    effective_user_id = user_id or key_obj.user_id

    if effective_team_id is not None:
        team_table = await get_team_object(
            team_id=str(effective_team_id),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
        if team_table is not None:
            team_model_max_budget = _normalize_budget_map(team_table.model_max_budget)

        if effective_user_id is not None and team_table is not None:
            team_member_model_max_budget = await _load_team_member_model_budget_from_db(
                effective_team_id=str(effective_team_id),
                effective_user_id=str(effective_user_id),
                team_table=team_table,
            )

    if _has_layered_model_budget(key_model_max_budget, team_model_max_budget, team_member_model_max_budget):
        return ResolvedModelBudgetMaps(
            key_model_max_budget=key_model_max_budget,
            team_model_max_budget=team_model_max_budget,
            team_member_model_max_budget=team_member_model_max_budget,
            resolution_source="db_fallback",
        )
    return empty


async def resolve_budget_maps_for_increment(
    *,
    metadata: dict,
    kwargs: dict,
    virtual_key: Optional[str],
    team_id: Optional[str],
    user_id: Optional[str],
) -> ResolvedModelBudgetMaps:
    key_budget = _normalize_budget_map(metadata.get("user_api_key_model_max_budget"))
    team_budget = _normalize_budget_map(metadata.get("user_api_key_team_model_max_budget"))
    member_budget = _normalize_budget_map(metadata.get("user_api_key_team_member_model_max_budget"))

    if _has_layered_model_budget(key_budget, team_budget, member_budget):
        return ResolvedModelBudgetMaps(
            key_model_max_budget=key_budget,
            team_model_max_budget=team_budget,
            team_member_model_max_budget=member_budget,
            resolution_source="flat_metadata",
        )

    user_api_key_auth = _extract_user_api_key_auth_from_kwargs(kwargs)
    if user_api_key_auth is not None:
        auth_key_budget = _normalize_budget_map(user_api_key_auth.model_max_budget)
        auth_team_budget = _normalize_budget_map(user_api_key_auth.auth_team_model_max_budget)
        auth_member_budget = _normalize_budget_map(user_api_key_auth.auth_team_member_model_max_budget)
        if _has_layered_model_budget(auth_key_budget, auth_team_budget, auth_member_budget):
            return ResolvedModelBudgetMaps(
                key_model_max_budget=auth_key_budget,
                team_model_max_budget=auth_team_budget,
                team_member_model_max_budget=auth_member_budget,
                resolution_source="user_api_key_auth",
            )

    if virtual_key is not None:
        return await _load_budget_maps_from_db_for_key_hash(
            api_key_hash=virtual_key,
            team_id=team_id,
            user_id=user_id,
        )

    return ResolvedModelBudgetMaps(None, None, None, "empty")


def log_key_info_usage_reads(
    *,
    api_key_hash: str,
    usage: dict[str, dict[str, object]],
    limiter: "_PROXY_VirtualKeyModelMaxBudgetLimiter",
    team_id: Optional[str],
    user_id: Optional[str],
) -> None:
    for budget_model, entry in usage.items():
        if not isinstance(entry, dict):
            continue
        scope = entry.get("scope")
        time_period = entry.get("time_period")
        if not isinstance(scope, str) or not isinstance(time_period, str):
            continue
        cache_key = _compute_spend_cache_key(
            scope=scope,  # type: ignore[arg-type]
            model=budget_model,
            budget_duration=time_period,
            virtual_key=api_key_hash,
            team_id=team_id,
            user_id=user_id,
            end_user_id=None,
            window_epoch=current_model_budget_window(time_period).epoch,
        )
        _log_model_max_budget_spend_trace(
            "key_info_usage_read",
            api_key_hash=_hash_prefix(api_key_hash),
            budget_model=budget_model,
            scope=scope,
            current_spend=entry.get("current_spend"),
            budget_limit=entry.get("budget_limit"),
            cache_key_used=cache_key,
        )


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
        budget_model, budget_config, scope = limiter._resolve_model_budget_config_for_scope(
            model=model,
            key_model_max_budget=key_model_max_budget if isinstance(key_model_max_budget, dict) else None,
            team_member_model_max_budget=(
                team_member_model_max_budget if isinstance(team_member_model_max_budget, dict) else None
            ),
            team_model_max_budget=team_model_max_budget if isinstance(team_model_max_budget, dict) else None,
        )
        if budget_config is None or budget_config.max_budget is None or budget_config.max_budget <= 0:
            continue

        cache_model = budget_model or model
        display_scope = scope or "key"
        if scope == "key":
            current_spend = await limiter._get_virtual_key_spend_for_model(
                user_api_key_hash=api_key_hash,
                model=cache_model,
                key_budget_config=budget_config,
            )
        elif scope in ("team_member", "team") and team_id is not None and user_id is not None:
            current_spend = await limiter._get_team_member_model_spend_for_model(
                team_id=team_id,
                user_id=user_id,
                model=cache_model,
                key_budget_config=budget_config,
            )
        else:
            current_spend = await limiter._get_virtual_key_spend_for_model(
                user_api_key_hash=api_key_hash,
                model=cache_model,
                key_budget_config=budget_config,
            )
            display_scope = "key"

        budget_limit = float(budget_config.max_budget)
        spend_value = float(current_spend or 0.0)
        percent_used = min(round((spend_value / budget_limit) * 100, 1), 999.9) if budget_limit > 0 else 0.0
        entry: dict[str, object] = {
            "current_spend": round(spend_value, 4),
            "budget_limit": budget_limit,
            "time_period": budget_config.budget_duration,
            "scope": display_scope,
            "percent_used": percent_used,
        }
        if budget_config.budget_duration:
            window = current_model_budget_window(budget_config.budget_duration)
            entry["window_start"] = window.window_start.isoformat()
            entry["reset_at"] = window.reset_at.isoformat()
        result[model] = entry
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
    ) -> Tuple[Optional[str], Optional[BudgetConfig], Optional[ModelBudgetSpendScope]]:
        for scope, model_max_budget in (
            ("key", key_model_max_budget),
            ("team_member", team_member_model_max_budget),
            ("team", team_model_max_budget),
        ):
            if not model_max_budget:
                continue
            internal_model_max_budget = _to_internal_model_max_budget(model_max_budget)
            budget_model = self._match_budget_model_key(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            if budget_model is not None:
                return budget_model, internal_model_max_budget[budget_model], scope
        return None, None, None

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
                key_model_max_budget if key_model_max_budget is not None else user_api_key_dict.model_max_budget
            )
            budget_model, budget_config, scope = self._resolve_model_budget_config_for_scope(
                model=model,
                key_model_max_budget=_key_model_max_budget,
                team_member_model_max_budget=team_member_model_max_budget,
                team_model_max_budget=team_model_max_budget,
            )
        else:
            _model_max_budget = model_max_budget if model_max_budget is not None else user_api_key_dict.model_max_budget
            if _model_max_budget is None:
                return True
            internal_model_max_budget = _to_internal_model_max_budget(_model_max_budget)
            verbose_proxy_logger.debug(
                "internal_model_max_budget %s",
                json.dumps(internal_model_max_budget, indent=4, default=str),
            )
            budget_model = self._match_budget_model_key(
                model=model, internal_model_max_budget=internal_model_max_budget
            )
            budget_config = internal_model_max_budget[budget_model] if budget_model is not None else None
            scope = "key"

        if budget_config is None:
            verbose_proxy_logger.debug(f"Model {model} not found in model max budget config")
            return True

        cache_model = budget_model or model
        if budget_config.max_budget and budget_config.max_budget > 0:
            if scope == "key":
                current_spend = await self._get_virtual_key_spend_for_model(
                    user_api_key_hash=user_api_key_dict.token,
                    model=cache_model,
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
                    model=cache_model,
                    key_budget_config=budget_config,
                )
            else:
                current_spend = await self._get_virtual_key_spend_for_model(
                    user_api_key_hash=user_api_key_dict.token,
                    model=cache_model,
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

    async def get_fallback_model_within_budget(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        model: str,
    ) -> Optional[str]:
        budget_fallbacks: dict[str, list[str]] = user_api_key_dict.budget_fallbacks or {}
        for fallback_model in budget_fallbacks.get(model, []):
            try:
                await self.is_key_within_model_budget(user_api_key_dict=user_api_key_dict, model=fallback_model)
                return fallback_model
            except litellm.BudgetExceededError:
                continue
        return None

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

        budget_model = self._match_budget_model_key(model=model, internal_model_max_budget=internal_model_max_budget)
        _current_model_budget_info = internal_model_max_budget[budget_model] if budget_model is not None else None
        if _current_model_budget_info is None:
            verbose_proxy_logger.debug(f"Model {model} not found in end_user_model_max_budget")
            return True

        cache_model = budget_model or model
        if _current_model_budget_info.max_budget and _current_model_budget_info.max_budget > 0:
            _current_spend = await self._get_end_user_spend_for_model(
                end_user_id=end_user_id,
                model=cache_model,
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

    async def _windowed_counter_get(self, cache_key: str) -> Optional[float]:
        redis_cache = self.dual_cache.redis_cache
        if redis_cache is not None:
            try:
                val = await redis_cache.async_get_cache(key=cache_key)
                return float(val) if val is not None else None
            except Exception:
                verbose_proxy_logger.debug(
                    "model_max_budget: Redis read failed for %s, falling back to in-memory", cache_key
                )
        val = await self.dual_cache.async_get_cache(key=cache_key)
        return float(val) if val is not None else None

    async def _get_windowed_model_spend(
        self,
        *,
        base_keys: List[str],
        budget_config: BudgetConfig,
        reconcile: Callable[[datetime], Awaitable[Optional[float]]],
    ) -> Optional[float]:
        if not budget_config.budget_duration:
            return None
        window = current_model_budget_window(budget_config.budget_duration)
        for base in base_keys:
            hit = await self._windowed_counter_get(_windowed_cache_key(base, window.epoch))
            if hit is not None:
                return hit

        reconciled = await reconcile(window.window_start)
        if reconciled is None:
            return None
        await self.dual_cache.async_set_cache(
            key=_windowed_cache_key(base_keys[0], window.epoch),
            value=reconciled,
            ttl=_window_ttl_seconds(window.reset_at),
        )
        return reconciled

    async def _get_end_user_spend_for_model(
        self,
        end_user_id: str,
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        stripped = self._get_model_without_custom_llm_provider(model)
        base_keys = [f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{model}:{key_budget_config.budget_duration}"]
        if stripped != model:
            base_keys.append(
                f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{end_user_id}:{stripped}:{key_budget_config.budget_duration}"
            )

        async def reconcile(window_start: datetime) -> Optional[float]:
            return await _sum_model_window_spend_logs(
                group_field="end_user",
                entity_where={"end_user": end_user_id},
                model=model,
                window_start=window_start,
            )

        return await self._get_windowed_model_spend(
            base_keys=base_keys, budget_config=key_budget_config, reconcile=reconcile
        )

    async def _get_team_member_model_spend_for_model(
        self,
        team_id: str,
        user_id: str,
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        stripped = self._get_model_without_custom_llm_provider(model)
        base_keys = [
            f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{model}:{key_budget_config.budget_duration}"
        ]
        if stripped != model:
            base_keys.append(
                f"{TEAM_MEMBER_MODEL_SPEND_CACHE_KEY_PREFIX}:{team_id}:{user_id}:{stripped}:{key_budget_config.budget_duration}"
            )

        async def reconcile(window_start: datetime) -> Optional[float]:
            return await _sum_model_window_spend_logs(
                group_field="team_id",
                entity_where={"team_id": team_id, "user": user_id},
                model=model,
                window_start=window_start,
            )

        return await self._get_windowed_model_spend(
            base_keys=base_keys, budget_config=key_budget_config, reconcile=reconcile
        )

    async def _get_virtual_key_spend_for_model(
        self,
        user_api_key_hash: Optional[str],
        model: str,
        key_budget_config: BudgetConfig,
    ) -> Optional[float]:
        """
        Get the current spend for a virtual key for a model in the current
        calendar-aligned window.

        Reads the window-keyed counter (Redis-first for cross-pod truth). On a
        clean miss the counter is reseeded from spend logs for the current
        window so a cache/Redis loss or redeploy cannot silently zero spend.

        Lookup model in this order:
            1. model: directly look up `model`
            2. If 1 does not exist, check if passed as {custom_llm_provider}/model
        """
        stripped = self._get_model_without_custom_llm_provider(model)
        base_keys = [
            f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{model}:{key_budget_config.budget_duration}"
        ]
        if stripped != model:
            base_keys.append(
                f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{user_api_key_hash}:{stripped}:{key_budget_config.budget_duration}"
            )

        async def reconcile(window_start: datetime) -> Optional[float]:
            if not user_api_key_hash:
                return None
            return await _sum_model_window_spend_logs(
                group_field="api_key",
                entity_where={"api_key": user_api_key_hash},
                model=model,
                window_start=window_start,
            )

        return await self._get_windowed_model_spend(
            base_keys=base_keys, budget_config=key_budget_config, reconcile=reconcile
        )

    def _get_request_model_budget_config(
        self, model: str, internal_model_max_budget: GenericBudgetConfigType
    ) -> Optional[BudgetConfig]:
        budget_model = self._match_budget_model_key(model=model, internal_model_max_budget=internal_model_max_budget)
        if budget_model is None:
            return None
        return internal_model_max_budget[budget_model]

    def _match_budget_model_key(self, model: str, internal_model_max_budget: GenericBudgetConfigType) -> Optional[str]:
        if model in internal_model_max_budget:
            return model

        stripped = self._get_model_without_custom_llm_provider(model)
        if stripped in internal_model_max_budget:
            return stripped

        normalized_tokens = stripped.replace("/", ".").split(".")
        for budget_model in internal_model_max_budget:
            if budget_model in normalized_tokens:
                return budget_model

        return None

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

    async def _increment_windowed_spend(
        self,
        *,
        spend_key: str,
        response_cost: float,
        reset_at: datetime,
        litellm_call_id: Optional[str],
        budget_duration: Optional[str],
    ) -> None:
        current_spend = await self._windowed_counter_get(spend_key)
        _log_model_max_budget_spend_trace(
            "spend_key_read_before",
            litellm_call_id=litellm_call_id,
            spend_key=spend_key,
            current_spend=current_spend,
        )
        await self._increment_spend_in_current_window(
            spend_key=spend_key,
            response_cost=response_cost,
            ttl=_window_ttl_seconds(reset_at),
        )
        _log_model_max_budget_spend_trace(
            "spend_key_write",
            litellm_call_id=litellm_call_id,
            spend_key=spend_key,
            delta=response_cost,
            budget_duration=budget_duration,
        )

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
        litellm_call_id: Optional[str] = None,
    ) -> None:
        if not budget_config.budget_duration:
            _log_model_max_budget_spend_trace(
                "callback_skipped",
                litellm_call_id=litellm_call_id,
                skip_reason="no_budget_duration",
                model=model,
                scope=scope,
            )
            return

        window = current_model_budget_window(budget_config.budget_duration)
        spend_key = _compute_spend_cache_key(
            scope=scope,
            model=model,
            budget_duration=budget_config.budget_duration,
            virtual_key=virtual_key,
            team_id=team_id,
            user_id=user_id,
            end_user_id=end_user_id,
            window_epoch=window.epoch,
        )
        if spend_key is None:
            _log_model_max_budget_spend_trace(
                "callback_skipped",
                litellm_call_id=litellm_call_id,
                skip_reason="no_virtual_key",
                model=model,
                scope=scope,
            )
            return

        _log_model_max_budget_spend_trace(
            "increment_attempt",
            litellm_call_id=litellm_call_id,
            budget_model=model,
            scope=scope,
            response_cost=response_cost,
            spend_key=spend_key,
        )
        await self._increment_windowed_spend(
            spend_key=spend_key,
            response_cost=response_cost,
            reset_at=window.reset_at,
            litellm_call_id=litellm_call_id,
            budget_duration=budget_config.budget_duration,
        )
        _log_model_max_budget_spend_trace(
            "increment_done",
            litellm_call_id=litellm_call_id,
            spend_key=spend_key,
            response_cost=response_cost,
            scope=scope,
            budget_duration=budget_config.budget_duration,
        )

    async def _increment_layered_budget_from_maps(
        self,
        *,
        ctx: ModelMaxBudgetIncrementContext,
        resolved_maps: ResolvedModelBudgetMaps,
        increment_source: ModelMaxBudgetIncrementSource,
    ) -> bool:
        budget_model, budget_config, scope = self._resolve_model_budget_config_for_scope(
            model=ctx.model,
            key_model_max_budget=resolved_maps.key_model_max_budget,
            team_member_model_max_budget=resolved_maps.team_member_model_max_budget,
            team_model_max_budget=resolved_maps.team_model_max_budget,
        )
        if budget_config is None or scope is None:
            _log_model_max_budget_spend_trace(
                "callback_skipped",
                litellm_call_id=ctx.litellm_call_id,
                skip_reason="no_model_match",
                increment_source=increment_source,
                model=ctx.model,
            )
            return False

        await self._increment_model_budget_spend(
            response_cost=ctx.response_cost,
            model=budget_model or ctx.model,
            budget_config=budget_config,
            scope=scope,
            virtual_key=ctx.virtual_key,
            team_id=ctx.team_id,
            user_id=ctx.user_id,
            end_user_id=ctx.end_user_id,
            litellm_call_id=ctx.litellm_call_id,
        )
        return True

    async def _increment_end_user_budget_if_needed(
        self,
        *,
        ctx: ModelMaxBudgetIncrementContext,
        end_user_budget: Optional[dict],
    ) -> bool:
        if ctx.end_user_id is None or end_user_budget is None:
            return False
        internal_model_max_budget = _to_internal_model_max_budget(end_user_budget)
        budget_model = self._match_budget_model_key(
            model=ctx.model, internal_model_max_budget=internal_model_max_budget
        )
        key_budget_config = internal_model_max_budget[budget_model] if budget_model is not None else None
        if key_budget_config is None or not key_budget_config.budget_duration:
            return False

        cache_model = budget_model or ctx.model
        window = current_model_budget_window(key_budget_config.budget_duration)
        end_user_spend_key = _windowed_cache_key(
            f"{END_USER_SPEND_CACHE_KEY_PREFIX}:{ctx.end_user_id}:{cache_model}:{key_budget_config.budget_duration}",
            window.epoch,
        )
        await self._increment_windowed_spend(
            spend_key=end_user_spend_key,
            response_cost=ctx.response_cost,
            reset_at=window.reset_at,
            litellm_call_id=ctx.litellm_call_id,
            budget_duration=key_budget_config.budget_duration,
        )
        return True

    async def increment_model_max_budget_from_success(
        self,
        kwargs: dict,
        *,
        response_cost: float,
        source: ModelMaxBudgetIncrementSource = "callback",
    ) -> None:
        ctx = _extract_increment_context_from_kwargs(kwargs=kwargs, response_cost=response_cost, source=source)
        if ctx is None:
            return

        if ctx.litellm_call_id is not None:
            dedupe_key = f"model_max_budget_incremented:{ctx.litellm_call_id}"
            if await self.dual_cache.async_get_cache(key=dedupe_key) is not None:
                _log_model_max_budget_spend_trace(
                    "callback_skipped",
                    litellm_call_id=ctx.litellm_call_id,
                    skip_reason="already_incremented",
                    increment_source=source,
                )
                return

        _log_model_max_budget_spend_trace(
            "callback_entered",
            litellm_call_id=ctx.litellm_call_id,
            increment_source=source,
            call_type=ctx.call_type,
            has_standard_logging=True,
            has_litellm_params=bool(kwargs.get("litellm_params", {}) or {}),
            route=ctx.route,
            model=ctx.model,
            model_group=ctx.model_group,
            virtual_key=_hash_prefix(ctx.virtual_key),
        )

        metadata = get_litellm_metadata_from_kwargs(kwargs)
        resolved_maps = await resolve_budget_maps_for_increment(
            metadata=metadata,
            kwargs=kwargs,
            virtual_key=ctx.virtual_key,
            team_id=ctx.team_id,
            user_id=ctx.user_id,
        )
        end_user_budget = _normalize_budget_map(metadata.get("user_api_key_end_user_model_max_budget"))
        has_layered_budget = _has_layered_model_budget(
            resolved_maps.key_model_max_budget,
            resolved_maps.team_model_max_budget,
            resolved_maps.team_member_model_max_budget,
        )

        _log_model_max_budget_spend_trace(
            "metadata_resolved",
            litellm_call_id=ctx.litellm_call_id,
            increment_source=source,
            metadata_source=_get_metadata_source_from_kwargs(kwargs),
            resolution_source=resolved_maps.resolution_source,
            has_user_api_key_auth=_extract_user_api_key_auth_from_kwargs(kwargs) is not None,
            key_budget_model_count=_budget_map_model_count(resolved_maps.key_model_max_budget),
            team_budget_model_count=_budget_map_model_count(resolved_maps.team_model_max_budget),
            member_budget_model_count=_budget_map_model_count(resolved_maps.team_member_model_max_budget),
            budget_model_names=_budget_map_model_names(
                resolved_maps.key_model_max_budget,
                resolved_maps.team_model_max_budget,
                resolved_maps.team_member_model_max_budget,
            ),
        )

        if not has_layered_budget and end_user_budget is None:
            _log_model_max_budget_spend_trace(
                "callback_skipped",
                litellm_call_id=ctx.litellm_call_id,
                skip_reason="no_budget_maps",
                increment_source=source,
                resolution_source=resolved_maps.resolution_source,
            )
            return

        did_increment = (
            (
                await self._increment_layered_budget_from_maps(
                    ctx=ctx, resolved_maps=resolved_maps, increment_source=source
                )
            )
            if has_layered_budget
            else False
        )
        did_increment = (
            await self._increment_end_user_budget_if_needed(ctx=ctx, end_user_budget=end_user_budget)
        ) or did_increment

        if did_increment and self.dual_cache.redis_cache is not None:
            await self._push_in_memory_increments_to_redis()
            _log_model_max_budget_spend_trace(
                "redis_push",
                litellm_call_id=ctx.litellm_call_id,
                increment_source=source,
                queue_size=len(self.redis_increment_operation_queue),
            )

        if did_increment and ctx.litellm_call_id is not None:
            await self.dual_cache.async_set_cache(
                key=f"model_max_budget_incremented:{ctx.litellm_call_id}",
                value=True,
                ttl=300,
            )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)
        response_cost: float = (
            standard_logging_payload.get("response_cost", 0) if standard_logging_payload is not None else 0
        )
        await self.increment_model_max_budget_from_success(
            kwargs,
            response_cost=response_cost,
            source="callback",
        )
