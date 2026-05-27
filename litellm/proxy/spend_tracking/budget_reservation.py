from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_utils import get_model_from_request
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router import Router


@dataclass
class _BudgetCounter:
    counter_key: str
    max_budget: float
    fallback_spend: float
    entity_type: str
    entity_id: str
    source_cache_key: Optional[str] = None
    spend_log_entity_id: Optional[str] = None
    window_start: Optional[datetime] = None


class _CounterReservationUnavailable(Exception):
    def __init__(
        self,
        touched_counter: bool = False,
        counter_invalidated: bool = False,
    ) -> None:
        self.touched_counter = touched_counter
        self.counter_invalidated = counter_invalidated
        super().__init__("Counter reservation unavailable")


def get_reserved_counter_keys(budget_reservation: Optional[dict]) -> set:
    if not budget_reservation:
        return set()
    entries = budget_reservation.get("entries") or []
    return {
        entry["counter_key"]
        for entry in entries
        if isinstance(entry, dict) and entry.get("counter_key") is not None
    }


async def reserve_budget_for_request(
    request_body: dict,
    route: str,
    llm_router: Optional[Router],
    valid_token: Optional[UserAPIKeyAuth],
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: Optional[str] = None,
    end_user_object: Optional[Any] = None,
) -> Optional[dict]:
    if valid_token is None or not RouteChecks.is_llm_api_route(route=route):
        return None
    if route in {"/models", "/v1/models", "/utils/token_counter"}:
        return None
    if get_model_from_request(request_body, route) is None:
        return None

    counters = await _get_budget_counters(
        request_body=request_body,
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        end_user_id=end_user_id,
        end_user_object=end_user_object,
    )
    if not counters:
        return None

    current_spend_by_counter_key: Dict[str, float] = {}
    reservation_cost = estimate_request_max_cost(
        request_body=request_body,
        route=route,
        llm_router=llm_router,
    )
    # estimate_request_max_cost still returns None when the model is unknown
    # to the cost map (no token-priced cost fields, e.g. image/audio routes).
    # In that case we fall back to read-time enforcement only.
    if reservation_cost is None or reservation_cost <= 0:
        return None

    applied_entries: List[Dict[str, Any]] = []
    try:
        for counter in counters:
            entry = _counter_to_reservation_entry(
                counter=counter,
                reserved_cost=reservation_cost,
            )
            applied_entries.append(entry)
            try:
                reserved_value = await _reserve_counter(
                    counter=counter,
                    reservation_cost=reservation_cost,
                )
            except _CounterReservationUnavailable as exc:
                if exc.touched_counter and not exc.counter_invalidated:
                    await _release_applied_entries_best_effort(
                        entries=[entry],
                        default_reserved_cost=reservation_cost,
                    )
                applied_entries.remove(entry)
                continue

            if reserved_value is not None:
                current_spend = reserved_value
            else:
                cached_spend = current_spend_by_counter_key.get(counter.counter_key)
                if cached_spend is None:
                    cached_spend = await _get_current_counter_value(counter=counter)
                current_spend = cached_spend + reservation_cost
            if current_spend > counter.max_budget:
                remaining_before_reservation = counter.max_budget - (
                    current_spend - reservation_cost
                )
                if remaining_before_reservation > 1e-12:
                    await _resize_applied_reservation(
                        entries=applied_entries,
                        current_reserved_cost=reservation_cost,
                        new_reserved_cost=remaining_before_reservation,
                    )
                    reservation_cost = remaining_before_reservation
                    continue
                raise litellm.BudgetExceededError(
                    current_cost=current_spend,
                    max_budget=counter.max_budget,
                    message=(
                        "Budget has been exceeded! "
                        f"{counter.entity_type}={counter.entity_id} "
                        f"Current cost: {current_spend}, "
                        f"Max budget: {counter.max_budget}"
                    ),
                )
    except Exception:
        await _release_applied_entries_best_effort(
            entries=applied_entries,
            default_reserved_cost=reservation_cost,
        )
        raise

    if not applied_entries:
        return None

    return {
        "reserved_cost": reservation_cost,
        "entries": applied_entries,
        "finalized": False,
    }


async def reconcile_budget_reservation(
    budget_reservation: Optional[dict],
    actual_cost: Optional[float],
    finalize: bool = True,
) -> None:
    if not budget_reservation or budget_reservation.get("finalized") is True:
        return

    reserved_cost = float(budget_reservation.get("reserved_cost") or 0.0)
    actual = float(actual_cost or 0.0)
    await _set_reserved_entries_actual_cost(
        entries=budget_reservation.get("entries") or [],
        actual_cost=actual,
        default_reserved_cost=reserved_cost,
    )
    if finalize:
        budget_reservation["finalized"] = True


async def release_budget_reservation(budget_reservation: Optional[dict]) -> None:
    await reconcile_budget_reservation(
        budget_reservation=budget_reservation,
        actual_cost=0.0,
    )


async def invalidate_budget_reservation_counters(
    budget_reservation: Optional[dict],
) -> None:
    if budget_reservation is None:
        return

    from litellm.proxy.proxy_server import _invalidate_spend_counter

    for counter_key in get_reserved_counter_keys(budget_reservation=budget_reservation):
        await _invalidate_spend_counter(counter_key=counter_key)


async def _get_budget_counters(
    request_body: dict,
    valid_token: UserAPIKeyAuth,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: Optional[str] = None,
    end_user_object: Optional[Any] = None,
) -> List[_BudgetCounter]:
    counters: List[_BudgetCounter] = []

    if valid_token.token is not None:
        if valid_token.max_budget is not None and valid_token.max_budget > 0:
            counters.append(
                _BudgetCounter(
                    counter_key=f"spend:key:{valid_token.token}",
                    source_cache_key=valid_token.token,
                    max_budget=float(valid_token.max_budget),
                    fallback_spend=float(valid_token.spend or 0.0),
                    entity_type="Key",
                    entity_id=valid_token.token,
                )
            )
        counters.extend(
            _get_budget_limit_counters(
                entity_prefix=f"spend:key:{valid_token.token}",
                entity_type="Key",
                entity_id=valid_token.token,
                budget_limits=valid_token.budget_limits,
                fallback_spend=float(valid_token.spend or 0.0),
            )
        )

    if team_object is not None and team_object.team_id is not None:
        team_id = team_object.team_id
        if team_object.max_budget is not None and team_object.max_budget > 0:
            counters.append(
                _BudgetCounter(
                    counter_key=f"spend:team:{team_id}",
                    source_cache_key=f"team_id:{team_id}",
                    max_budget=float(team_object.max_budget),
                    fallback_spend=float(team_object.spend or 0.0),
                    entity_type="Team",
                    entity_id=team_id,
                )
            )
        counters.extend(
            _get_budget_limit_counters(
                entity_prefix=f"spend:team:{team_id}",
                entity_type="Team",
                entity_id=team_id,
                budget_limits=team_object.budget_limits,
                fallback_spend=float(team_object.spend or 0.0),
            )
        )

    if (
        (team_object is None or team_object.team_id is None)
        and user_object is not None
        and user_object.user_id is not None
        and user_object.max_budget is not None
        and user_object.max_budget > 0
    ):
        counters.append(
            _BudgetCounter(
                counter_key=f"spend:user:{user_object.user_id}",
                source_cache_key=user_object.user_id,
                max_budget=float(user_object.max_budget),
                fallback_spend=float(user_object.spend or 0.0),
                entity_type="User",
                entity_id=user_object.user_id,
            )
        )

    end_user_counter = await _get_end_user_budget_counter(
        valid_token=valid_token,
        end_user_id=end_user_id,
        end_user_object=end_user_object,
    )
    if end_user_counter is not None:
        counters.append(end_user_counter)

    counters.extend(
        await _get_tag_budget_counters(
            request_body=request_body,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    )

    team_member_counter = await _get_team_member_budget_counter(
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        user_api_key_cache=user_api_key_cache,
    )
    if team_member_counter is not None:
        counters.append(team_member_counter)

    org_counter = await _get_org_budget_counter(
        valid_token=valid_token,
        team_object=team_object,
        user_api_key_cache=user_api_key_cache,
    )
    if org_counter is not None:
        counters.append(org_counter)

    return counters


async def _get_end_user_budget_counter(
    valid_token: UserAPIKeyAuth,
    end_user_id: Optional[str],
    end_user_object: Optional[Any],
) -> Optional[_BudgetCounter]:
    end_user_id = end_user_id or valid_token.end_user_id
    if end_user_id is None:
        return None

    source_cache_key = f"end_user_id:{end_user_id}"
    max_budget = _to_float(valid_token.end_user_max_budget)
    fallback_spend = 0.0
    if end_user_object is not None:
        fallback_spend = _to_float(_get_value(end_user_object, "spend")) or 0.0
        if max_budget is None:
            budget_table = _get_value(end_user_object, "litellm_budget_table")
            max_budget = _to_float(_get_value(budget_table, "max_budget"))

    if max_budget is None or max_budget <= 0:
        return None

    return _BudgetCounter(
        counter_key=f"spend:end_user:{end_user_id}",
        source_cache_key=source_cache_key,
        max_budget=max_budget,
        fallback_spend=fallback_spend,
        entity_type="EndUser",
        entity_id=end_user_id,
    )


async def _get_tag_budget_counters(
    request_body: dict,
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
    proxy_logging_obj: ProxyLogging,
) -> List[_BudgetCounter]:
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
    from litellm.proxy.auth.auth_checks import get_tag_objects_batch

    tag_names = _dedupe_tags(get_tags_from_request_body(request_body=request_body))
    if not tag_names:
        return []

    tag_objects = await get_tag_objects_batch(
        tag_names=tag_names,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    counters: List[_BudgetCounter] = []
    for tag_name in tag_names:
        tag_object = tag_objects.get(tag_name)
        if tag_object is None:
            continue
        budget_table = _get_value(tag_object, "litellm_budget_table")
        max_budget = _to_float(_get_value(budget_table, "max_budget"))
        if max_budget is None or max_budget <= 0:
            continue
        counters.append(
            _BudgetCounter(
                counter_key=f"spend:tag:{tag_name}",
                source_cache_key=f"tag:{tag_name}",
                max_budget=max_budget,
                fallback_spend=_to_float(_get_value(tag_object, "spend")) or 0.0,
                entity_type="Tag",
                entity_id=tag_name,
            )
        )
    return counters


def _dedupe_tags(tags: List[str]) -> List[str]:
    seen = set()
    deduped_tags = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped_tags.append(tag)
    return deduped_tags


async def _get_team_member_budget_counter(
    valid_token: UserAPIKeyAuth,
    team_object: Optional[LiteLLM_TeamTable],
    user_object: Optional[LiteLLM_UserTable],
    user_api_key_cache: DualCache,
) -> Optional[_BudgetCounter]:
    if (
        team_object is None
        or team_object.team_id is None
        or user_object is None
        or valid_token.user_id is None
    ):
        return None

    membership_cache_key = (
        f"team_membership:{valid_token.user_id}:{team_object.team_id}"
    )
    cached_team_membership = await user_api_key_cache.async_get_cache(
        key=membership_cache_key
    )
    team_membership: Optional[LiteLLM_TeamMembership] = None
    if isinstance(cached_team_membership, LiteLLM_TeamMembership):
        team_membership = cached_team_membership
    elif isinstance(cached_team_membership, dict):
        team_membership = LiteLLM_TeamMembership(**cached_team_membership)

    team_member_budget: Optional[float] = None
    if team_membership is not None and team_membership.litellm_budget_table is not None:
        team_member_budget = team_membership.litellm_budget_table.max_budget
    else:
        default_budget_id = (team_object.metadata or {}).get("team_member_budget_id")
        if isinstance(default_budget_id, str):
            default_budget = await user_api_key_cache.async_get_cache(
                key=f"team_member_default_budget:{default_budget_id}",
            )
            team_member_budget = _to_float(_get_value(default_budget, "max_budget"))

    if team_member_budget is None or team_member_budget <= 0:
        return None

    team_member_spend = (
        cast(LiteLLM_TeamMembership, team_membership).spend
        if team_membership is not None
        else 0.0
    )
    return _BudgetCounter(
        counter_key=f"spend:team_member:{valid_token.user_id}:{team_object.team_id}",
        source_cache_key=membership_cache_key,
        max_budget=float(team_member_budget),
        fallback_spend=float(team_member_spend or 0.0),
        entity_type="TeamMember",
        entity_id=f"{valid_token.user_id}:{team_object.team_id}",
    )


async def _get_org_budget_counter(
    valid_token: UserAPIKeyAuth,
    team_object: Optional[LiteLLM_TeamTable],
    user_api_key_cache: DualCache,
) -> Optional[_BudgetCounter]:
    org_id: Optional[str] = None
    if valid_token.org_id is not None:
        org_id = valid_token.org_id
    elif team_object is not None and team_object.organization_id is not None:
        org_id = team_object.organization_id
    if org_id is None:
        return None

    org_table = await user_api_key_cache.async_get_cache(
        key=f"org_id:{org_id}:with_budget",
    )
    if org_table is None:
        return None

    org_budget_table = _get_value(org_table, "litellm_budget_table")
    if org_budget_table is None:
        return None

    org_max_budget = _to_float(_get_value(org_budget_table, "max_budget"))
    if org_max_budget is None or org_max_budget <= 0:
        return None

    org_spend = _to_float(_get_value(org_table, "spend")) or 0.0
    return _BudgetCounter(
        counter_key=f"spend:org:{org_id}",
        source_cache_key=f"org_id:{org_id}:with_budget",
        max_budget=org_max_budget,
        fallback_spend=org_spend,
        entity_type="Organization",
        entity_id=org_id,
    )


def _get_budget_limit_counters(
    entity_prefix: str,
    entity_type: str,
    entity_id: str,
    budget_limits: Optional[Sequence[Any]],
    fallback_spend: float,
) -> List[_BudgetCounter]:
    counters: List[_BudgetCounter] = []
    if not budget_limits:
        return counters

    for window in budget_limits:
        window_dict = _coerce_window(window)
        budget_duration = window_dict.get("budget_duration")
        max_budget = window_dict.get("max_budget")
        if not budget_duration or max_budget is None or max_budget <= 0:
            continue
        window_start = get_budget_window_start(window_dict)
        if window_start is None:
            verbose_proxy_logger.warning(
                "Skipping budget window with invalid duration for %s=%s: %s",
                entity_type,
                entity_id,
                budget_duration,
            )
            continue
        counters.append(
            _BudgetCounter(
                counter_key=f"{entity_prefix}:window:{budget_duration}",
                max_budget=float(max_budget),
                fallback_spend=0.0,
                entity_type=entity_type,
                entity_id=f"{entity_id}:{budget_duration}",
                spend_log_entity_id=entity_id,
                window_start=window_start,
            )
        )
    return counters


def _coerce_window(window: Any) -> dict:
    if isinstance(window, dict):
        return window
    if isinstance(window, str):
        try:
            parsed = json.loads(window)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if hasattr(window, "model_dump"):
        return window.model_dump()
    return {}


async def _reserve_counter(
    counter: _BudgetCounter,
    reservation_cost: float,
) -> Optional[float]:
    from litellm.proxy.proxy_server import (
        _ensure_spend_counter_initialized,
        _ensure_window_spend_counter_initialized,
        _invalidate_spend_counter,
        _increment_spend_counter_cache,
    )

    attempted_increment = False
    try:
        if counter.source_cache_key is not None:
            await _ensure_spend_counter_initialized(
                counter_key=counter.counter_key,
                source_cache_key=counter.source_cache_key,
            )
        elif (
            counter.spend_log_entity_id is not None and counter.window_start is not None
        ):
            initialized = await _ensure_window_spend_counter_initialized(
                counter_key=counter.counter_key,
                entity_type=counter.entity_type,
                entity_id=counter.spend_log_entity_id,
                window_start=counter.window_start,
            )
            if initialized is False:
                verbose_proxy_logger.warning(
                    "Skipping budget reservation for %s because window spend could not be loaded",
                    counter.counter_key,
                )
                raise _CounterReservationUnavailable

        attempted_increment = True
        reserved_value = await _increment_spend_counter_cache(
            counter_key=counter.counter_key,
            increment=reservation_cost,
        )
        return float(reserved_value) if reserved_value is not None else None
    except _CounterReservationUnavailable:
        raise
    except Exception:
        verbose_proxy_logger.warning(
            "Skipping budget reservation for %s because spend counter reservation failed",
            counter.counter_key,
            exc_info=True,
        )
        counter_invalidated = False
        try:
            await _invalidate_spend_counter(counter_key=counter.counter_key)
            counter_invalidated = True
        except Exception:
            verbose_proxy_logger.warning(
                "Failed to invalidate spend counter after budget reservation failure for %s",
                counter.counter_key,
                exc_info=True,
            )
        raise _CounterReservationUnavailable(
            touched_counter=attempted_increment,
            counter_invalidated=counter_invalidated,
        )


async def _get_current_counter_value(counter: _BudgetCounter) -> float:
    from litellm.proxy.proxy_server import get_current_spend

    return await get_current_spend(
        counter_key=counter.counter_key,
        fallback_spend=counter.fallback_spend,
    )


async def _set_reserved_entries_actual_cost(
    entries: List[dict],
    actual_cost: float,
    default_reserved_cost: float,
) -> None:
    for entry in entries:
        await _set_reserved_entry_actual_cost(
            entry=entry,
            actual_cost=actual_cost,
            default_reserved_cost=default_reserved_cost,
        )


async def _set_reserved_entry_actual_cost(
    entry: dict,
    actual_cost: float,
    default_reserved_cost: float,
) -> None:
    from litellm.proxy.proxy_server import _increment_spend_counter_cache

    counter_key = entry.get("counter_key")
    if counter_key is None:
        return
    reserved_cost = _get_entry_reserved_cost(
        entry=entry,
        default_reserved_cost=default_reserved_cost,
    )
    target_adjustment = actual_cost - reserved_cost
    applied_adjustment = float(entry.get("applied_adjustment") or 0.0)
    adjustment = target_adjustment - applied_adjustment
    if adjustment == 0:
        return
    await _ensure_counter_can_apply_adjustment(
        counter_key=counter_key,
        adjustment=adjustment,
    )
    await _increment_spend_counter_cache(
        counter_key=counter_key,
        increment=adjustment,
    )
    entry["applied_adjustment"] = target_adjustment


async def _ensure_counter_can_apply_adjustment(
    counter_key: str,
    adjustment: float,
) -> None:
    from litellm.proxy.proxy_server import (
        _invalidate_spend_counter,
        spend_counter_cache,
    )

    current_value = await spend_counter_cache.async_get_cache(key=counter_key)
    if current_value is None:
        await _invalidate_spend_counter(counter_key=counter_key)
        raise RuntimeError(
            f"Cannot apply budget reservation adjustment to missing counter {counter_key}"
        )

    try:
        current_float = float(current_value)
    except (TypeError, ValueError):
        await _invalidate_spend_counter(counter_key=counter_key)
        raise RuntimeError(
            f"Cannot apply budget reservation adjustment to non-numeric counter {counter_key}"
        )

    if adjustment < 0 and current_float + adjustment < -1e-12:
        await _invalidate_spend_counter(counter_key=counter_key)
        raise RuntimeError(
            f"Budget reservation adjustment would make counter negative {counter_key}"
        )


async def _release_applied_entries_best_effort(
    entries: List[dict],
    default_reserved_cost: float,
) -> None:
    for entry in entries:
        try:
            await _set_reserved_entry_actual_cost(
                entry=entry,
                actual_cost=0.0,
                default_reserved_cost=default_reserved_cost,
            )
        except Exception:
            counter_key = entry.get("counter_key")
            verbose_proxy_logger.exception(
                "Failed to release partial budget reservation during exception cleanup"
            )
            if counter_key is None:
                continue
            try:
                from litellm.proxy.proxy_server import _invalidate_spend_counter

                await _invalidate_spend_counter(counter_key=counter_key)
            except Exception:
                verbose_proxy_logger.exception(
                    "Failed to invalidate partial budget reservation counter during exception cleanup"
                )


async def _resize_applied_reservation(
    entries: List[dict],
    current_reserved_cost: float,
    new_reserved_cost: float,
) -> None:
    await _set_reserved_entries_actual_cost(
        entries=entries,
        actual_cost=new_reserved_cost,
        default_reserved_cost=current_reserved_cost,
    )
    for entry in entries:
        entry["reserved_cost"] = new_reserved_cost
        entry["applied_adjustment"] = 0.0


def _counter_to_reservation_entry(
    counter: _BudgetCounter,
    reserved_cost: float,
) -> Dict[str, Any]:
    return {
        "counter_key": counter.counter_key,
        "entity_type": counter.entity_type,
        "entity_id": counter.entity_id,
        "reserved_cost": reserved_cost,
        "applied_adjustment": 0.0,
    }


def _get_entry_reserved_cost(entry: dict, default_reserved_cost: float) -> float:
    try:
        return float(entry.get("reserved_cost", default_reserved_cost) or 0.0)
    except (TypeError, ValueError):
        return default_reserved_cost


def get_budget_window_start(window: Any) -> Optional[datetime]:
    window_dict = _coerce_window(window)
    budget_duration = window_dict.get("budget_duration")
    if budget_duration is None:
        return None
    try:
        duration_seconds = duration_in_seconds(str(budget_duration))
    except Exception:
        return None

    reset_at = _coerce_datetime(window_dict.get("reset_at"))
    if reset_at is None:
        return datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    return reset_at - timedelta(seconds=duration_seconds)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def estimate_request_max_cost(
    request_body: dict,
    route: str,
    llm_router: Optional[Router],
) -> Optional[float]:
    model = get_model_from_request(request_body, route)
    if model is None:
        return None

    models = [model] if isinstance(model, str) else model
    estimates = [
        _estimate_request_max_cost_for_model(
            request_body=request_body,
            route=route,
            model=model_name,
            llm_router=llm_router,
        )
        for model_name in models
    ]
    estimates = [estimate for estimate in estimates if estimate is not None]
    if not estimates:
        return None
    return max(cast(List[float], estimates))


def _estimate_request_max_cost_for_model(
    request_body: dict,
    route: str,
    model: str,
    llm_router: Optional[Router],
) -> Optional[float]:
    model_info = _get_model_cost_info(model=model, llm_router=llm_router)
    if model_info is None:
        return None

    image_cost = _estimate_image_generation_cost(
        request_body=request_body,
        model_info=model_info,
    )
    if image_cost is not None:
        return image_cost

    input_cost_per_token = _to_float(model_info.get("input_cost_per_token"))
    output_cost_per_token = _to_float(model_info.get("output_cost_per_token"))
    input_tokens = _estimate_input_tokens(
        request_body=request_body,
        route=route,
        model=model,
        model_info=model_info,
    )
    output_tokens = _estimate_output_tokens(
        request_body=request_body,
        route=route,
        model_info=model_info,
    )
    if input_tokens is None or output_tokens is None:
        return None

    cost = 0.0
    if input_cost_per_token is not None:
        cost += input_tokens * input_cost_per_token
    elif input_tokens > 0:
        return None

    output_multiplier = _get_output_multiplier(request_body=request_body)
    if output_cost_per_token is not None:
        cost += output_tokens * output_multiplier * output_cost_per_token
    elif output_tokens > 0:
        return None

    return cost


def _estimate_image_generation_cost(
    request_body: dict,
    model_info: Dict[str, Any],
) -> Optional[float]:
    """
    Reserve `n × per-image cost` for image-generation requests so concurrent
    requests against a depleted budget cannot all slip past the admission gate
    onto the provider. Token-based pricing (e.g. gpt-image-1) is handled by
    the chat-route token path; per-pixel and size/quality-tiered pricing
    (DALL-E 2 size variants, premium tiers) are not handled here and fall
    through to read-time enforcement.

    The "output" vs "input" cost-per-image naming is inconsistent across
    providers — OpenAI's dall-e-3 entry uses ``input_cost_per_image`` while
    aiml/dall-e-3 uses ``output_cost_per_image`` — so both are summed.
    """
    # Gate strictly on `mode`. Several chat and embedding models carry
    # ``input_cost_per_image`` / ``output_cost_per_image`` to price multimodal
    # *vision input* (e.g. ``gemini-3.1-pro-preview``, ``azure/gpt-realtime-*``,
    # ``amazon.titan-embed-image-v1``). Falling back to "treat as image-gen if
    # an image cost field is present" would short-circuit the token-priced
    # path for those models and reserve a fraction of a cent instead of the
    # true per-token cost. All real image-generation entries in
    # ``model_prices_and_context_window.json`` carry ``mode: image_generation``
    # or ``mode: image_edit``, so the field-presence fallback is unnecessary.
    if model_info.get("mode") not in ("image_generation", "image_edit"):
        return None

    output_cost_per_image = _to_float(model_info.get("output_cost_per_image"))
    input_cost_per_image = _to_float(model_info.get("input_cost_per_image"))
    cost_per_image = (output_cost_per_image or 0.0) + (input_cost_per_image or 0.0)
    if cost_per_image <= 0:
        return None

    n = _to_int(request_body.get("n")) or 1
    return cost_per_image * max(n, 1)


def _get_model_cost_info(
    model: str,
    llm_router: Optional[Router],
) -> Optional[Dict[str, Any]]:
    if llm_router is not None:
        try:
            model_group_info = llm_router.get_model_group_info(model_group=model)
            if model_group_info is not None:
                return model_group_info.model_dump()
        except Exception:
            verbose_proxy_logger.debug(
                "Unable to load router model group info for budget reservation",
                exc_info=True,
            )

    try:
        return dict(litellm.get_model_info(model=model))
    except Exception:
        return None


def _estimate_input_tokens(
    request_body: dict,
    route: str,
    model: str,
    model_info: Dict[str, Any],
) -> Optional[int]:
    try:
        if "messages" in request_body:
            return litellm.token_counter(
                model=model,
                messages=request_body.get("messages") or [],
                tools=request_body.get("tools"),
                tool_choice=request_body.get("tool_choice"),
            )
        if "prompt" in request_body:
            return _count_text_tokens(model=model, text=request_body.get("prompt"))
        if "input" in request_body:
            return _count_text_tokens(model=model, text=request_body.get("input"))
        if "query" in request_body or "documents" in request_body:
            query_tokens = _count_text_tokens(
                model=model, text=request_body.get("query")
            )
            document_tokens = _count_text_tokens(
                model=model,
                text=request_body.get("documents"),
            )
            return query_tokens + document_tokens
    except Exception:
        verbose_proxy_logger.debug(
            "Unable to count input tokens for budget reservation", exc_info=True
        )

    max_input_tokens = _to_int(model_info.get("max_input_tokens"))
    if max_input_tokens is not None:
        return max_input_tokens

    return None


DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK = 16384


def _estimate_output_tokens(
    request_body: dict,
    route: str,
    model_info: Dict[str, Any],
) -> Optional[int]:
    if _is_input_only_route(route=route):
        return 0

    requested: Optional[int] = None
    for key in ("max_completion_tokens", "max_tokens", "max_output_tokens"):
        requested = _to_int(request_body.get(key))
        if requested is not None:
            break

    # Clamp at min(requested-or-default, model_max-or-default). Two purposes:
    # (1) Without an explicit cap we still need a finite reservation so the
    #     atomic admission counter actually bounds concurrent in-flight cost
    #     (mirrors parallel_request_limiter_v3's DEFAULT_MAX_TOKENS_ESTIMATE).
    # (2) An adversarial caller cannot send max_tokens=999999999 to inflate
    #     the reservation up to remaining team headroom and pin the counter
    #     at the cap — the model can only physically emit max_output_tokens
    #     anyway, so reserving more is both wasteful and a DoS surface.
    model_ceiling = (
        _to_int(model_info.get("max_output_tokens"))
        or DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK
    )
    if requested is None:
        requested = DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK
    return min(requested, model_ceiling)


def _count_text_tokens(model: str, text: Any) -> int:
    if text is None:
        return 0

    token_count = 0
    stack = [text]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, list):
            stack.extend(item)
            continue
        if isinstance(item, dict):
            token_count += litellm.token_counter(model=model, text=json.dumps(item))
            continue
        token_count += litellm.token_counter(model=model, text=str(item))
    return token_count


def _get_output_multiplier(request_body: dict) -> int:
    output_multiplier = 1
    for key in ("n", "best_of"):
        value = _to_int(request_body.get(key))
        if value is not None:
            output_multiplier = max(output_multiplier, value)
    return output_multiplier


def _is_input_only_route(route: str) -> bool:
    return any(
        route_part in route
        for route_part in (
            "embeddings",
            "rerank",
            "moderations",
        )
    )


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
