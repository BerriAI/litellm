from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Final, Literal, NoReturn, SupportsFloat, SupportsIndex, SupportsInt, TypeAlias, cast

from fastapi import HTTPException, Request, status
from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.google_genai.adapters.transformation import GOOGLE_GENAI_CONFIG_FIELDS
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.duration_parser import (
    duration_in_seconds,
    is_annual_budget_duration,
    normalize_budget_window_start,
    rolled_budget_window_start,
)
from litellm.litellm_core_utils.duration_parser import (
    get_budget_window_start as calendar_budget_window_start,
)
from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs
from litellm.models.budget import BudgetLimit
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_utils import get_model_from_request
from litellm.proxy.auth.budget_throttle import should_throttle_budget_exceeded
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    compute_budget_reset_at,
    get_budget_reset_settings,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, UserApiKeyCache
from litellm.proxy.spend_tracking.customer_cost_margin import (
    apply_customer_cost_margin_to_estimate,
    key_customer_cost_margin,
)
from litellm.proxy.spend_tracking.input_tokens import (
    count_input_tokens,
    count_input_tokens_for_model,
    inline_pdf_file_count,
    models_counted_for,
    models_served_for,
    openai_chat_pdf_file_tokens,
)
from litellm.proxy.spend_tracking.native_budgets import (
    new_reservation_id,
    reserve_native,
    settle_native_reservation,
)
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement, spend_counter_batch_scope
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.spend_counters import (
    end_user_counter,
    key_counter,
    key_window_counter,
    model_access_group_counter,
    org_counter,
    project_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    team_window_counter,
    user_counter,
    user_window_counter,
)
from litellm.redis.stores.spend_counters.types import customer_window_counter
from litellm.router import Router
from litellm.rust_bridge._native import BOUND_DEFAULT_MAX_OUTPUT_TOKENS
from litellm.types.proxy.auth.customer_hierarchy import (
    AppDurationLimit,
    CustomerScopeKey,
)
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from litellm.types.router import DeploymentTypedDict, ModelGroupInfo
from litellm.types.utils import CallTypes


@dataclass
class BudgetCounter:
    counter_key: str
    max_budget: float
    fallback_spend: float
    entity_type: str
    entity_id: str
    source_cache_key: AuthObject | None = None
    spend_log_entity_id: str | None = None
    window_duration: str | None = None
    window_start: datetime | None = None
    reset_at: datetime | None = None


_BudgetCounter: TypeAlias = BudgetCounter


def customer_window_counters(
    scope_key: CustomerScopeKey,
    entity_id: str,
    budget_limits: Sequence[BudgetLimit | AppDurationLimit],
    *,
    now: datetime | None = None,
    reset_settings: BudgetResetSettings | None = None,
) -> tuple[BudgetCounter, ...]:
    """Use the existing reset calendar with balances owned by the entity, not its shared budget row."""
    current: Final = now if now is not None else datetime.now(timezone.utc)
    settings: Final = reset_settings or get_budget_reset_settings()
    return tuple(_customer_window_counter(scope_key, entity_id, limit, current, settings) for limit in budget_limits)


def _customer_window_counter(
    scope_key: CustomerScopeKey,
    entity_id: str,
    limit: BudgetLimit | AppDurationLimit,
    now: datetime,
    settings: BudgetResetSettings,
) -> _BudgetCounter:
    configured_reset: Final = (
        limit.reset_at.replace(tzinfo=timezone.utc)
        if limit.reset_at is not None and limit.reset_at.tzinfo is None
        else limit.reset_at
    )
    standard_reset: Final = compute_budget_reset_at(
        limit.budget_duration, settings, now, previous_reset=configured_reset
    )
    reset_at: Final = configured_reset if configured_reset is not None and configured_reset > now else standard_reset
    calendar_start: Final = calendar_budget_window_start(
        limit.budget_duration,
        reset_at.astimezone(standard_reset.tzinfo),
        reset_time_of_day=settings.reset_time_of_day,
    ).astimezone(timezone.utc)
    saved_start: Final = limit.window_start if isinstance(limit, BudgetLimit) and reset_at is configured_reset else None
    start: Final = normalize_budget_window_start(
        saved_start.replace(tzinfo=saved_start.tzinfo or timezone.utc).astimezone(timezone.utc)
        if saved_start is not None
        else rolled_budget_window_start(calendar_start, configured_reset)
        if configured_reset is not None and reset_at is standard_reset
        else calendar_start
    )
    return _BudgetCounter(
        counter_key=customer_window_counter(scope_key, entity_id, limit.budget_duration, start),
        max_budget=limit.max_budget,
        fallback_spend=0.0,
        entity_type=scope_key,
        entity_id=entity_id,
        spend_log_entity_id=entity_id,
        window_duration=limit.budget_duration,
        window_start=start,
        reset_at=reset_at.astimezone(timezone.utc),
    )


_CACHED_VALUE: Final = TypeAdapter(object)
_WINDOW_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


class _CounterReservationUnavailable(Exception):
    def __init__(
        self,
        touched_counter: bool = False,
        counter_invalidated: bool = False,
    ) -> None:
        self.touched_counter = touched_counter
        self.counter_invalidated = counter_invalidated
        super().__init__("Counter reservation unavailable")


def _raise_reservation_unavailable(counter_key: str) -> NoReturn:
    verbose_proxy_logger.warning(
        "fail_closed_budget_enforcement: rejecting request — budget reservation for %s could not be written",
        counter_key,
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "Budget enforcement unavailable: the budget reservation could not "
            "be written to the spend counter backend, and "
            "fail_closed_budget_enforcement is enabled, so the request was "
            "rejected to avoid exceeding the configured budget. Retry shortly."
        ),
    )


def get_reserved_counter_keys(budget_reservation: dict | None) -> set:
    if not budget_reservation:
        return set()
    entries: Final = budget_reservation.get("entries") or []
    return {
        entry["counter_key"] for entry in entries if isinstance(entry, dict) and entry.get("counter_key") is not None
    }


_lease_renewals: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio only weak-refs pending tasks


def start_reservation_lease_renewal(
    budget_reservation: Mapping[str, object],
    counter_keys: frozenset[str],
    request_task: asyncio.Task[object] | None = None,
) -> None:
    """A reservation lives inside spend counter keys that expire on their Redis TTL. Renew the TTL
    while the request is in flight so a request longer than the TTL does not drop its
    reservation and admit concurrent requests against the DB floor on any worker."""
    from litellm.proxy.proxy_server import spend_counter_store

    redis_ttl: Final = spend_counter_store.redis_ttl
    if redis_ttl is None or not counter_keys:
        return
    task: Final = asyncio.create_task(
        _renew_reservation_lease(
            budget_reservation=budget_reservation,
            counter_keys=counter_keys,
            interval=redis_ttl / 2,
            request_task=request_task or asyncio.current_task(),
        )
    )
    _lease_renewals.add(task)
    task.add_done_callback(_lease_renewals.discard)


async def _renew_reservation_lease(
    budget_reservation: Mapping[str, object],
    counter_keys: frozenset[str],
    interval: float,
    request_task: asyncio.Task[object] | None,
) -> None:
    """Stops on finalization or once the request task that took the reservation is gone, so a
    disconnect path that skipped reconciliation falls back to the plain counter TTL."""
    from litellm.proxy.proxy_server import refresh_spend_counter_ttl

    deadline: Final = time.monotonic() + litellm.request_timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(interval)
        if budget_reservation.get("finalized") is True or (request_task is not None and request_task.done()):
            return
        for counter_key in counter_keys:
            await refresh_spend_counter_ttl(counter_key=counter_key)


async def reserve_budget_for_request(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: str | None = None,
    end_user_object: object = None,
    apply_user_budget_to_team_keys: bool = False,
    fail_closed_budget_enforcement: bool = False,
    raw_body: bytes | None = None,
    path_params: Mapping[str, object] | None = None,
    request: Request | None = None,
) -> dict | None:
    if valid_token is None or not RouteChecks.runs_inference_checks(route=route):
        return None
    if RouteChecks.is_unbilled_route(route):
        return None
    model: Final = get_model_from_request(request_body, route, llm_router=llm_router, path_params=path_params)
    if model is None:
        return None
    if isinstance(model, str) and await _forwards_subscription_oauth(
        request, model, route, llm_router, valid_token, _WINDOW_OBJECT.validate_python(request_body)
    ):
        return None

    counters: Final = await _get_budget_counters(
        request_body=request_body,
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        end_user_id=end_user_id,
        end_user_object=end_user_object,
        apply_user_budget_to_team_keys=apply_user_budget_to_team_keys,
    )
    if not counters:
        return None

    sent_bodies: Final = _sent_generate_content_bodies(_WINDOW_OBJECT.validate_python(request_body), route)
    estimates: Final = await asyncio.gather(
        *(
            _estimate_hold(
                sent_body=sent_body,
                route=route,
                llm_router=llm_router,
                team_id=valid_token.team_id,
                raw_body=raw_body if len(sent_bodies) == 1 else None,
                path_params=path_params,
            )
            for sent_body in sent_bodies
        )
    )
    hold: Final = _largest_hold(estimates)
    reservation_cost: Final = apply_customer_cost_margin_to_estimate(
        hold.reservation_cost, key_customer_cost_margin(valid_token, valid_token.customer_hierarchy)
    )
    # estimate_request_max_cost still returns None when the model is unknown
    # to the cost map (no token-priced cost fields, e.g. image/audio routes).
    # In that case we fall back to read-time enforcement only.
    if reservation_cost is None or reservation_cost <= 0:
        return None
    return await _native_budget_plan(
        counters=counters,
        valid_token=valid_token,
        reservation_cost=reservation_cost,
        input_token_counts=hold.input_token_counts,
        input_cost=apply_customer_cost_margin_to_estimate(
            hold.input_cost, key_customer_cost_margin(valid_token, valid_token.customer_hierarchy)
        ),
        fail_closed_budget_enforcement=fail_closed_budget_enforcement,
    )


def _sent_generate_content_bodies(request_body: Mapping[str, object], route: str) -> tuple[Mapping[str, object], ...]:
    """Every body a generateContent request with `extra_body` can reach the provider as. A Gemini deployment merges
    it over the top-level fields it names, a config under any of its names included, while the completion adapter
    other providers go through drops it."""
    extra_body: Final = as_str_mapping(request_body.get("extra_body"))
    if extra_body is None or not route.endswith(_GENERATE_CONTENT_ROUTE_SUFFIXES):
        return (request_body,)
    replaces_config: Final = any(field in extra_body for field in GOOGLE_GENAI_CONFIG_FIELDS)
    replaced: Final = frozenset(("extra_body", *(GOOGLE_GENAI_CONFIG_FIELDS if replaces_config else ())))
    return (
        {key: value for key, value in request_body.items() if key != "extra_body"},
        {**{key: value for key, value in request_body.items() if key not in replaced}, **extra_body},
    )


@dataclass(frozen=True, slots=True)
class _HoldEstimate:
    input_token_counts: Mapping[str, int | None]
    reservation_cost: float | None
    input_cost: float | None


async def _estimate_hold(
    sent_body: Mapping[str, object],
    route: str,
    llm_router: Router | None,
    team_id: str | None,
    raw_body: bytes | None,
    path_params: Mapping[str, object] | None,
) -> _HoldEstimate:
    request_body: Final = dict(sent_body)
    input_token_counts: Final = await count_request_input_tokens(
        request_body=request_body,
        route=route,
        llm_router=llm_router,
        team_id=team_id,
        raw_body=raw_body,
        path_params=path_params,
    )
    return _HoldEstimate(
        input_token_counts=input_token_counts,
        reservation_cost=estimate_request_max_cost(
            request_body=request_body,
            route=route,
            llm_router=llm_router,
            input_token_counts=input_token_counts,
            path_params=path_params,
        ),
        input_cost=estimate_request_input_cost(
            request_body=request_body,
            route=route,
            llm_router=llm_router,
            input_token_counts=input_token_counts,
            path_params=path_params,
        ),
    )


def _largest_hold(estimates: Sequence[_HoldEstimate]) -> _HoldEstimate:
    """The costliest body's hold with the highest cancel floor any body has, since the hold cannot tell which
    deployment the request reaches."""
    largest: Final = max(estimates, key=lambda estimate: estimate.reservation_cost or 0.0)
    input_costs: Final = tuple(estimate.input_cost for estimate in estimates if estimate.input_cost is not None)
    return _HoldEstimate(
        input_token_counts=largest.input_token_counts,
        reservation_cost=largest.reservation_cost,
        input_cost=max(input_costs, default=None),
    )


async def _forwards_subscription_oauth(
    request: Request | None,
    model: str,
    route: str,
    llm_router: Router | None,
    valid_token: UserAPIKeyAuth,
    request_body: Mapping[str, object],
) -> bool:
    if request is None or not request.scope.get("headers"):
        return False
    from litellm.llms.anthropic.common_utils import is_anthropic_oauth_key
    from litellm.proxy.litellm_pre_call_utils import (
        caller_alias_target,
        forwarded_anthropic_oauth_token,
        pre_call_served_model,
    )

    if not any(is_anthropic_oauth_key(request.headers.get(name)) for name in ("authorization", "x-api-key")):
        return False
    if route.startswith("/anthropic/"):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            anthropic_route_uses_client_oauth,
        )

        return await anthropic_route_uses_client_oauth(request, valid_token)
    if route not in ("/messages", "/v1/messages", "/chat/completions", "/v1/chat/completions"):
        return False
    if request_body.get("user_config") is not None:
        return False
    from litellm.proxy.common_utils.http_parsing_utils import resolve_inference_model
    from litellm.proxy.proxy_server import (
        general_settings_view,
        prisma_client,
        proxy_config,
        proxy_logging_obj,
        user_model_view,
    )
    from litellm.proxy.route_llm_request import router_settings_override_updates

    hierarchical_settings: Final = await proxy_config.get_hierarchical_router_settings(
        user_api_key_dict=valid_token, prisma_client=prisma_client, proxy_logging_obj=proxy_logging_obj
    )
    effective_request: Final = MappingProxyType(
        {
            **request_body,
            **({"router_settings_override": hierarchical_settings} if hierarchical_settings is not None else {}),
        }
    )
    override_updates: Final = router_settings_override_updates(effective_request)
    if any(
        field in request_body or field in override_updates
        for field in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "models", "model_list")
    ):
        return False
    if llm_router is not None:
        settings: Final = llm_router.settings()
        if any(
            _CACHED_VALUE.validate_python(getattr(settings, field))
            for field in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks")
        ):
            return False
    if forwarded_anthropic_oauth_token(request.headers) is None:
        return False
    prepared_model: Final = resolve_inference_model(
        caller_alias_target(model, valid_token, llm_router), general_settings_view(), user_model_view()
    )
    if not isinstance(prepared_model, str):
        return False
    served_model: Final = pre_call_served_model(
        prepared_model,
        valid_token,
        hierarchical_settings.get("model_group_alias") if hierarchical_settings is not None else None,
    )
    if llm_router is not None:
        upstreams: Final = llm_router.registry().resolve(served_model, kind="upstreams", team_id=valid_token.team_id)
        return bool(upstreams) and all(provider == "anthropic" for _, provider in upstreams)
    try:
        return litellm.get_llm_provider(model=served_model)[1] == "anthropic"
    except litellm.BadRequestError:
        return False


async def _native_budget_plan(
    counters: Sequence[_BudgetCounter],
    valid_token: UserAPIKeyAuth,
    reservation_cost: float,
    input_token_counts: Mapping[str, int | None],
    input_cost: float | None,
    fail_closed_budget_enforcement: bool,
) -> dict[str, object] | None:  # mutable-ok: settlement stamps finalized on the reservation it returns
    with _counters_batch_scope(frozenset(counter.counter_key for counter in counters)):
        reservable: Final = await _initialize_reservation_counters(
            counters=counters,
            fail_closed_budget_enforcement=fail_closed_budget_enforcement,
        )
    if not reservable:
        return None
    input_tokens: Final = max((tokens for tokens in input_token_counts.values() if tokens is not None), default=None)
    return await reserve_native(
        {
            "counters": [
                {
                    "counter_key": counter.counter_key,
                    "max_budget": counter.max_budget,
                    "fallback_spend": counter.fallback_spend,
                    "entity_type": counter.entity_type,
                    "entity_id": counter.entity_id,
                    "spend_log_entity_id": counter.spend_log_entity_id,
                }
                for counter in reservable
            ],
            "reservation_cost": float(reservation_cost),
            "input_cost": float(input_cost or 0.0),
            "input_tokens": input_tokens,
            "throttle_counter": key_counter(str(valid_token.token))
            if should_throttle_budget_exceeded(valid_token)
            else None,
            "fail_closed": fail_closed_budget_enforcement,
            "status_code": int(litellm.budget_exceeded_status_code),
            "reservation_id": new_reservation_id(),
        },
        valid_token,
        {counter.counter_key: counter.reset_at.isoformat() for counter in reservable if counter.reset_at is not None},
    )


async def reconcile_budget_reservation(
    budget_reservation: dict | None,
    actual_cost: float | None,
    finalize: bool = True,
    apply_consistent: bool = True,
) -> tuple[PendingSpendIncrement, ...]:
    if not budget_reservation or budget_reservation.get("finalized") is True:
        return ()
    _ = float(budget_reservation.get("reserved_cost") or 0.0)
    pending: Final = await settle_native_reservation(
        budget_reservation, float(actual_cost or 0.0), apply_consistent=apply_consistent
    )
    if finalize:
        budget_reservation["finalized"] = True
    return pending


def stamp_budget_reservation_actual_cost(budget_reservation: dict | None, actual_cost: float | None) -> None:
    """Record that every reserved counter now holds ``actual_cost``, once the adjustments handed back by
    ``reconcile_budget_reservation(apply_consistent=False)`` have been written."""
    if not budget_reservation:
        return
    reserved_cost: Final = float(budget_reservation.get("reserved_cost") or 0.0)
    actual: Final = float(actual_cost or 0.0)
    settlement_time: Final = datetime.now(timezone.utc)
    for entry in budget_reservation.get("entries") or []:
        if "counter_key" in entry:
            entry["applied_adjustment"] = _entry_settled_cost(
                _WINDOW_OBJECT.validate_python(entry), actual, settlement_time
            ) - _get_entry_reserved_cost(entry=entry, default_reserved_cost=reserved_cost)


async def release_budget_reservation(budget_reservation: dict | None) -> None:
    await reconcile_budget_reservation(
        budget_reservation=budget_reservation,
        actual_cost=0.0,
    )


async def release_budget_reservation_on_cancel(
    budget_reservation: dict[str, object] | None,  # mutable-ok: stamps finalized on the caller's reservation
) -> None:
    """Reconcile a still-open reservation when the request is cancelled mid-flight.

    A client disconnect or timeout cancels the request task, which surfaces as
    CancelledError / GeneratorExit rather than a normal exception, so neither the
    success cost callback nor the failure hook runs and the pre-call reservation
    is never reconciled. Left alone it pins the spend counter above real spend
    and 429s subsequent requests until the counter's TTL expires.

    Reconcile to the request's input-token cost rather than refunding to zero:
    by the time a request is cancelled in-flight the provider call was already
    dispatched, so the input tokens were billed even if no chunk reached the
    client. Refunding to zero would let a caller abort pre-token to dodge that
    charge; the worst-case output portion of the reservation is still released.

    asyncio.shield keeps the reconcile running to completion even though the
    surrounding task is being cancelled. The `finalized` guard makes this a no-op
    when success/failure handling already reconciled, so calling it on every
    cancellation path is safe.
    """
    if not budget_reservation or budget_reservation.get("finalized") is True:
        return
    input_cost: Final = budget_reservation.get("input_cost")
    incurred_cost: Final = float(input_cost) if isinstance(input_cost, (int, float)) else 0.0
    try:
        await asyncio.shield(
            reconcile_budget_reservation(budget_reservation=budget_reservation, actual_cost=incurred_cost)
        )
    except (asyncio.CancelledError, Exception):
        pass


async def invalidate_budget_reservation_counters(
    budget_reservation: dict | None,
) -> None:
    if budget_reservation is None:
        return

    from litellm.proxy.proxy_server import invalidate_spend_counter

    for counter_key in get_reserved_counter_keys(budget_reservation=budget_reservation):
        await invalidate_spend_counter(counter_key=counter_key)


async def release_or_invalidate_budget_reservation(
    budget_reservation: dict[str, object] | None,  # mutable-ok: stamps finalized on the caller's reservation
) -> None:
    """Reconcile a still-open reservation on a terminal path that settles no cost.

    A failed or upstream-refused request never runs the success cost callback, so
    its pre-call reservation stays open and keeps the spend counter pinned above
    real spend until the counter's TTL expires, 429ing later requests on the same
    key. Release it to zero; if the release itself fails (e.g. the counter store is
    unreachable) drop the reserved counters directly and mark the reservation
    finalized so nothing reprocesses it. Idempotent: the finalized guard makes a
    second call a no-op once success or failure handling already reconciled.
    """
    if budget_reservation is None or budget_reservation.get("finalized") is True:
        return
    try:
        await asyncio.shield(release_budget_reservation(budget_reservation=budget_reservation))
    except Exception:  # noqa: BLE001  # a cleanup failure must not pin the counter; drop it directly instead
        verbose_proxy_logger.exception("Failed to release budget reservation; invalidating counters")
        try:
            await invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
        except Exception:  # noqa: BLE001  # nothing left to try; the finalized stamp below keeps it from being reprocessed
            verbose_proxy_logger.exception("Failed to invalidate budget reservation counters after release failed")
        finally:
            budget_reservation["finalized"] = True


async def release_cancelled_budget_reservation(
    budget_reservation: dict[str, object] | None,  # mutable-ok: stamps finalized on the caller's reservation
) -> None:
    if budget_reservation is not None and budget_reservation.get("callback_bound") is True:
        await release_budget_reservation_on_cancel(budget_reservation)
        return
    await asyncio.shield(release_or_invalidate_budget_reservation(budget_reservation))


async def release_unbound_budget_reservation(budget_reservation: Mapping[str, object]) -> None:
    """Release a reservation no logging callback took ownership of, once the request ended.

    A handler whose litellm call never builds a logging object (batch cancel, file
    content, anything without the client decorator) runs no cost callback, so nothing
    else would ever reconcile its reservation. A bound reservation is left alone: its
    success or failure handler settles it, possibly after the response has been sent.
    """
    if not isinstance(budget_reservation, dict) or budget_reservation.get("callback_bound") is True:
        return
    await release_or_invalidate_budget_reservation(budget_reservation=budget_reservation)


async def _get_budget_counters(
    request_body: dict,
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: str | None = None,
    end_user_object: object = None,
    apply_user_budget_to_team_keys: bool = False,
) -> list[_BudgetCounter]:
    counters: Final[list[_BudgetCounter]] = []

    if valid_token.customer_hierarchy is not None:
        for scope in valid_token.customer_hierarchy.scopes:
            counters.extend(customer_window_counters(scope.key, scope.entity_id, scope.budget_limits))

    if valid_token.token is not None:
        if valid_token.max_budget is not None and valid_token.max_budget > 0:
            counters.append(
                _BudgetCounter(
                    counter_key=key_counter(valid_token.token),
                    source_cache_key=AuthObject.virtual_key(valid_token.token),
                    max_budget=float(valid_token.max_budget),
                    fallback_spend=float(valid_token.spend or 0.0),
                    entity_type="Key",
                    entity_id=valid_token.token,
                )
            )
        counters.extend(
            _get_budget_limit_counters(
                window_counter=partial(key_window_counter, valid_token.token),
                entity_type="Key",
                entity_id=valid_token.token,
                budget_limits=valid_token.budget_limits,
                fallback_spend=float(valid_token.spend or 0.0),
            )
        )

    if team_object is not None and team_object.team_id is not None:
        team_id: Final = team_object.team_id
        if team_object.max_budget is not None and team_object.max_budget > 0:
            counters.append(
                _BudgetCounter(
                    counter_key=team_counter(team_id),
                    source_cache_key=AuthObject.team(team_id),
                    max_budget=float(team_object.max_budget),
                    fallback_spend=float(team_object.spend or 0.0),
                    entity_type="Team",
                    entity_id=team_id,
                )
            )
        counters.extend(
            _get_budget_limit_counters(
                window_counter=partial(team_window_counter, team_id),
                entity_type="Team",
                entity_id=team_id,
                budget_limits=team_object.budget_limits,
                fallback_spend=float(team_object.spend or 0.0),
            )
        )

    is_team_key: Final = team_object is not None and team_object.team_id is not None
    if (not is_team_key or apply_user_budget_to_team_keys) and user_object is not None:
        counters.extend(
            _get_budget_limit_counters(
                window_counter=partial(user_window_counter, user_object.user_id),
                entity_type="User",
                entity_id=user_object.user_id,
                budget_limits=user_object.budget_limits,
                fallback_spend=0.0,
            )
        )
    user_budget: Final = user_object.effective_max_budget() if user_object is not None else None
    if (
        (not is_team_key or apply_user_budget_to_team_keys)
        and user_object is not None
        and user_object.user_id is not None
        and user_budget is not None
        and user_budget > 0
    ):
        counters.append(
            _BudgetCounter(
                counter_key=user_counter(user_object.user_id),
                source_cache_key=AuthObject.user(user_object.user_id),
                max_budget=user_budget,
                fallback_spend=float(user_object.spend or 0.0),
                entity_type="User",
                entity_id=user_object.user_id,
            )
        )

    end_user_counter: Final = await _get_end_user_budget_counter(
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

    counters.extend(
        await _get_model_access_group_budget_counters(
            valid_token=valid_token,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
        )
    )

    team_member_counter: Final = await _get_team_member_budget_counter(
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        user_api_key_cache=user_api_key_cache,
    )
    if team_member_counter is not None:
        counters.append(team_member_counter)

    org_counter: Final = await _get_org_budget_counter(
        valid_token=valid_token,
        team_object=team_object,
        user_api_key_cache=user_api_key_cache,
    )
    if org_counter is not None:
        counters.append(org_counter)

    project_counter: Final = await _get_project_budget_counter(
        valid_token=valid_token,
        user_api_key_cache=user_api_key_cache,
    )
    if project_counter is not None:
        counters.append(project_counter)

    return counters


async def _get_end_user_budget_counter(
    valid_token: UserAPIKeyAuth,
    end_user_id: str | None,
    end_user_object: object,
) -> _BudgetCounter | None:
    end_user_id = end_user_id or valid_token.end_user_id
    if end_user_id is None:
        return None

    source_cache_key: Final = AuthObject.end_user(end_user_id)
    max_budget = _to_float(valid_token.end_user_max_budget)
    fallback_spend = 0.0
    if end_user_object is not None:
        fallback_spend = _to_float(_get_value(end_user_object, "spend")) or 0.0
        if max_budget is None:
            budget_table: Final = _get_value(end_user_object, "litellm_budget_table")
            max_budget = _to_float(_get_value(budget_table, "max_budget"))

    if max_budget is None or max_budget <= 0:
        return None

    return _BudgetCounter(
        counter_key=end_user_counter(end_user_id),
        source_cache_key=source_cache_key,
        max_budget=max_budget,
        fallback_spend=fallback_spend,
        entity_type="EndUser",
        entity_id=end_user_id,
    )


async def _get_tag_budget_counters(
    request_body: dict,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> list[_BudgetCounter]:
    from litellm.proxy.auth.auth_checks import get_tag_objects_batch
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body

    tag_names: Final = _dedupe_tags(get_tags_from_request_body(request_body=request_body))
    if not tag_names:
        return []

    tag_objects: Final = await get_tag_objects_batch(
        tag_names=tag_names,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )

    counters: Final[list[_BudgetCounter]] = []
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
                counter_key=tag_counter(tag_name),
                source_cache_key=AuthObject.tag(tag_name),
                max_budget=max_budget,
                fallback_spend=_to_float(_get_value(tag_object, "spend")) or 0.0,
                entity_type="Tag",
                entity_id=tag_name,
            )
        )
    return counters


async def _get_model_access_group_budget_counters(
    valid_token: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
) -> list[_BudgetCounter]:
    """Reservation counters for the model access groups that authorized this request.

    The names come off the auth object rather than the request body: ``common_checks`` already
    resolved which granted groups serve the requested model, and re-deriving that here would both
    duplicate the walk and risk disagreeing with what the spend writer attributes.
    """
    from litellm.proxy.auth.auth_checks import get_model_access_group_budgets_batch

    group_names: Final = tuple(dict.fromkeys(valid_token.matched_model_access_groups or ()))
    if not group_names:
        return []

    budgets: Final = await get_model_access_group_budgets_batch(
        access_group_names=group_names,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
    )
    candidates: Final = (_model_access_group_counter(group, budgets.get(group)) for group in group_names)
    return [counter for counter in candidates if counter is not None]


def _model_access_group_counter(group: str, budget: ModelAccessGroupBudget | None) -> _BudgetCounter | None:
    """A counter for one group, or nothing when the group carries no budget to reserve against."""
    if budget is None or budget.max_budget is None or budget.max_budget <= 0:
        return None
    return _BudgetCounter(
        counter_key=model_access_group_counter(group),
        source_cache_key=AuthObject.model_access_group(group),
        max_budget=budget.max_budget,
        fallback_spend=budget.spend,
        entity_type="Model access group",
        entity_id=group,
    )


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: Final = set()
    deduped_tags: Final = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped_tags.append(tag)
    return deduped_tags


@with_service_target(AUTH_OBJECTS_TARGET)
async def _get_team_member_budget_counter(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    user_api_key_cache: UserApiKeyCache,
) -> _BudgetCounter | None:
    if team_object is None or team_object.team_id is None or user_object is None or valid_token.user_id is None:
        return None

    membership_cache_key: Final = AuthObject.team_membership_reservation(
        user_id=valid_token.user_id, team_id=team_object.team_id
    )
    cached_team_membership: Final = _CACHED_VALUE.validate_python(
        await user_api_key_cache.async_get_cache(key=membership_cache_key)
    )
    team_membership: LiteLLM_TeamMembership | None = None
    if isinstance(cached_team_membership, LiteLLM_TeamMembership):
        team_membership = cached_team_membership
    elif isinstance(cached_team_membership, dict):
        team_membership = LiteLLM_TeamMembership.model_validate(cached_team_membership)

    member_budget_row: Final = team_membership.litellm_budget_table if team_membership is not None else None
    now: Final = datetime.now(timezone.utc)
    team_member_budget: float | None = None
    if team_membership is not None and member_budget_row is not None and member_budget_row.max_budget is not None:
        team_member_budget = team_membership.effective_budget_with_rollover(now=now)
    else:
        default_budget_id: Final = (team_object.metadata or {}).get("team_member_budget_id")
        if isinstance(default_budget_id, str):
            default_budget: Final = _CACHED_VALUE.validate_python(
                await user_api_key_cache.async_get_cache(key=AuthObject.team_member_default_budget(default_budget_id))
            )
            default_cap: Final = _to_float(_get_value(default_budget, "max_budget"))
            if default_cap is not None and default_cap > 0:
                team_member_budget = default_cap + (
                    member_budget_row.active_temp_budget_increase(now=now) if member_budget_row is not None else 0.0
                )

    if team_member_budget is None or team_member_budget <= 0:
        return None

    team_member_spend = cast(LiteLLM_TeamMembership, team_membership).spend if team_membership is not None else 0.0
    return _BudgetCounter(
        counter_key=team_member_counter(valid_token.user_id, team_object.team_id),
        source_cache_key=membership_cache_key,
        max_budget=float(team_member_budget),
        fallback_spend=float(team_member_spend or 0.0),
        entity_type="TeamMember",
        entity_id=f"{valid_token.user_id}:{team_object.team_id}",
    )


@with_service_target(AUTH_OBJECTS_TARGET)
async def _get_org_budget_counter(
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    user_api_key_cache: UserApiKeyCache,
) -> _BudgetCounter | None:
    org_id: str | None = None
    if valid_token.org_id is not None:
        org_id = valid_token.org_id
    elif team_object is not None and team_object.organization_id is not None:
        org_id = team_object.organization_id
    if org_id is None:
        return None

    org_table: Final = _CACHED_VALUE.validate_python(
        await user_api_key_cache.async_get_cache(key=AuthObject.org_with_budget(org_id))
    )
    if org_table is None:
        return None

    org_budget_table: Final = _get_value(org_table, "litellm_budget_table")
    if org_budget_table is None:
        return None

    org_max_budget: Final = _to_float(_get_value(org_budget_table, "max_budget"))
    if org_max_budget is None or org_max_budget <= 0:
        return None

    org_spend: Final = _to_float(_get_value(org_table, "spend")) or 0.0
    return _BudgetCounter(
        counter_key=org_counter(org_id),
        source_cache_key=AuthObject.org_with_budget(org_id),
        max_budget=org_max_budget,
        fallback_spend=org_spend,
        entity_type="Organization",
        entity_id=org_id,
    )


@with_service_target(AUTH_OBJECTS_TARGET)
async def _get_project_budget_counter(
    valid_token: UserAPIKeyAuth,
    user_api_key_cache: UserApiKeyCache,
) -> _BudgetCounter | None:
    if valid_token.project_id is None:
        return None

    source_cache_key: Final = AuthObject.project(valid_token.project_id)
    project_object: Final = _CACHED_VALUE.validate_python(
        await user_api_key_cache.async_get_cache(key=source_cache_key)
    )
    if project_object is None:
        return None

    project_budget_table: Final = _get_value(project_object, "litellm_budget_table")
    if project_budget_table is None:
        return None

    project_max_budget: Final = _to_float(_get_value(project_budget_table, "max_budget"))
    if project_max_budget is None or project_max_budget <= 0 or not math.isfinite(project_max_budget):
        return None

    return _BudgetCounter(
        counter_key=project_counter(valid_token.project_id),
        source_cache_key=source_cache_key,
        max_budget=project_max_budget,
        fallback_spend=_to_float(_get_value(project_object, "spend")) or 0.0,
        entity_type="Project",
        entity_id=valid_token.project_id,
    )


def _get_budget_limit_counters(
    window_counter: Callable[[str], str],
    entity_type: str,
    entity_id: str,
    budget_limits: Sequence[object] | None,
    fallback_spend: float,
) -> list[_BudgetCounter]:
    counters: Final[list[_BudgetCounter]] = []
    if not budget_limits:
        return counters

    for window in budget_limits:
        window_dict = _coerce_window(window)
        budget_duration = window_dict.get("budget_duration")
        max_budget = _to_float(window_dict.get("max_budget"))
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
                counter_key=window_counter(str(budget_duration)),
                max_budget=float(max_budget),
                fallback_spend=0.0,
                entity_type=entity_type,
                entity_id=f"{entity_id}:{budget_duration}",
                spend_log_entity_id=entity_id,
                window_duration=str(budget_duration),
                window_start=window_start,
            )
        )
    return counters


def _coerce_window(window: object) -> Mapping[str, object]:
    if isinstance(window, Mapping):
        return window
    if isinstance(window, str):
        try:
            return _WINDOW_OBJECT.validate_python(json.loads(window))
        except Exception:
            return {}
    model_dump: Final = getattr(window, "model_dump", None)
    if not callable(model_dump):
        return {}
    dumped: Final[object] = model_dump()
    return dumped if isinstance(dumped, Mapping) else {}


async def _initialize_reservation_counters(
    counters: Sequence[_BudgetCounter],
    fail_closed_budget_enforcement: bool,
) -> tuple[_BudgetCounter, ...]:
    """The counters whose current value is loaded, in order; one that cannot be loaded is skipped (or rejects the
    request under fail-closed enforcement) exactly as it was when each counter was reserved on its own.
    """
    return tuple([counter async for counter in _loaded_reservation_counters(counters, fail_closed_budget_enforcement)])


async def _loaded_reservation_counters(
    counters: Sequence[_BudgetCounter], fail_closed_budget_enforcement: bool
) -> AsyncIterator[_BudgetCounter]:
    for counter in counters:
        if await _reservation_counter_loaded(counter, fail_closed_budget_enforcement):
            yield counter


async def _reservation_counter_loaded(counter: _BudgetCounter, fail_closed_budget_enforcement: bool) -> bool:
    try:
        await _initialize_reservation_counter(counter=counter)
    except _CounterReservationUnavailable:
        if fail_closed_budget_enforcement:
            _raise_reservation_unavailable(counter_key=counter.counter_key)
        return False
    return True


async def _initialize_reservation_counter(counter: _BudgetCounter) -> None:
    from litellm.proxy.proxy_server import (
        ensure_spend_counter_initialized,
        ensure_window_spend_counter_initialized,
        invalidate_spend_counter,
    )

    try:
        if counter.source_cache_key is not None:
            await ensure_spend_counter_initialized(
                counter_key=counter.counter_key,
                source_cache_key=counter.source_cache_key,
            )
        elif counter.spend_log_entity_id is not None and counter.window_start is not None:
            initialized: Final = await ensure_window_spend_counter_initialized(
                counter_key=counter.counter_key,
                entity_type=counter.entity_type,
                entity_id=counter.spend_log_entity_id,
                window_duration=counter.window_duration,
                window_start=counter.window_start,
            )
            if initialized is False:
                verbose_proxy_logger.warning(
                    "Skipping budget reservation for %s because window spend could not be loaded",
                    counter.counter_key,
                )
                raise _CounterReservationUnavailable
    except _CounterReservationUnavailable:
        raise
    except Exception:
        verbose_proxy_logger.warning(
            "Skipping budget reservation for %s because spend counter reservation failed",
            counter.counter_key,
            exc_info=True,
        )
        try:
            await invalidate_spend_counter(counter_key=counter.counter_key)
        except Exception:
            verbose_proxy_logger.warning(
                "Failed to invalidate spend counter after budget reservation failure for %s",
                counter.counter_key,
                exc_info=True,
            )
        raise _CounterReservationUnavailable


def _counters_batch_scope(counter_keys: frozenset[str]) -> spend_counter_batch_scope:
    """Each counter is read once, then written, so one MGET up front serves every read in the loop."""
    from litellm.proxy.proxy_server import spend_counter_store

    return spend_counter_batch_scope(spend_counter_store, counter_keys=counter_keys)


def _entry_settled_cost(entry: Mapping[str, object], actual_cost: float, settlement_time: datetime | None) -> float:
    reset_at: Final = _coerce_datetime(entry.get("window_reset_at"))
    return 0.0 if settlement_time is not None and reset_at is not None and reset_at <= settlement_time else actual_cost


async def reseed_reserved_counter(counter_key: str, actual_cost: float) -> None:
    from litellm.proxy.proxy_server import increment_spend_counter_cache, reseed_spend_counter_from_db

    reseeded: Final = await reseed_spend_counter_from_db(counter_key=counter_key)
    if reseeded and actual_cost > 0:
        await increment_spend_counter_cache(counter_key=counter_key, increment=actual_cost)


def _get_entry_reserved_cost(entry: dict, default_reserved_cost: float) -> float:
    try:
        return float(entry.get("reserved_cost", default_reserved_cost) or 0.0)
    except (TypeError, ValueError):
        return default_reserved_cost


def fixed_budget_window(window: object) -> tuple[str, datetime] | None:
    fields: Final = _coerce_window(window)
    start: Final = get_budget_window_start(window)
    duration: Final = fields.get("budget_duration")
    if not fields.get("reset_at") or not isinstance(duration, str) or start is None:
        return None
    return duration, start


def get_budget_window_start(window: object, reset_settings: BudgetResetSettings | None = None) -> datetime | None:
    window_dict: Final = _coerce_window(window)
    budget_duration: Final = window_dict.get("budget_duration")
    if budget_duration is None:
        return None
    saved_start: Final = _coerce_datetime(window_dict.get("window_start"))
    if saved_start is not None:
        start: Final = saved_start if saved_start.tzinfo is not None else saved_start.replace(tzinfo=timezone.utc)
        return normalize_budget_window_start(start.astimezone(timezone.utc))
    try:
        if is_annual_budget_duration(str(budget_duration)):
            annual_reset_at: Final = _coerce_datetime(window_dict.get("reset_at"))
            if annual_reset_at is not None:
                settings: Final = reset_settings or get_budget_reset_settings()
                return calendar_budget_window_start(
                    str(budget_duration), annual_reset_at, settings.timezone, settings.reset_time_of_day
                ).astimezone(timezone.utc)
        duration_seconds: Final = duration_in_seconds(str(budget_duration))
    except Exception:
        return None

    reset_at = _coerce_datetime(window_dict.get("reset_at"))
    if reset_at is None:
        return normalize_budget_window_start(datetime.now(timezone.utc) - timedelta(seconds=duration_seconds))
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    return normalize_budget_window_start(reset_at - timedelta(seconds=duration_seconds))


def _coerce_datetime(value: object) -> datetime | None:
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
    llm_router: Router | None,
    input_token_counts: Mapping[str, int | None] | None = None,
    path_params: Mapping[str, object] | None = None,
) -> float | None:
    return _bound(request_body, route, llm_router, input_token_counts, path_params)[0]


def estimate_request_input_cost(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    input_token_counts: Mapping[str, int | None] | None = None,
    path_params: Mapping[str, object] | None = None,
) -> float | None:
    """Cost of the request's input tokens alone.

    Once the provider request is dispatched the input tokens are billed even if
    the client disconnects before the first chunk, so this is the cost floor a
    cancelled in-flight request has already incurred. A cancelled reservation is
    reconciled to this instead of being refunded to zero.
    """
    return _bound(request_body, route, llm_router, input_token_counts, path_params)[1]


_GroupRates = TypedDict(
    "_GroupRates",
    {
        "from": ReadOnly[Literal["group"]],
        "input": ReadOnly[float | None],
        "output": ReadOnly[float | None],
        "max_input_tokens": ReadOnly[int | None],
        "max_output_tokens": ReadOnly[int | None],
    },
)
_InfoRow = TypedDict("_InfoRow", {"from": ReadOnly[Literal["row"]], "row": ReadOnly[int]})


class _CostShape(TypedDict):
    flat: ReadOnly[_GroupRates | _InfoRow]
    tiers: ReadOnly[int | None]


class _BoundModel(TypedDict):
    counted_input: ReadOnly[int | None]
    shapes: ReadOnly[Sequence[_CostShape]]


class _RequestCaps(TypedDict):
    requested_output: ReadOnly[int | None]
    n: ReadOnly[int | None]
    best_of: ReadOnly[int | None]
    input_only: ReadOnly[bool]


class _BoundDocument(TypedDict):
    models: ReadOnly[Sequence[_BoundModel]]
    request: ReadOnly[_RequestCaps]


@dataclass(frozen=True, slots=True)
class _ModelShapes:
    """One requested model's cost shapes, the model info rows they read listed in ``rows`` by index."""

    shapes: tuple[_CostShape, ...]
    rows: tuple[Mapping[str, object], ...]


_NO_SHAPES: Final = _ModelShapes(shapes=(), rows=())


def _bound(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    input_token_counts: Mapping[str, int | None] | None,
    path_params: Mapping[str, object] | None,
) -> tuple[float | None, float | None]:
    """The legacy budget hold and the input cost floor, priced by the Rust pricing core's ``bound``."""
    models: Final = _get_request_models(
        request_body=request_body, route=route, llm_router=llm_router, path_params=path_params
    )
    per_model: Final = tuple(_model_shapes(model, llm_router) for model in models)
    offsets: Final = tuple(accumulate((len(shapes.rows) for shapes in per_model), initial=0))
    document: Final = _BoundDocument(
        models=tuple(
            _BoundModel(
                counted_input=_counted_input_tokens(request_body, model, input_token_counts),
                shapes=tuple(_offset_shape(shape, offset) for shape in shapes.shapes),
            )
            for model, shapes, offset in zip(models, per_model, offsets)
        ),
        request=_RequestCaps(
            requested_output=_requested_output_tokens(request_body),
            n=_to_int(request_body.get("n")),
            best_of=_to_int(request_body.get("best_of")),
            input_only=_is_input_only_route(route=route),
        ),
    )
    return model_cost_inputs.bound(document, tuple(chain.from_iterable(shapes.rows for shapes in per_model)))


def _offset_shape(shape: _CostShape, offset: int) -> _CostShape:
    flat: Final = shape["flat"]
    return _CostShape(
        flat=_InfoRow({"from": "row", "row": flat["row"] + offset}) if flat["from"] == "row" else flat,
        tiers=None if shape["tiers"] is None else shape["tiers"] + offset,
    )


def _counted_input_tokens(
    request_body: dict, model: str, input_token_counts: Mapping[str, int | None] | None
) -> int | None:
    counted: Final = (input_token_counts or {}).get(model)
    return counted if counted is not None else count_input_tokens_for_model(request_body=request_body, model=model)


def _model_shapes(model: str, llm_router: Router | None) -> _ModelShapes:
    """Cost shapes to estimate a request against for one model group.

    Reservation runs before routing, so the deployment that will serve the request
    is unknown. Rather than guess, we estimate the cost against every eligible
    pricing shape in the group (the group's flat rates plus each deployment's
    tiered table) and let the caller reserve the maximum, so a cheaper sibling
    deployment can never leave the request under-reserved.
    """
    try:
        base: Final = _get_model_cost_info(model=model, llm_router=llm_router)
        if base is None:
            return _NO_SHAPES
        tiered_tables: Final = _get_deployment_tiered_pricing_tables(model=model, llm_router=llm_router)
    except Exception:
        verbose_proxy_logger.debug(
            "Unable to load model cost info for budget reservation",
            exc_info=True,
        )
        return _NO_SHAPES
    flat: Final = (
        _GroupRates(
            {
                "from": "group",
                "input": base.input_cost_per_token,
                "output": base.output_cost_per_token,
                "max_input_tokens": _to_int(base.max_input_tokens),
                "max_output_tokens": _to_int(base.max_output_tokens),
            }
        )
        if isinstance(base, ModelGroupInfo)
        else _InfoRow({"from": "row", "row": 0})
    )
    base_rows: Final = () if isinstance(base, ModelGroupInfo) else (base,)
    return _ModelShapes(
        shapes=(
            _CostShape(flat=flat, tiers=None),
            *(_CostShape(flat=flat, tiers=len(base_rows) + index) for index in range(len(tiered_tables))),
        ),
        rows=(*base_rows, *({"tiered_pricing": table} for table in tiered_tables)),
    )


def _get_model_cost_info(
    model: str,
    llm_router: Router | None,
) -> ModelGroupInfo | Mapping[str, object] | None:
    if llm_router is not None:
        model_group_info: Final = llm_router.registry().group(model, field="cached_info")
        if model_group_info is not None:
            return model_group_info
    return dict(litellm.get_model_info(model=model))


def _deployment_tiered_pricing_table(
    deployment: DeploymentTypedDict,
    llm_router: Router,
) -> Sequence[Mapping[str, object]] | None:
    model_id: Final = _get_value(_get_value(deployment, "model_info"), "id")
    backend_model: Final = _get_value(_get_value(deployment, "litellm_params"), "model")
    if not isinstance(model_id, str) or not isinstance(backend_model, str):
        return None
    deployment_model_info: Final = llm_router.registry().deployment_model_info(model_id, backend_model, cached=True)
    if deployment_model_info is None:
        return None
    tiered_pricing: Final = deployment_model_info.get("tiered_pricing")
    if isinstance(tiered_pricing, list) and tiered_pricing:
        return tiered_pricing
    return None


def _get_deployment_tiered_pricing_tables(
    model: str,
    llm_router: Router | None,
) -> Sequence[Sequence[Mapping[str, object]]]:
    if llm_router is None:
        return []
    deployments: Final = llm_router.registry().deployments(model) or []
    return [
        table
        for deployment in deployments
        if (table := _deployment_tiered_pricing_table(deployment, llm_router)) is not None
    ]


def _get_request_models(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    path_params: Mapping[str, object] | None = None,
) -> Sequence[str]:
    model: Final = get_model_from_request(request_body, route, llm_router=llm_router, path_params=path_params)
    if model is None:
        return ()
    return (model,) if isinstance(model, str) else tuple(model)


async def count_request_input_tokens(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    team_id: str | None,
    raw_body: bytes | None = None,
    path_params: Mapping[str, object] | None = None,
) -> Mapping[str, int | None]:
    """Input-token count per candidate model, counted once per request.

    The counts are reused by both the max-cost and the input-cost estimate."""
    models: Final = _get_request_models(
        request_body=request_body, route=route, llm_router=llm_router, path_params=path_params
    )
    if not models:
        return MappingProxyType({})
    counted_models: Final = MappingProxyType(
        {model: models_counted_for(model, llm_router, team_id) for model in models}
    )
    served_models: Final = (
        tuple(dict.fromkeys(chain.from_iterable(models_served_for(model, llm_router, team_id) for model in models)))
        if "contents" in request_body
        else ()
    )
    counts: Final = await count_input_tokens(
        request_body=request_body,
        raw_body=raw_body,
        models=tuple(dict.fromkeys(chain.from_iterable(counted_models.values()))),
        served_models=served_models,
    )
    chat_pdf_files: Final = (
        inline_pdf_file_count(request_body.get("messages"))
        if CallTypes.acompletion in (get_call_types_for_route(route) or ())
        else 0
    )
    return MappingProxyType(
        {model: _largest_count(counts, names, chat_pdf_files) for model, names in counted_models.items()}
    )


def _largest_count(counts: Mapping[str, int], names: Sequence[str], chat_pdf_files: int) -> int | None:
    return max(
        (counts[name] + openai_chat_pdf_file_tokens(name, chat_pdf_files) for name in names if name in counts),
        default=None,
    )


DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK: Final = BOUND_DEFAULT_MAX_OUTPUT_TOKENS
_GENERATE_CONTENT_ROUTE_SUFFIXES: Final = (":generateContent", ":streamGenerateContent")


_OUTPUT_TOKEN_FIELDS: Final = ("max_completion_tokens", "max_tokens", "max_output_tokens")


def _requested_output_tokens(request_body: Mapping[str, object]) -> int | None:
    inference_config: Final = request_body.get("inferenceConfig")
    candidates: Final = (
        *(request_body.get(field) for field in _OUTPUT_TOKEN_FIELDS),
        inference_config.get("maxTokens") if isinstance(inference_config, Mapping) else None,
    )
    return next((tokens for tokens in map(_to_int, candidates) if tokens is not None), None)


def _is_input_only_route(route: str) -> bool:
    return any(
        route_part in route
        for route_part in (
            "embeddings",
            "rerank",
            "moderations",
        )
    )


def _to_float(value: object) -> float | None:
    if not isinstance(value, (SupportsFloat, SupportsIndex, str, bytes, bytearray)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _non_negative_int(value: object) -> int | None:
    parsed: Final = _to_int(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (SupportsInt, SupportsIndex, str, bytes, bytearray)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _get_value(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)
