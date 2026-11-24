"""Deprecated: the ledger-off post-call spend counter step.

Frozen as it was before the ledger. With the ledger on, the whole step runs in Rust
(``Budgets::charge_counters`` through ``rust_spend_counters.charge_spend_counters``), and nothing here
runs. Delete this module together with the ledger-off setting; do not add to it.

Moved out of ``proxy_server`` unchanged. The proxy's globals and shared spend counter helpers are read
from ``proxy_server`` at call time, so patching them there still reaches this code.
"""

import asyncio
from collections.abc import Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm._internal_context import service_target
from litellm.models.budget_schedule import is_scheduled_duration, window_counter_name
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import build_window_spend_transaction
from litellm.proxy.spend_tracking.budget_reservation import (
    BudgetCounter,
    get_budget_window_start,
    stamp_budget_reservation_actual_cost,
)
from litellm.proxy.spend_tracking.committed_spend import (
    committed_spend_key,
    committed_spend_keys,
    has_committed_twin,
    seed_committed_spend,
)
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement, bind_spend_counter_keys
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.spend_counters import (
    SPEND_COUNTERS_TARGET,
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
from litellm.types.proxy.carried_budget_state import EntitySpendCounterKeys

_UNTYPED_VALUE: Final = TypeAdapter(object)


async def increment_spend_counters_batched(
    token: str | None,
    team_id: str | None,
    user_id: str | None,
    response_cost: float | None,
    org_id: str | None,
    budget_reservation: dict | None,
    end_user_id: str | None,
    tags: list[str] | None,
    request_started_at: datetime | None,
    model_access_groups: Sequence[str] | None,
    project_id: str | None = None,
    committed_spend_scope: EntitySpendCounterKeys | None = None,
    windows_from_ledger: bool = False,
    customer_counters: Sequence[BudgetCounter] = (),
) -> None:
    """Runs inside one spend counter batch: the reservation reconcile and the warm checks share a single MGET, and
    the reconcile adjustments go out in the same INCRBYFLOAT pipeline as the counter increments.
    """
    from litellm.proxy import proxy_server

    committed_counter_keys: Final = (
        tuple(counter_key for counter_key in committed_spend_scope.tracked() if has_committed_twin(counter_key))
        if committed_spend_scope is not None and response_cost is not None and response_cost > 0
        else ()
    )
    bind_spend_counter_keys(frozenset(committed_spend_keys(committed_counter_keys)))
    reservation_update: Final = await _reconcile_budget_reservation_for_counter_update(
        budget_reservation=budget_reservation,
        response_cost=response_cost,
    )
    reserved_counter_keys: Final = reservation_update.reserved_counter_keys

    if response_cost is None or response_cost == 0:
        await proxy_server._apply_spend_counter_increments(pending=reservation_update.pending)  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state
        stamp_budget_reservation_actual_cost(budget_reservation=budget_reservation, actual_cost=response_cost)
        if budget_reservation is not None:
            budget_reservation["finalized"] = True
        return

    cost: Final[float] = response_cost

    async def _resolved_window_increment(
        entity_type: Litellm_EntityType,
        entity_id: str,
        window_counter: str,
        duration: str,
        window_start: datetime | None,
        window_reset_at: datetime | str | None,
    ) -> PendingSpendIncrement | None:
        pending_window: Final = (
            await _prepare_window_spend_counter_increment(
                counter_key=window_counter,
                entity_type=(
                    entity_type.value.capitalize()
                    if entity_type in (Litellm_EntityType.KEY, Litellm_EntityType.TEAM, Litellm_EntityType.USER)
                    else entity_type.value
                ),
                entity_id=entity_id,
                window_duration=duration,
                window_start=window_start,
                increment=cost,
            )
            if window_counter not in reserved_counter_keys
            else None
        )
        if not windows_from_ledger:
            await _enqueue_window_spend_row_update(
                entity_type=entity_type,
                entity_id=entity_id,
                reset_at=window_reset_at,
                window_duration=duration,
                window_start=window_start,
                increment=cost,
                request_started_at=request_started_at,
            )
        return pending_window

    async def _window_increment(
        entity_type: Litellm_EntityType, entity_id: str, window: object
    ) -> PendingSpendIncrement | None:
        duration: Final = (
            window["budget_duration"] if isinstance(window, dict) else getattr(window, "budget_duration", None)
        )
        window_start: Final = get_budget_window_start(window)
        duration_name: Final = TypeAdapter(str).validate_python(duration)
        if window_start is None and is_scheduled_duration(duration_name):
            return None
        duration_text: Final = window_counter_name(duration_name, window_start)
        window_reset_at: Final = (
            window.get("reset_at") if isinstance(window, dict) else getattr(window, "reset_at", None)
        )
        window_counter: Final = (
            key_window_counter(entity_id, duration_text)
            if entity_type is Litellm_EntityType.KEY
            else user_window_counter(entity_id, duration_text)
            if entity_type is Litellm_EntityType.USER
            else team_window_counter(entity_id, duration_text)
        )
        return await _resolved_window_increment(
            entity_type, entity_id, window_counter, duration_text, window_start, window_reset_at
        )

    async def _customer_windows() -> tuple[PendingSpendIncrement | BaseException, ...]:
        pending: Final = await asyncio.gather(
            *(
                _resolved_window_increment(
                    Litellm_EntityType(counter.entity_type),
                    counter.entity_id,
                    counter.counter_key,
                    counter.window_duration,
                    counter.window_start,
                    counter.reset_at,
                )
                for counter in customer_counters
                if counter.window_duration is not None
            ),
            return_exceptions=True,
        )
        return tuple(item for item in pending if item is not None)

    async def _window_increments(
        entity_type: Litellm_EntityType, entity_id: str, budget_limits: object
    ) -> tuple[PendingSpendIncrement | BaseException, ...]:
        if not isinstance(budget_limits, (list, tuple)):
            return ()
        window_pending: Final = await asyncio.gather(
            *(_window_increment(entity_type, entity_id, window) for window in budget_limits), return_exceptions=True
        )
        return tuple(item for item in window_pending if item is not None)

    async def _cached_window_increments(
        entity_type: Litellm_EntityType, entity_id: str, cache_key: AuthObject
    ) -> tuple[_SpendScopeItem, ...]:
        cached: Final[object] = _UNTYPED_VALUE.validate_python(
            await proxy_server.user_api_key_cache.async_get_cache(key=cache_key)
        )
        if cached is None:
            return (_UncachedBudgetWindows(entity_type=entity_type, entity_id=entity_id),)
        return await _window_increments(entity_type, entity_id, proxy_server._cached_budget_limits(cached))  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state

    async def _stored_window_increments(
        uncached: _UncachedBudgetWindows,
    ) -> tuple[PendingSpendIncrement | BaseException, ...]:
        return await _window_increments(
            uncached.entity_type,
            uncached.entity_id,
            await proxy_server._stored_budget_limits(uncached.entity_type, uncached.entity_id),  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state
        )

    async def _key_scope(key_token: str) -> tuple[_SpendScopeItem, ...]:
        # key_token arrives pre-hashed from metadata["user_api_key"] (auth flow
        # hashes raw "sk-..." keys before they reach the callback). The
        # startswith("sk-") check is a safety net matching update_cache —
        # if a raw key somehow arrives, hash it; otherwise use as-is to
        # avoid double-hashing (budget checks read valid_token.token which
        # is single-hashed).
        hashed_token: Final = (
            proxy_server.hash_token(token=key_token)
            if isinstance(key_token, str) and key_token.startswith("sk-")
            else key_token
        )
        key_counter_key: Final = key_counter(hashed_token)
        key_pending: Final[tuple[PendingSpendIncrement, ...]] = (
            ()
            if key_counter_key in reserved_counter_keys
            else (
                await _prepare_spend_counter_increment(
                    counter_key=key_counter_key,
                    source_cache_key=AuthObject.virtual_key(hashed_token),
                    increment=cost,
                ),
            )
        )

        return key_pending + await _cached_window_increments(
            Litellm_EntityType.KEY, hashed_token, AuthObject.virtual_key(hashed_token)
        )

    async def _team_scope(scope_team_id: str) -> tuple[_SpendScopeItem, ...]:
        team_counter_key: Final = team_counter(scope_team_id)
        team_pending: Final[tuple[PendingSpendIncrement, ...]] = (
            ()
            if team_counter_key in reserved_counter_keys
            else (
                await _prepare_spend_counter_increment(
                    counter_key=team_counter_key,
                    source_cache_key=AuthObject.team(scope_team_id),
                    increment=cost,
                ),
            )
        )

        return team_pending + await _cached_window_increments(
            Litellm_EntityType.TEAM, scope_team_id, AuthObject.team(scope_team_id)
        )

    async def _team_member_scope(
        scope_user_id: str, scope_team_id: str
    ) -> tuple[PendingSpendIncrement | BaseException, ...]:
        team_member_counter_key: Final = team_member_counter(scope_user_id, scope_team_id)
        if team_member_counter_key in reserved_counter_keys:
            return ()
        return (
            await _prepare_spend_counter_increment(
                counter_key=team_member_counter_key,
                source_cache_key=AuthObject.team_membership_reservation(user_id=scope_user_id, team_id=scope_team_id),
                increment=cost,
            ),
        )

    async def _user_scope(scope_user_id: str) -> tuple[_SpendScopeItem, ...]:
        user_counter_key: Final = user_counter(scope_user_id)
        pending: Final = (
            ()
            if user_counter_key in reserved_counter_keys
            else (
                await _prepare_spend_counter_increment(
                    counter_key=user_counter_key,
                    source_cache_key=AuthObject.user(scope_user_id),
                    increment=cost,
                ),
            )
        )
        return pending + await _cached_window_increments(
            Litellm_EntityType.USER, scope_user_id, AuthObject.user(scope_user_id)
        )

    scope_coros: Final = tuple(
        coro
        for coro in (
            _customer_windows() if customer_counters else None,
            _key_scope(token) if token is not None else None,
            _team_scope(team_id) if team_id is not None else None,
            _team_member_scope(user_id, team_id) if user_id is not None and team_id is not None else None,
            _user_scope(user_id) if user_id is not None else None,
            _prepare_end_user_and_tag_spend_increments(
                end_user_id=end_user_id,
                tags=tags,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if end_user_id is not None or tags is not None
            else None,
            _prepare_model_access_group_spend_increments(
                model_access_groups=model_access_groups,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if model_access_groups
            else None,
            _prepare_org_spend_increment(
                org_id=org_id,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if org_id is not None
            else None,
            _prepare_project_spend_increment(
                project_id=project_id,
                response_cost=cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            if project_id is not None
            else None,
            _prepare_committed_spend_increments(counter_keys=committed_counter_keys, increment=cost)
            if committed_counter_keys
            else None,
        )
        if coro is not None
    )

    # return_exceptions so a failing scope does not leave its siblings running
    # as orphaned tasks that race the caller's reservation-counter invalidation;
    # all scopes settle, then the first error propagates as before.
    scope_items: Final = tuple(_flatten_spend_scopes(await asyncio.gather(*scope_coros, return_exceptions=True)))
    uncached_windows: Final = tuple(item for item in scope_items if isinstance(item, _UncachedBudgetWindows))
    await proxy_server._apply_spend_counter_increments(  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state
        pending=reservation_update.pending
        + tuple(item for item in scope_items if isinstance(item, PendingSpendIncrement))
    )
    stored_window_items: Final = (
        tuple(
            _flatten_spend_scopes(
                await asyncio.gather(
                    *(_stored_window_increments(uncached) for uncached in uncached_windows), return_exceptions=True
                )
            )
        )
        if uncached_windows
        else ()
    )
    stored_window_pending: Final = tuple(
        item for item in stored_window_items if isinstance(item, PendingSpendIncrement)
    )
    if stored_window_pending:
        await proxy_server._apply_spend_counter_increments(pending=stored_window_pending)  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state
    stamp_budget_reservation_actual_cost(budget_reservation=budget_reservation, actual_cost=response_cost)
    scope_errors: Final = tuple(item for item in scope_items + stored_window_items if isinstance(item, BaseException))
    if scope_errors:
        raise scope_errors[0]

    if budget_reservation is not None:
        budget_reservation["finalized"] = True


@dataclass(frozen=True, slots=True)
class _UncachedBudgetWindows:
    entity_type: Litellm_EntityType
    entity_id: str


_SpendScopeItem: TypeAlias = PendingSpendIncrement | BaseException | _UncachedBudgetWindows


def _flatten_spend_scopes(
    scope_results: Sequence[tuple[_SpendScopeItem, ...] | BaseException],
) -> Iterator[_SpendScopeItem]:
    for scope in scope_results:
        if isinstance(scope, BaseException):
            yield scope
            continue
        yield from scope


@dataclass(frozen=True, slots=True)
class _ReservationCounterUpdate:
    """The reserved counters the direct increment must skip, and the adjustments that settle them on the actual
    cost, still to be written; both empty when the reservation could not be reconciled and was dropped.
    """

    reserved_counter_keys: frozenset[str] = frozenset()
    pending: tuple[PendingSpendIncrement, ...] = ()


async def _reconcile_budget_reservation_for_counter_update(
    budget_reservation: dict[str, object] | None,
    response_cost: float | None,
) -> _ReservationCounterUpdate:
    from litellm.proxy import proxy_server

    if budget_reservation is None or budget_reservation.get("finalized") is True:
        return _ReservationCounterUpdate()

    from litellm.proxy.spend_tracking.budget_reservation import (
        get_reserved_counter_keys,
        invalidate_budget_reservation_counters,
        reconcile_budget_reservation,
    )

    reserved_counter_keys: Final = get_reserved_counter_keys(budget_reservation=budget_reservation)
    try:
        pending: Final = await reconcile_budget_reservation(
            budget_reservation=budget_reservation,
            actual_cost=response_cost or 0.0,
            finalize=False,
            apply_consistent=False,
        )
    except Exception:
        proxy_server.verbose_proxy_logger.warning(
            "Failed to reconcile budget reservation after persisted spend; invalidating reserved counters and falling back to direct increment",
            exc_info=True,
        )
        try:
            await invalidate_budget_reservation_counters(budget_reservation=budget_reservation)
        except Exception:
            proxy_server.verbose_proxy_logger.exception(
                "Failed to invalidate reserved counters after reservation reconciliation failed"
            )
        return _ReservationCounterUpdate()
    return _ReservationCounterUpdate(reserved_counter_keys=frozenset(reserved_counter_keys), pending=pending)


async def _committed_spend_seeded(counter_key: str) -> bool:
    from litellm.proxy import proxy_server

    return await proxy_server._is_spend_counter_cache_warm(  # pyright: ignore[reportPrivateUsage]  # the frozen ledger-off step reads proxy_server state
        committed_spend_key(counter_key)
    ) or await seed_committed_spend(
        spend_counter_store=proxy_server.spend_counter_store,
        prisma_client=proxy_server.prisma_client,
        counter_key=counter_key,
    )


async def _prepare_committed_spend_increments(
    counter_keys: tuple[str, ...], increment: float
) -> tuple[PendingSpendIncrement, ...]:
    seeded: Final = await asyncio.gather(*(_committed_spend_seeded(counter_key) for counter_key in counter_keys))
    return tuple(
        PendingSpendIncrement(counter_key=committed_spend_key(counter_key), increment=increment)
        for counter_key, is_seeded in zip(counter_keys, seeded)
        if is_seeded
    )


async def _prepare_end_user_and_tag_spend_increments(
    end_user_id: str | None,
    tags: list[str] | None,
    response_cost: float,
    reserved_counter_keys: AbstractSet[str],
) -> tuple[PendingSpendIncrement | BaseException, ...]:
    unique_tags: Final = (
        tuple(dict.fromkeys(tag for tag in tags if tag and isinstance(tag, str))) if tags is not None else ()
    )
    results: Final = await asyncio.gather(
        *(
            coro
            for coro in (
                _prepare_unreserved_spend_counter_increment(
                    counter_key=end_user_counter(end_user_id),
                    source_cache_key=AuthObject.end_user(end_user_id),
                    increment=response_cost,
                    reserved_counter_keys=reserved_counter_keys,
                )
                if end_user_id is not None
                else None,
                *(
                    _prepare_unreserved_spend_counter_increment(
                        counter_key=tag_counter(tag_name),
                        source_cache_key=AuthObject.tag(tag_name),
                        increment=response_cost,
                        reserved_counter_keys=reserved_counter_keys,
                    )
                    for tag_name in unique_tags
                ),
            )
            if coro is not None
        ),
        return_exceptions=True,
    )
    return tuple(item for item in results if item is not None)


async def _prepare_model_access_group_spend_increments(
    model_access_groups: Sequence[object],
    response_cost: float,
    reserved_counter_keys: AbstractSet[str],
) -> tuple[PendingSpendIncrement | BaseException, ...]:
    """Charge the model access groups that authorized this request.

    Without this the counter auth reads is written only by the reservation path, so
    ``disable_budget_reservation`` would leave ``_model_access_group_max_budget_check`` enforcing
    against the DB row's spend, which lags by up to the cache TTL.

    Typed ``object`` rather than ``str`` because the names reach the cost callback out of request
    metadata, which the coercion upstream filters to a list but not to strings. A non-string that
    slipped through would build a counter key nothing else ever reads.
    """
    unique_groups: Final = tuple(
        dict.fromkeys(group for group in model_access_groups if group and isinstance(group, str))
    )
    results: Final = await asyncio.gather(
        *(
            _prepare_unreserved_spend_counter_increment(
                counter_key=model_access_group_counter(group),
                source_cache_key=AuthObject.model_access_group(group),
                increment=response_cost,
                reserved_counter_keys=reserved_counter_keys,
            )
            for group in unique_groups
        ),
        return_exceptions=True,
    )
    return tuple(item for item in results if item is not None)


async def _prepare_org_spend_increment(
    org_id: str | None,
    response_cost: float,
    reserved_counter_keys: AbstractSet[str],
) -> tuple[PendingSpendIncrement, ...]:
    if org_id is None:
        return ()

    pending: Final = await _prepare_unreserved_spend_counter_increment(
        counter_key=org_counter(org_id),
        source_cache_key=(AuthObject.org_with_budget(org_id), AuthObject.org(org_id)),
        increment=response_cost,
        reserved_counter_keys=reserved_counter_keys,
    )
    return (pending,) if pending is not None else ()


async def _prepare_project_spend_increment(
    project_id: str | None,
    response_cost: float,
    reserved_counter_keys: AbstractSet[str],
) -> tuple[PendingSpendIncrement, ...]:
    if project_id is None:
        return ()

    pending: Final = await _prepare_unreserved_spend_counter_increment(
        counter_key=project_counter(project_id),
        source_cache_key=AuthObject.project(project_id),
        increment=response_cost,
        reserved_counter_keys=reserved_counter_keys,
    )
    return (pending,) if pending is not None else ()


async def _prepare_unreserved_spend_counter_increment(
    counter_key: str,
    source_cache_key: AuthObject | Sequence[AuthObject],
    increment: float,
    reserved_counter_keys: AbstractSet[str],
) -> PendingSpendIncrement | None:
    if counter_key in reserved_counter_keys:
        return None

    return await _prepare_spend_counter_increment(
        counter_key=counter_key,
        source_cache_key=source_cache_key,
        increment=increment,
    )


async def _prepare_spend_counter_increment(
    counter_key: str,
    source_cache_key: AuthObject | Sequence[AuthObject],
    increment: float,
) -> PendingSpendIncrement:
    """
    Initialize counter from the authoritative DB spend value if not yet
    set, then return the pending increment for the caller to apply in one
    pipelined Redis call.

    On first access per pod:
    1. Check spend_counter_store (Redis, then this worker's memory)
    2. If not found, reseed from the DB via `SpendCounterReseed.coalesced`.
       Falls back to the cached object's `.spend` via user_api_key_cache
       only if prisma is unavailable, since that value can lag the flusher.
    3. Seed counter via async_increment_cache (not async_set_cache) to avoid a
       check-then-set race: if two pods cold-start simultaneously, both may see
       the counter as absent and seed it. Using increment means the worst case
       is over-counting (conservative, blocks slightly early) rather than
       under-counting (would allow overspend).
    4. Increment is returned for the caller to apply via pipeline
    """
    from litellm.proxy import proxy_server

    with service_target(SPEND_COUNTERS_TARGET):
        await proxy_server.ensure_spend_counter_initialized(
            counter_key=counter_key,
            source_cache_key=source_cache_key,
        )
    return PendingSpendIncrement(counter_key=counter_key, increment=increment)


async def _enqueue_window_spend_row_update(
    entity_type: Litellm_EntityType,
    entity_id: str,
    reset_at: datetime | str | None,
    window_duration: str,
    window_start: datetime | None,
    increment: float,
    request_started_at: datetime | None,
) -> None:
    """Queue this request's cost against the LiteLLM_BudgetWindowSpend row for
    the window, so enforcement can read a maintained total instead of
    aggregating LiteLLM_SpendLogs.

    request_started_at is this request's LiteLLM_SpendLogs startTime; the flush
    stops the one-time seed there so a request its increment already covers is
    not counted twice.

    Enqueued even when the cache increment was skipped for a reserved counter:
    the reservation only pre-charged the counter, and the row still owes the
    actual cost.

    Windows with no reset_at slide with wall clock, so their window_start moves
    on every request and no single row can represent them. Those are left to
    the read path's LiteLLM_SpendLogs fallback rather than rewritten per
    request.
    """
    from litellm.proxy import proxy_server

    if window_start is None or not reset_at:
        return
    try:
        await proxy_server.proxy_logging_obj.db_spend_update_writer.window_spend_update_queue.add_update(
            build_window_spend_transaction(
                entity_type=entity_type.value,
                entity_id=entity_id,
                window_duration=window_duration,
                window_start=window_start,
                spend=increment,
                started_at=request_started_at,
                reset_at=reset_at,
            )
        )
    except Exception as e:  # noqa: BLE001  # spend tracking must never fail the cost callback
        proxy_server.verbose_proxy_logger.debug(
            "Unable to enqueue budget window spend update for %s=%s window=%s: %s",
            entity_type.value,
            entity_id,
            window_duration,
            e,
        )


async def _prepare_window_spend_counter_increment(
    counter_key: str,
    entity_type: str,
    entity_id: str,
    window_duration: str | None,
    window_start: datetime | None,
    increment: float,
) -> PendingSpendIncrement | None:
    from litellm.proxy import proxy_server

    if window_start is None:
        proxy_server.verbose_proxy_logger.warning(
            "Skipping spend counter increment for invalid budget window %s",
            counter_key,
        )
        return None

    with service_target(SPEND_COUNTERS_TARGET):
        initialized: Final = await proxy_server.ensure_window_spend_counter_initialized(
            counter_key=counter_key,
            entity_type=entity_type,
            entity_id=entity_id,
            window_duration=window_duration,
            window_start=window_start,
        )
    if initialized is False:
        return None
    return PendingSpendIncrement(counter_key=counter_key, increment=increment)
