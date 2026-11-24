"""The ledger's post-call spend counter step, run in Rust by ``NativeBudgets.charge_counters``.

Rust settles the customer hierarchy, reconciles the reservation, warms and seeds every counter, sends the increments
in one pipeline and stamps the reservation. The host here answers only what the proxy alone knows, a whole batch per
call: budget windows from the auth cache or their stored row, stored spend from the database, and the auth cache's
spend for a counter the database cannot answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import flush_post_call_redis_batches
from litellm.constants import AUTH_OBJECTS_TARGET
from litellm.proxy.auth.customer_spend import customer_spend_counter
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.redis.stores._route import post_call_request
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.spend_counters import END_USER_COUNTER_PREFIX
from litellm.rust_bridge.execution import Execution, drive
from litellm.types.integrations.slack_alerting import Litellm_EntityType
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth
from litellm.types.proxy.carried_budget_state import EntitySpendCounterKeys

if TYPE_CHECKING:
    from litellm.proxy.spend_tracking.budget_reservation import BudgetCounter


class _Request(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _WindowSource(_Request):
    entity_type: Literal["key", "team", "user"]
    entity_id: str
    cache_key: str

    def auth_object(self) -> AuthObject:
        match self.entity_type:
            case "key":
                return AuthObject.virtual_key(self.entity_id)
            case "team":
                return AuthObject.team(self.entity_id)
            case "user":
                return AuthObject.user(self.entity_id)
        assert_never(self.entity_type)


class _BudgetWindows(_Request):
    entities: tuple[_WindowSource, ...]
    stored: bool


class _EntitySpend(_Request):
    source: Literal["entity"]
    counter_key: str


class _WindowSpend(_Request):
    source: Literal["window"]
    entity_type: str
    entity_id: str
    window_duration: str | None
    window_start: datetime


class _StoredSpend(_Request):
    counters: tuple[Annotated[_EntitySpend | _WindowSpend, Field(discriminator="source")], ...]


class _NamedCacheSource(_Request):
    kind: Literal["key", "team", "user", "end_user", "tag", "model_access_group", "org_with_budget", "org", "project"]
    id: str

    def auth_object(self) -> AuthObject:
        match self.kind:
            case "key":
                return AuthObject.virtual_key(self.id)
            case "team":
                return AuthObject.team(self.id)
            case "user":
                return AuthObject.user(self.id)
            case "end_user":
                return AuthObject.end_user(self.id)
            case "tag":
                return AuthObject.tag(self.id)
            case "model_access_group":
                return AuthObject.model_access_group(self.id)
            case "org_with_budget":
                return AuthObject.org_with_budget(self.id)
            case "org":
                return AuthObject.org(self.id)
            case "project":
                return AuthObject.project(self.id)
        assert_never(self.kind)


class _MembershipIds(_Request):
    user_id: str
    team_id: str


class _MembershipCacheSource(_Request):
    kind: Literal["team_membership_reservation"]
    id: _MembershipIds

    def auth_object(self) -> AuthObject:
        return AuthObject.team_membership_reservation(user_id=self.id.user_id, team_id=self.id.team_id)


class _CachedSpend(_Request):
    sources: tuple[tuple[_NamedCacheSource | _MembershipCacheSource, ...], ...]


_BudgetWindow: TypeAlias = tuple[str, str | None]
_LIMITS: Final = TypeAdapter(tuple[object, ...])
_FIELDS: Final = TypeAdapter(Mapping[str, object])
_ANY_VALUE: Final = TypeAdapter(object)


def _window_field(window: object, name: str) -> object:
    try:
        return _FIELDS.validate_python(window).get(name)
    except ValidationError:
        return getattr(window, name, None)


def _budget_window(window: object) -> _BudgetWindow | None:
    """The window's duration and the start of the window this request falls in, as the proxy computes it."""
    from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start

    duration: Final = _window_field(window, "budget_duration")
    if not isinstance(duration, str):
        return None
    start: Final = get_budget_window_start(window)
    return duration, None if start is None else start.isoformat()


def _budget_windows(budget_limits: object) -> tuple[_BudgetWindow, ...]:
    if not isinstance(budget_limits, (list, tuple)):
        return ()
    return tuple(window for window in map(_budget_window, _LIMITS.validate_python(budget_limits)) if window is not None)


async def _stored_spend(counter: _EntitySpend | _WindowSpend) -> float | None:
    from litellm.proxy.proxy_server import prisma_client

    match counter:
        case _EntitySpend(counter_key=counter_key) if counter_key.startswith(END_USER_COUNTER_PREFIX):
            return await SpendCounterReseed.end_user_from_db(prisma_client=prisma_client, counter_key=counter_key)
        case _EntitySpend(counter_key=counter_key):
            return await SpendCounterReseed.from_db(prisma_client=prisma_client, counter_key=counter_key)
        case _WindowSpend():
            return await SpendCounterReseed.window_from_db(
                prisma_client=prisma_client,
                entity_type=counter.entity_type,
                entity_id=counter.entity_id,
                window_duration=counter.window_duration,
                window_start=counter.window_start,
            )


class _CounterHost:
    """The proxy's half of the counter step: the auth cache and the database, which only the proxy holds."""

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def budget_windows(self, request: object) -> tuple[tuple[_BudgetWindow, ...] | None, ...]:
        from litellm.proxy.proxy_server import (
            _cached_budget_limits,  # pyright: ignore[reportPrivateUsage]  # the proxy's cached budget-limit reader
            _stored_budget_limits,  # pyright: ignore[reportPrivateUsage]  # the proxy's stored budget-limit reader
            user_api_key_cache,
        )

        windows: Final = _BudgetWindows.model_validate(request)

        async def of(source: _WindowSource) -> tuple[_BudgetWindow, ...] | None:
            if windows.stored:
                return _budget_windows(
                    await _stored_budget_limits(Litellm_EntityType(source.entity_type), source.entity_id)
                )
            cached: Final[object] = (
                await user_api_key_cache.get_team_fields(source.auth_object(), ("budget_limits",))
                if source.entity_type == "team"
                else _ANY_VALUE.validate_python(await user_api_key_cache.async_get_cache(key=source.auth_object()))
            )
            return None if cached is None else _budget_windows(_cached_budget_limits(cached))

        return tuple(await asyncio.gather(*map(of, windows.entities)))

    async def stored_spend(self, request: object) -> tuple[float | None, ...]:
        return tuple(await asyncio.gather(*map(_stored_spend, _StoredSpend.model_validate(request).counters)))

    async def cached_spend(self, request: object) -> tuple[float, ...]:
        from litellm.proxy.proxy_server import (
            _get_source_cache_base_spend,  # pyright: ignore[reportPrivateUsage]  # the proxy's source-cache spend reader
        )

        return tuple(
            await asyncio.gather(
                *(
                    _get_source_cache_base_spend(source_cache_key=tuple(source.auth_object() for source in sources))
                    for sources in _CachedSpend.model_validate(request).sources
                )
            )
        )


_HOST: Final = _CounterHost()


def _streamless(execution: Execution, head: object) -> object:
    raise RuntimeError("the spend counter step cannot open a stream")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _names(values: Iterable[object] | None) -> tuple[str | None, ...] | None:
    return None if values is None else tuple(_text(value) for value in values)


def _customer_scopes(customer_hierarchy: object) -> tuple[Mapping[str, object], ...]:
    if customer_hierarchy is None:
        return ()
    try:
        hierarchy: Final = CustomerHierarchyAuth.model_validate(customer_hierarchy)
    except ValidationError:
        return ()
    return tuple({"counter_key": customer_spend_counter(scope), "spend": scope.spend} for scope in hierarchy.scopes)


async def charge_spend_counters(
    token: str | None,
    team_id: str | None,
    user_id: str | None,
    response_cost: float | None,
    org_id: str | None,
    budget_reservation: dict[str, object] | None,  # mutable-ok: the step stamps the caller's shared reservation
    end_user_id: str | None,
    tags: Sequence[object] | None,
    model_access_groups: Sequence[object] | None,
    project_id: str | None,
    committed_spend_scope: EntitySpendCounterKeys | None,
    customer_counters: Sequence[BudgetCounter],
    customer_hierarchy: object,
) -> None:
    """The ledger's whole post-call counter step for one charge, run by the Rust engine on the proxy's store."""
    from litellm.proxy.proxy_server import spend_counter_store
    from litellm.proxy.spend_tracking.native_budgets import native_budgets, reservation_inputs

    budgets: Final = native_budgets()
    if budgets is None:
        raise RuntimeError("the ledger's spend counter step runs on the Rust spend counter store")
    record, window_resets = reservation_inputs(budget_reservation)
    step: Final[Mapping[str, object]] = {
        "token": _text(token),
        "team_id": _text(team_id),
        "user_id": _text(user_id),
        "org_id": _text(org_id),
        "end_user_id": _text(end_user_id),
        "project_id": _text(project_id),
        "tags": _names(tags),
        "model_access_groups": _names(model_access_groups),
        "response_cost": response_cost,
        "committed_tracked": () if committed_spend_scope is None else tuple(committed_spend_scope.tracked()),
        "customer_windows": tuple(
            {
                "counter_key": counter.counter_key,
                "entity_type": counter.entity_type,
                "entity_id": counter.entity_id,
                "window_duration": counter.window_duration,
                "window_start": None if counter.window_start is None else counter.window_start.isoformat(),
            }
            for counter in customer_counters
        ),
        "customer_scopes": _customer_scopes(customer_hierarchy),
        "window_resets": window_resets,
    }
    _ = await drive(
        budgets.charge_counters(
            step,
            record,
            budget_reservation,
            post_call_request() if spend_counter_store.has_redis else None,
            _HOST,
        ),
        _streamless,
    )


async def charge_counter_steps(steps: Sequence[Mapping[str, object]]) -> None:
    """Runs counter steps the Rust spend log callback built, one after another, with this proxy answering their host
    calls, on the request's post-call batch, then sends that batch. A failed step is logged and the rest still run,
    as ``_raise_failure_spend_counters`` does per row."""
    from litellm.proxy.proxy_server import spend_counter_store
    from litellm.proxy.spend_tracking.native_budgets import native_budgets

    budgets: Final = native_budgets()
    if budgets is None:
        raise RuntimeError("the ledger's spend counter step runs on the Rust spend counter store")
    request: Final = post_call_request() if spend_counter_store.has_redis else None
    for step in steps:
        try:
            _ = await drive(budgets.charge_counters(step, None, None, request, _HOST), _streamless)
        except Exception:  # noqa: BLE001  # one row's counter failure must not keep the other rows' counters from landing
            verbose_proxy_logger.exception("Failed to raise spend counters for a failure spend row")
    await flush_post_call_redis_batches()
