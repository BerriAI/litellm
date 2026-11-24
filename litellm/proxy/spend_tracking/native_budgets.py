from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Final, Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic import Field as PydanticField

import litellm
from litellm.integrations.custom_logger_v2.derivations import (
    budget_reservation,
    mapping,
    write_budget_reservation,
)
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_counter_batch import PendingSpendIncrement, spend_counter_batch_scope
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.integrations.custom_logger_v2 import BudgetPlan, BudgetReservation, coerce_record
from litellm.types.proxy.native_budgets import BudgetCheck, NativeSettlement

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeBudgets


def new_reservation_id() -> str:
    return uuid.uuid4().hex


class _BudgetExceededFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    current_cost: float
    max_budget: float
    message: str
    entity_type: str | None
    entity_id: str | None


class _DetailFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    detail: str


class _MessageFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str


class _Held(BaseModel):
    model_config = ConfigDict(frozen=True)

    admission: Literal["held"]
    reservation: Mapping[str, object]


class _Skipped(BaseModel):
    model_config = ConfigDict(frozen=True)

    admission: Literal["skipped"]


class _Refused(BaseModel):
    model_config = ConfigDict(frozen=True)

    admission: Literal["refused"]
    kind: Literal["budget_exceeded", "budget_unavailable", "budget_reservation_failed"]
    fields: Mapping[str, object]


_ADMISSION: Final[TypeAdapter[_Held | _Skipped | _Refused]] = TypeAdapter(
    Annotated[_Held | _Skipped | _Refused, PydanticField(discriminator="admission")]
)
_SETTLEMENT: Final = TypeAdapter(NativeSettlement)
_CHECK: Final[TypeAdapter[BudgetCheck]] = TypeAdapter(BudgetCheck)
_HELD_ENTRIES: Final = TypeAdapter(Iterable[object])


def _refusal(refused: _Refused) -> Exception:
    match refused.kind:
        case "budget_exceeded":
            exceeded: Final = _BudgetExceededFields.model_validate(refused.fields)
            return litellm.BudgetExceededError(
                current_cost=exceeded.current_cost,
                max_budget=exceeded.max_budget,
                message=exceeded.message,
                entity_type=exceeded.entity_type,
                entity_id=exceeded.entity_id,
            )
        case "budget_unavailable":
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_DetailFields.model_validate(refused.fields).detail,
            )
        case "budget_reservation_failed":
            return ProxyException(
                message="Authentication Error, " + _MessageFields.model_validate(refused.fields).message,
                type=ProxyErrorTypes.auth_error,
                param=None,
                code=status.HTTP_401_UNAUTHORIZED,
            )


def _with_window_reset(entry: object, window_resets: Mapping[str, str]) -> object:
    held: Final = mapping(entry)
    if held is None:
        return entry
    reset: Final = window_resets.get(str(held.get("counter_key")))
    return entry if reset is None else {**held, "window_reset_at": reset}


def _with_window_resets(value: Mapping[str, object], window_resets: Mapping[str, str]) -> Mapping[str, object]:
    entries: Final = value.get("entries")
    if not window_resets or not isinstance(entries, list | tuple):
        return value
    return {
        **value,
        "entries": [_with_window_reset(entry, window_resets) for entry in _HELD_ENTRIES.validate_python(entries)],
    }


def _planned_counters(plan: BudgetPlan) -> tuple[tuple[str, float], ...]:
    return tuple(
        (counter.counter_key, float(counter.fallback_spend or 0.0))
        for counter in plan.counters or ()
        if counter.counter_key is not None
    )


async def observed_counter_spend(plan: BudgetPlan) -> tuple[Mapping[str, object], ...]:
    from litellm.proxy.proxy_server import get_current_spend, spend_counter_store

    counters: Final = _planned_counters(plan)
    with spend_counter_batch_scope(spend_counter_store, counter_keys=frozenset(key for key, _ in counters)):
        return tuple(
            [
                {
                    "counter_key": counter_key,
                    "spend": await get_current_spend(counter_key=counter_key, fallback_spend=fallback_spend),
                }
                for counter_key, fallback_spend in counters
            ]
        )


_engine: Final[list[tuple[SpendCounterStore, NativeBudgets]]] = []  # mutable-ok: the engine for the proxy's store


def native_budgets() -> NativeBudgets | None:
    from litellm.proxy.proxy_server import spend_counter_store

    if _engine and _engine[0][0] is spend_counter_store:
        return _engine[0][1]
    from litellm.rust_bridge._native import NativeBudgets, NativeSpendCounters

    handle: Final = spend_counter_store.handle
    if not isinstance(handle, NativeSpendCounters):
        return None
    budgets: Final = NativeBudgets(handle)
    _engine[:] = [(spend_counter_store, budgets)]
    return budgets


async def reserve_native(
    plan: Mapping[str, object],
    identity: UserAPIKeyAuth,
    window_resets: Mapping[str, str],
) -> dict[str, object] | None:  # mutable-ok: settlement stamps the reservation it returns
    budgets: Final = native_budgets()
    if budgets is None:
        return None
    record: Final = coerce_record(BudgetPlan, plan)
    admission: Final = _ADMISSION.validate_python(await budgets.reserve(record, await observed_counter_spend(record)))
    if isinstance(admission, _Refused):
        raise _refusal(admission)
    if isinstance(admission, _Skipped):
        return None
    held: Final = _with_window_resets(admission.reservation, window_resets)
    reservation: Final[dict[str, object]] = {  # mutable-ok: settlement stamps the reservation it returns
        "reserved_cost": 0.0,
        "entries": [],
        "finalized": False,
        "callback_bound": False,
        "input_cost": 0.0,
        "input_tokens": None,
    }
    identity.budget_reservation = reservation
    write_budget_reservation(HookArgs(auth=identity), held)
    from litellm.proxy.spend_tracking.budget_reservation import start_reservation_lease_renewal

    start_reservation_lease_renewal(
        budget_reservation=reservation,
        counter_keys=frozenset(
            key for key in map(_held_counter_key, _held_entries(reservation.get("entries") or ())) if key is not None
        ),
        request_task=asyncio.current_task(),
    )
    return reservation


@dataclass(frozen=True, slots=True)
class _Reserved:
    budget_reservation: object


def _reservation_record(reservation: object) -> BudgetReservation | None:
    return budget_reservation(HookArgs(auth=_Reserved(reservation)))


def _window_reset(entry: object) -> str | None:
    held: Final = mapping(entry)
    reset: Final = None if held is None else held.get("window_reset_at")
    if isinstance(reset, datetime):
        return reset.isoformat()
    return reset if isinstance(reset, str) else None


def _held_entries(raw: object) -> tuple[object, ...]:
    try:
        return tuple(_HELD_ENTRIES.validate_python(raw))
    except ValidationError:
        raise TypeError(f"'{type(raw).__name__}' object is not iterable") from None


def _held_counter_key(entry: object) -> str | None:
    held: Final = mapping(entry)
    counter_key: Final = None if held is None else held.get("counter_key")
    return counter_key if isinstance(counter_key, str) else None


def reservation_inputs(
    reservation: Mapping[str, object] | None,
) -> tuple[BudgetReservation | None, tuple[str | None, ...]]:
    """The reservation as the typed record the Rust engine reads, and each entry's window reset."""
    if not reservation:
        return None, ()
    return _reservation_record(reservation), tuple(map(_window_reset, _held_entries(reservation.get("entries") or ())))


async def _observed_spend(counter_key: str | None) -> float | None:
    if counter_key is None:
        return None
    from litellm.proxy.proxy_server import read_spend_counter_cache_value

    try:
        value, _ = await read_spend_counter_cache_value(counter_key=counter_key)
    except (TypeError, ValueError):
        return None
    return value


async def settle_native_reservation(
    reservation: Mapping[str, object],
    actual_cost: float,
    apply_consistent: bool = True,
) -> tuple[PendingSpendIncrement, ...]:
    budgets: Final = native_budgets()
    if budgets is None:
        return ()
    entries: Final = _held_entries(reservation.get("entries") or ())
    record: Final = _reservation_record(reservation)
    if record is None:
        return ()
    from litellm.proxy.proxy_server import spend_counter_store
    from litellm.proxy.spend_tracking.budget_reservation import reseed_reserved_counter

    counter_keys: Final = tuple(map(_held_counter_key, entries))
    with spend_counter_batch_scope(
        spend_counter_store, counter_keys=frozenset(key for key in counter_keys if key is not None)
    ):
        observed: Final = tuple([await _observed_spend(counter_key) for counter_key in counter_keys])
        settled: Final = _SETTLEMENT.validate_python(
            await budgets.settle(record, tuple(map(_window_reset, entries)), observed, actual_cost, apply_consistent)
        )
        for reseed in settled.reseed:
            await reseed_reserved_counter(reseed.counter_key, reseed.actual_cost)
    for entry, target in zip(entries, settled.adjustments):
        if target is not None and isinstance(entry, dict):
            entry["applied_adjustment"] = target
    return tuple(
        PendingSpendIncrement(counter_key=pending.counter_key, increment=pending.increment)
        for pending in settled.pending
    )


class BudgetDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: Literal["skip", "pass", "alert", "throttle", "refuse"]
    pct: int | float | None = None
    current_cost: int | float | None = None
    max_budget: int | float | None = None
    message: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None

    @property
    def skipped(self) -> bool:
        return self.decision == "skip"

    @property
    def alerts(self) -> bool:
        return self.decision == "alert"

    @property
    def refused(self) -> bool:
        return self.decision in ("refuse", "throttle")

    def error(self) -> litellm.BudgetExceededError:
        return litellm.BudgetExceededError(
            current_cost=self.current_cost if self.current_cost is not None else 0.0,
            max_budget=self.max_budget if self.max_budget is not None else 0.0,
            message=self.message,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
        )


def native_budget_decision(check: Mapping[str, object]) -> BudgetDecision:
    from litellm.rust_bridge._native import budget_check_decision

    return BudgetDecision.model_validate(budget_check_decision(_CHECK.validate_python(check)))
