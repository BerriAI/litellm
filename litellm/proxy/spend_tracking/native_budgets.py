from __future__ import annotations

import asyncio
import uuid
import weakref
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import budget_reservation, write_budget_reservation
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch, SetField
from litellm.integrations.custom_logger_v2.rust import (
    RustLogger,
    WireAbort,
    register_abort_error,
    rust_logger,
)
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.integrations.custom_logger_v2 import BudgetPlan, BudgetReservation, coerce_record
from litellm.types.proxy.native_budgets import BudgetCheck, NativeSettlement
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeBudgets

NATIVE_CALL_ID_KEY: Final = "native_call_id"


@dataclass(slots=True)
class _PendingRequest:
    reservation_id: str
    reservation: dict[str, object]  # mutable-ok: the shared reservation dict auth handed the request
    auth: UserAPIKeyAuth
    identity: UserAPIKeyAuth
    request_task: asyncio.Task[object] | None
    route: str
    request_data: Mapping[str, object]
    window_resets: Mapping[str, str]
    request: Request | None = None

    def hold(self, held: BudgetReservation | None) -> None:
        self.reservation[NATIVE_CALL_ID_KEY] = self.reservation_id
        from litellm.proxy.spend_tracking.budget_reservation import start_reservation_lease_renewal

        start_reservation_lease_renewal(
            budget_reservation=self.reservation,
            counter_keys=frozenset(
                entry.counter_key
                for entry in (held.entries or () if held is not None else ())
                if entry is not None and entry.counter_key is not None
            ),
            request_task=self.request_task,
        )


_pending: Final[dict[str, _PendingRequest]] = {}  # mutable-ok: by reservation id until pre_call claims the plan
_refusals: Final[weakref.WeakKeyDictionary[Exception, _PendingRequest]] = weakref.WeakKeyDictionary()


def new_reservation_id() -> str:
    return uuid.uuid4().hex


def pending_native_reservation(
    plan: Mapping[str, object],
    identity: UserAPIKeyAuth,
    route: str,
    request_data: Mapping[str, object],
    window_resets: Mapping[str, str],
) -> dict[str, object]:  # mutable-ok: settlement stamps the reservation it returns
    reservation_id: Final = plan.get("reservation_id")
    reservation: Final[dict[str, object]] = {  # mutable-ok: settlement stamps the reservation it returns
        "reserved_cost": 0.0,
        "entries": [],
        "finalized": False,
        "callback_bound": False,
        "input_cost": 0.0,
        "input_tokens": None,
    }
    request_task: Final = asyncio.current_task()
    if isinstance(reservation_id, str):
        _pending[reservation_id] = _PendingRequest(
            reservation_id=reservation_id,
            reservation=reservation,
            auth=identity,
            identity=identity.model_copy(),
            request_task=request_task,
            route=route,
            request_data={
                **request_data,
                **{
                    name: dict(bucket)
                    for name in ("metadata", "litellm_metadata")
                    if isinstance(bucket := request_data.get(name), Mapping)
                },
            },
            window_resets=window_resets,
        )
        if request_task is not None:
            request_task.add_done_callback(lambda _: _pending.pop(reservation_id, None))
    identity.bind_budget_reservation(reservation, coerce_record(BudgetPlan, plan))
    return reservation


_Admission: TypeAlias = "asyncio.Future[Patch | Abort | None]"
_admissions: Final[dict[int, _Admission]] = {}  # mutable-ok: in-flight admissions by reservation


def native_admission(reservation: Mapping[str, object] | None) -> _Admission | None:
    return None if reservation is None else _admissions.get(id(reservation))


async def await_native_admission(reservation: Mapping[str, object] | None) -> None:
    admission: Final = native_admission(reservation)
    if admission is None:
        return
    with suppress(Exception):
        _ = await asyncio.shield(admission)


_REFUSAL_TYPES: Final = (litellm.BudgetExceededError, HTTPException, ProxyException)


def bind_native_request(identity: UserAPIKeyAuth, request: Request) -> None:
    plan: Final = identity.budget_plan
    pending: Final = None if plan is None or plan.reservation_id is None else _pending.get(plan.reservation_id)
    if pending is not None and pending.reservation is identity.budget_reservation:
        pending.request = request


async def raise_native_admission_refusal(error: Exception) -> None:
    pending: Final = _refusals.pop(error, None) if isinstance(error, _REFUSAL_TYPES) else None
    if pending is None or pending.request is None:
        return
    from litellm.proxy.auth.auth_exception_handler import UserAPIKeyAuthExceptionHandler

    _ = await UserAPIKeyAuthExceptionHandler.handle_authentication_error(
        e=error,
        request=pending.request,
        request_data=pending.request_data,
        route=pending.route,
        parent_otel_span=pending.identity.parent_otel_span,
        api_key="",
        resolved_identity=pending.identity,
    )


_NATIVE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.acompletion,
        CallTypes.completion,
        CallTypes.atext_completion,
        CallTypes.text_completion,
        CallTypes.aembedding,
        CallTypes.embedding,
        CallTypes.anthropic_messages,
        CallTypes.aanthropic_messages,
        CallTypes.responses,
        CallTypes.aresponses,
    }
)


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


_SETTLEMENT: Final = TypeAdapter(NativeSettlement)
_CHECK: Final[TypeAdapter[BudgetCheck]] = TypeAdapter(BudgetCheck)


def native_budget_route(route: str) -> bool:
    call_types: Final = get_call_types_for_route(route) or ()
    return any(call_type in _NATIVE_CALL_TYPES for call_type in call_types)


def _budget_exceeded(wire: WireAbort) -> Exception:
    fields: Final = _BudgetExceededFields.model_validate(wire.error.fields if wire.error else {})
    return litellm.BudgetExceededError(
        current_cost=fields.current_cost,
        max_budget=fields.max_budget,
        message=fields.message,
        entity_type=fields.entity_type,
        entity_id=fields.entity_id,
    )


def _budget_unavailable(wire: WireAbort) -> Exception:
    fields: Final = _DetailFields.model_validate(wire.error.fields if wire.error else {})
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=fields.detail)


def _reservation_failed(wire: WireAbort) -> Exception:
    fields: Final = _MessageFields.model_validate(wire.error.fields if wire.error else {})
    return ProxyException(
        message="Authentication Error, " + fields.message,
        type=ProxyErrorTypes.auth_error,
        param=None,
        code=status.HTTP_401_UNAUTHORIZED,
    )


def _claim(view: object) -> _PendingRequest | None:
    plan: Final = getattr(view, Field.budget_plan.value, None)
    if not isinstance(plan, BudgetPlan) or plan.reservation_id is None:
        return None
    return _pending.pop(plan.reservation_id, None)


@dataclass(frozen=True, slots=True)
class _Reserved:
    budget_reservation: object


def _reservation_record(reservation: object) -> BudgetReservation | None:
    return budget_reservation(HookArgs(auth=_Reserved(reservation)))


def _with_window_reset(entry: object, window_resets: Mapping[str, str]) -> object:
    if not isinstance(entry, Mapping):
        return entry
    reset: Final = window_resets.get(str(entry.get("counter_key")))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # an entry the native wrote
    return entry if reset is None else {**entry, "window_reset_at": reset}  # pyright: ignore[reportUnknownVariableType]  # an entry the native wrote


def _with_window_resets(value: object, window_resets: Mapping[str, str]) -> object:
    if not window_resets or not isinstance(value, Mapping):
        return value
    entries: Final = value.get("entries")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the reservation the native wrote
    if not isinstance(entries, list | tuple):
        return value  # pyright: ignore[reportUnknownVariableType]  # the reservation the native wrote
    return {**value, "entries": [_with_window_reset(entry, window_resets) for entry in entries]}  # pyright: ignore[reportUnknownVariableType]  # the reservation the native wrote


def _stamped(outcome: Patch | Abort | None, window_resets: Mapping[str, str]) -> Patch | Abort | None:
    if not isinstance(outcome, Patch):
        return outcome
    return Patch(
        tuple(
            SetField(op.field, _with_window_resets(op.value, window_resets))
            if isinstance(op, SetField) and op.field is Field.budget_reservation
            else op
            for op in outcome.ops
        )
    )


def _reservation_write(outcome: Patch | Abort | None) -> object:
    if not isinstance(outcome, Patch):
        return None
    return next(
        (op.value for op in outcome.ops if isinstance(op, SetField) and op.field is Field.budget_reservation),
        None,
    )


def _write_abandoned(pending: _PendingRequest, admission: _Admission) -> None:
    if admission.cancelled() or admission.exception() is not None:
        return
    held: Final = _reservation_write(admission.result())
    if held is not None:
        write_budget_reservation(HookArgs(auth=pending.auth), held)


class NativeBudgetsLogger(RustLogger):
    budgets: NativeBudgets

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        pending: Final = _claim(view) if hook is Hook.pre_call else None
        if pending is None:
            return await super().run_rust(hook, reads, view, ctx)
        key: Final = id(pending.reservation)
        admission: Final = asyncio.ensure_future(self._admit(pending, hook, reads, view, ctx))
        _admissions[key] = admission
        admission.add_done_callback(lambda _: _admissions.pop(key, None))
        try:
            return await asyncio.shield(admission)
        except asyncio.CancelledError:
            admission.add_done_callback(partial(_write_abandoned, pending))
            raise

    async def _admit(
        self,
        pending: _PendingRequest,
        hook: Hook,
        reads: frozenset[Field],
        view: object,
        ctx: CallContext[object],
    ) -> Patch | Abort | None:
        outcome: Final = _stamped(await super().run_rust(hook, reads, view, ctx), pending.window_resets)
        held: Final = _reservation_write(outcome)
        if held is not None:
            pending.hold(_reservation_record(held))
        if isinstance(outcome, Abort) and outcome.exception is not None:
            _refusals[outcome.exception] = pending
        return outcome


class _CurrentSpend(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    counter_key: str
    fallback_spend: float


_CURRENT_SPEND: Final = TypeAdapter(_CurrentSpend)


class BudgetsHost:
    async def current_spend(self, request: object) -> float:
        call: Final = _CURRENT_SPEND.validate_python(request)
        from litellm.proxy.proxy_server import get_current_spend

        return await get_current_spend(counter_key=call.counter_key, fallback_spend=call.fallback_spend)


@dataclass(frozen=True, slots=True)
class _Owner:
    store: SpendCounterStore
    budgets: NativeBudgets
    logger: NativeBudgetsLogger


_installed: Final[list[_Owner]] = []  # mutable-ok: one process-wide native owner, set once at startup


def native_budgets() -> NativeBudgets | None:
    return _installed[0].budgets if _installed else None


def native_owner_reserves() -> bool:
    return bool(_installed) and any(callback is _installed[0].logger for callback in litellm.callbacks)


def install_native_budgets(store: SpendCounterStore, callbacks: list[object]) -> None:  # mutable-ok: litellm.callbacks
    if _installed and _installed[0].store is store:
        if not any(callback is _installed[0].logger for callback in callbacks):
            callbacks.insert(0, _installed[0].logger)
        return
    for stale in [callback for callback in callbacks if isinstance(callback, NativeBudgetsLogger)]:
        callbacks.remove(stale)
    _installed.clear()
    from litellm.rust_bridge._native import NativeBudgets, NativeSpendCounters

    handle: Final = store.handle
    if not isinstance(handle, NativeSpendCounters):
        return
    budgets: Final = NativeBudgets(handle)
    template: Final = rust_logger(budgets.callback())
    owner_type: Final = type(type(template).__name__, (NativeBudgetsLogger, type(template)), {"budgets": budgets})
    logger: Final = owner_type(template.rust, host=BudgetsHost())
    register_abort_error("budget_exceeded", _budget_exceeded)
    register_abort_error("budget_unavailable", _budget_unavailable)
    register_abort_error("budget_reservation_failed", _reservation_failed)
    callbacks.insert(0, logger)
    _installed.append(_Owner(store=store, budgets=budgets, logger=logger))


def _window_reset(entry: object) -> str | None:
    reset: Final = entry.get("window_reset_at") if isinstance(entry, Mapping) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # a held entry
    if isinstance(reset, datetime):
        return reset.isoformat()
    return reset if isinstance(reset, str) else None


async def settle_native_reservation(
    reservation: dict[str, object],  # mutable-ok: stamps each settled entry in place
    actual_cost: float,
) -> None:
    budgets: Final = native_budgets()
    if budgets is None or NATIVE_CALL_ID_KEY not in reservation:
        return
    entries: Final[tuple[object, ...]] = tuple(reservation.get("entries") or ())  # pyright: ignore[reportArgumentType]  # the held entries, iterated as budgets iterates them
    record: Final = _reservation_record(reservation)
    if record is None:
        return
    settled: Final = _SETTLEMENT.validate_python(
        await budgets.settle(record, tuple(map(_window_reset, entries)), actual_cost)
    )
    from litellm.proxy.spend_tracking.budget_reservation import reseed_reserved_counter

    for reseed in settled.reseed:
        await reseed_reserved_counter(reseed.counter_key, reseed.actual_cost)
    for entry, target in zip(entries, settled.adjustments):
        if target is not None and isinstance(entry, dict):
            entry["applied_adjustment"] = target


def is_native_reservation(reservation: Mapping[str, object] | None) -> bool:
    return reservation is not None and NATIVE_CALL_ID_KEY in reservation


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


def native_budget_decision(check: Mapping[str, object]) -> BudgetDecision | None:
    if native_budgets() is None:
        return None
    from litellm.rust_bridge._native import budget_check_decision

    return BudgetDecision.model_validate(budget_check_decision(_CHECK.validate_python(check)))
