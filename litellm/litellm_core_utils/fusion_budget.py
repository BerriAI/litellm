"""Synchronize Fusion's hidden provider calls with its shared budget reservation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Final
from uuid import uuid4

from litellm.constants import (
    FUSION_BUDGET_ACTIVE_KEY,
    FUSION_BUDGET_CALL_ID_METADATA_KEY,
    FUSION_BUDGET_PENDING_CALL_IDS_KEY,
    FUSION_BUDGET_UNPRICED_CALL_IDS_KEY,
)

_BUDGET_RESERVATION_METADATA_KEY: Final = "user_api_key_budget_reservation"
_CALLBACK_POLL_INTERVAL_SECONDS: Final = 0.01
_CALLBACK_WAIT_TIMEOUT_SECONDS: Final = 5.0


def _fusion_budget_reservation(
    metadata: Mapping[str, object],
) -> dict[str, object] | None:  # mutable-ok: callers coordinate through one shared request ledger
    reservation: Final = metadata.get(_BUDGET_RESERVATION_METADATA_KEY)
    if not isinstance(reservation, dict) or reservation.get(FUSION_BUDGET_ACTIVE_KEY) is not True:
        return None
    return reservation


def register_fusion_budget_call(metadata: dict[str, object]) -> None:  # mutable-ok: request metadata owns token
    """Register one hidden logical call before it is dispatched to a provider."""
    reservation: Final = _fusion_budget_reservation(metadata)
    if reservation is None:
        return
    token: Final = uuid4().hex
    pending: Final = reservation.setdefault(
        FUSION_BUDGET_PENDING_CALL_IDS_KEY,
        [],  # mutable-ok: request-scoped pending-call ledger
    )
    if not isinstance(pending, list):
        return
    pending.append(token)
    metadata[FUSION_BUDGET_CALL_ID_METADATA_KEY] = token  # rebind-ok: caller passes mutable request metadata


def complete_fusion_budget_call(
    metadata: Mapping[str, object],
    *,
    cost_known: bool,
) -> None:
    """Finish a registered call once its cost was recorded, or mark it conservatively unknown."""
    reservation: Final = _fusion_budget_reservation(metadata)
    token: Final = metadata.get(FUSION_BUDGET_CALL_ID_METADATA_KEY)
    if reservation is None or not isinstance(token, str):
        return
    pending: Final = reservation.get(FUSION_BUDGET_PENDING_CALL_IDS_KEY)
    if not isinstance(pending, list):
        return
    try:
        pending.remove(token)
    except ValueError:
        return
    if cost_known:
        return
    unpriced: Final = reservation.setdefault(
        FUSION_BUDGET_UNPRICED_CALL_IDS_KEY,
        [],  # mutable-ok: request-scoped conservative-cost ledger
    )
    if isinstance(unpriced, list):
        unpriced.append(token)


def cancel_fusion_budget_call(metadata: Mapping[str, object]) -> None:
    """Finish a call deliberately cancelled by Fusion without marking the whole request unpriced.

    Panel and analyst timeouts actively cancel their in-flight child call. They
    are different from a completed provider call whose cost callback went
    missing: the latter must retain the conservative full-reservation fallback,
    while the former must not turn one timed-out advisory member into a charge
    for every possible Fusion call.
    """
    complete_fusion_budget_call(metadata, cost_known=True)


async def wait_for_fusion_budget_calls(
    metadata: Mapping[str, object],
    *,
    timeout_seconds: float = _CALLBACK_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Wait for registered hidden costs, falling back to the reserved maximum on timeout."""
    reservation: Final = _fusion_budget_reservation(metadata)
    if reservation is None:
        return
    loop: Final = asyncio.get_running_loop()
    deadline: Final = loop.time() + timeout_seconds
    while True:
        pending = reservation.get(  # rebind-ok: poll the shared ledger until every callback reports
            FUSION_BUDGET_PENDING_CALL_IDS_KEY
        )
        if not isinstance(pending, list) or not pending:
            return
        if loop.time() >= deadline:
            unresolved = tuple(  # rebind-ok: timeout snapshot is local to this polling iteration
                token for token in pending if isinstance(token, str)
            )
            pending.clear()
            unpriced = reservation.setdefault(  # rebind-ok: timeout ledger is read only in this iteration
                FUSION_BUDGET_UNPRICED_CALL_IDS_KEY,
                [],  # mutable-ok: timeout converts unresolved calls to a conservative charge
            )
            if isinstance(unpriced, list):
                unpriced.extend(token for token in unresolved if token not in unpriced)
            return
        await asyncio.sleep(_CALLBACK_POLL_INTERVAL_SECONDS)


def fusion_budget_reconciliation_cost(
    budget_reservation: Mapping[str, object],
    known_cost: float,
) -> float:
    """Use actual cost normally and the pre-call maximum if any hidden cost stayed unknown."""
    unpriced: Final = budget_reservation.get(FUSION_BUDGET_UNPRICED_CALL_IDS_KEY)
    if not isinstance(unpriced, list) or not unpriced:
        return known_cost
    reserved_cost_value: Final = budget_reservation.get("reserved_cost")
    reserved_cost: Final = (
        float(reserved_cost_value)
        if isinstance(reserved_cost_value, (int, float)) and not isinstance(reserved_cost_value, bool)
        else 0.0
    )
    return max(known_cost, reserved_cost)
