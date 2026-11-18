"""Baseline entity spend keeps the budget period it was incurred in.

A reset moves ``budget_reset_at`` and the current ``spend`` column on to the next
window. A charge buffered before that reset still names the old period, so the
writer can leave the new window alone. ``total_spend`` on a key has no window
and still takes the charge. A buffered amount with no period is legacy and is
applied the way it always was.
"""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import groupby
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import BaselineSpendCharge, Litellm_EntityType, SpendUpdateQueueItem


class AccountingBudgetPeriod(TypedDict):
    table: ReadOnly[str]
    entity_id: ReadOnly[str]
    budget_reset_at: ReadOnly[str]


_REMAINDER_TOLERANCE: Final = 1e-9

BASELINE_SPEND_FIELDS: Final = MappingProxyType(
    {
        "LiteLLM_VerificationToken": "key_list_transactions",
        "LiteLLM_UserTable": "user_list_transactions",
        "LiteLLM_TeamTable": "team_list_transactions",
    }
)
_QUEUE_FIELD_BY_ENTITY: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: "key_list_transactions",
        Litellm_EntityType.USER: "user_list_transactions",
        Litellm_EntityType.TEAM: "team_list_transactions",
    }
)
_PERIOD_METADATA_FIELDS: Final = (
    ("LiteLLM_VerificationToken", "api_key", "user_api_key_budget_reset_at"),
    ("LiteLLM_UserTable", "user", "user_api_key_user_budget_reset_at"),
    ("LiteLLM_TeamTable", "team_id", "user_api_key_team_budget_reset_at"),
)
_ACCEPTED_FIELDS: Final = frozenset(BASELINE_SPEND_FIELDS.values())


def canonical_budget_reset_at(value: object) -> str | None:
    """Naive UTC timestamp at microsecond precision, the form the reset SQL compares."""
    parsed: Final = _as_datetime(value)
    if parsed is None:
        return None
    aware: Final = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    return aware.replace(tzinfo=None).isoformat(timespec="microseconds")


def budget_reset_at_for_where(canonical: str) -> datetime:
    return datetime.fromisoformat(canonical)


def spend_queue_group_key(update: SpendUpdateQueueItem) -> str:
    period: Final = canonical_budget_reset_at(update.get("budget_reset_at"))
    return f"{update.get('entity_type')}:{update.get('entity_id')}:{period or ''}"


def request_budget_metadata(kwargs: Mapping[str, object] | None) -> Mapping[str, object]:
    if kwargs is None:
        return MappingProxyType({})
    params: Final = kwargs.get("litellm_params")
    if not isinstance(params, Mapping):
        return MappingProxyType({})
    metadata: Final = params.get("metadata")
    if not isinstance(metadata, Mapping):
        return MappingProxyType({})
    return metadata


def accounting_budget_periods(
    *,
    api_key: str | None,
    user_id: str | None,
    team_id: str | None,
    metadata: Mapping[str, object],
) -> tuple[AccountingBudgetPeriod, ...]:
    identities: Final = {"api_key": api_key, "user": user_id, "team_id": team_id}
    return tuple(
        AccountingBudgetPeriod(table=table, entity_id=entity_id, budget_reset_at=period)
        for table, identity_key, metadata_key in _PERIOD_METADATA_FIELDS
        if isinstance(entity_id := identities[identity_key], str)
        and (period := canonical_budget_reset_at(metadata.get(metadata_key))) is not None
    )


def baseline_charges_from_queue_items(updates: Sequence[SpendUpdateQueueItem]) -> tuple[BaselineSpendCharge, ...]:
    return summed_baseline_charges(
        tuple(
            BaselineSpendCharge(
                field=field,
                entity_id=update.get("entity_id") or "",
                amount=float(update.get("response_cost") or 0),
                budget_reset_at=period,
            )
            for update in updates
            if (field := _QUEUE_FIELD_BY_ENTITY.get(update.get("entity_type"))) is not None
            and (period := canonical_budget_reset_at(update.get("budget_reset_at"))) is not None
        )
    )


def read_baseline_spend(transactions: Mapping[str, object]) -> tuple[BaselineSpendCharge, ...]:
    raw: Final = transactions.get("baseline_spend")
    if not isinstance(raw, (list, tuple)):
        return ()
    return summed_baseline_charges(tuple(charge for item in raw if (charge := _as_charge(item)) is not None))


def summed_baseline_charges(charges: Sequence[BaselineSpendCharge]) -> tuple[BaselineSpendCharge, ...]:
    ordered: Final = tuple(sorted(charges, key=_charge_key))
    return tuple(_summed_charge(tuple(group)) for _, group in groupby(ordered, key=_charge_key))


@dataclass(frozen=True, slots=True)
class AppliedBaselineSpend:
    entity_id: str
    amount: float
    budget_reset_at: str | None


def applied_baseline_spend(
    amounts: Mapping[str, float],
    charges: Sequence[BaselineSpendCharge],
    field: str,
) -> tuple[AppliedBaselineSpend, ...]:
    """Per-period charges for one table, plus any amount the periods do not cover.

    The float map still holds the summed amount so an older reader keeps charging
    something. This split is what a current reader applies instead of that sum.
    """
    field_charges: Final = tuple(charge for charge in charges if charge["field"] == field)
    return tuple(_applied_rows(amounts, field_charges))


def spend_queue_items(
    entity_type: Litellm_EntityType,
    amounts: Mapping[str, float],
    charges: Sequence[BaselineSpendCharge],
    field: str,
) -> tuple[SpendUpdateQueueItem, ...]:
    return tuple(_queue_item(entity_type, item) for item in applied_baseline_spend(amounts, charges, field))


def _applied_rows(
    amounts: Mapping[str, float], charges: Sequence[BaselineSpendCharge]
) -> Iterator[AppliedBaselineSpend]:
    for entity_id, amount in sorted(amounts.items()):
        yield from _applied_for_entity(
            entity_id, amount, tuple(charge for charge in charges if charge["entity_id"] == entity_id)
        )


def _applied_for_entity(
    entity_id: str, amount: float, charges: Sequence[BaselineSpendCharge]
) -> tuple[AppliedBaselineSpend, ...]:
    if not charges:
        return (AppliedBaselineSpend(entity_id, amount, None),)
    ordered: Final = tuple(sorted(charges, key=lambda charge: charge["budget_reset_at"]))
    covered: Final = math.fsum(charge["amount"] for charge in ordered)
    windowed: Final = tuple(
        AppliedBaselineSpend(entity_id, charge["amount"], charge["budget_reset_at"]) for charge in ordered
    )
    remainder: Final = amount - covered
    if remainder <= _REMAINDER_TOLERANCE:
        return windowed
    return windowed + (AppliedBaselineSpend(entity_id, remainder, None),)


def _queue_item(entity_type: Litellm_EntityType, item: AppliedBaselineSpend) -> SpendUpdateQueueItem:
    if item.budget_reset_at is None:
        return SpendUpdateQueueItem(entity_type=entity_type, entity_id=item.entity_id, response_cost=item.amount)
    return SpendUpdateQueueItem(
        entity_type=entity_type,
        entity_id=item.entity_id,
        response_cost=item.amount,
        budget_reset_at=item.budget_reset_at,
    )


def _charge_key(charge: BaselineSpendCharge) -> tuple[str, str, str]:
    return (charge["field"], charge["entity_id"], charge["budget_reset_at"])


def _summed_charge(charges: tuple[BaselineSpendCharge, ...]) -> BaselineSpendCharge:
    first: Final = charges[0]
    return BaselineSpendCharge(
        field=first["field"],
        entity_id=first["entity_id"],
        amount=math.fsum(charge["amount"] for charge in charges),
        budget_reset_at=first["budget_reset_at"],
    )


def _as_charge(value: object) -> BaselineSpendCharge | None:
    if not isinstance(value, Mapping):
        return None
    field: Final = value.get("field")
    entity_id: Final = value.get("entity_id")
    amount: Final = value.get("amount")
    period: Final = canonical_budget_reset_at(value.get("budget_reset_at"))
    if not isinstance(field, str) or field not in _ACCEPTED_FIELDS or not isinstance(entity_id, str):
        return None
    if not isinstance(amount, (int, float)) or isinstance(amount, bool) or period is None:
        return None
    return BaselineSpendCharge(field=field, entity_id=entity_id, amount=float(amount), budget_reset_at=period)


def _as_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
