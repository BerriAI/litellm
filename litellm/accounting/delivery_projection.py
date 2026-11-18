import math
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, groupby
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

from litellm.accounting.projection import Contribution, project
from litellm.proxy._types import BaselineSpendCharge
from litellm.proxy.db.baseline_spend_period import BASELINE_SPEND_FIELDS, summed_baseline_charges
from litellm.repositories.accounting_delivery_repository import DeliveryProjection


def _destination(contribution: Contribution) -> tuple[str, tuple[tuple[str, str], ...], str, str]:
    values: Final = dict(contribution.values)
    window: Final = values.get("window_start") or "" if contribution.table == "LiteLLM_BudgetWindowSpend" else ""
    period: Final = values.get("budget_reset_at") or "" if contribution.table in BASELINE_SPEND_FIELDS else ""
    return (contribution.table, contribution.key, window, period)


def _sum(contributions: tuple[Contribution, ...]) -> Contribution:
    first: Final = contributions[0]
    increments: Final = tuple(
        (column, sum(dict(item.increments).get(column, 0) for item in contributions))
        for column in dict.fromkeys(
            column for column, _ in chain.from_iterable(item.increments for item in contributions)
        )
    )
    starts: Final = tuple(
        value
        for column, value in chain.from_iterable(item.values for item in contributions)
        if column == "started_at" and value
    )
    values: Final = tuple(
        (column, min(starts) if column == "started_at" and starts else value) for column, value in first.values
    )
    return Contribution(first.table, first.key, increments, values)


def project_deliveries(deliveries: Iterable[DeliveryProjection]) -> tuple[Contribution, ...]:
    ordered: Final = sorted(chain.from_iterable(project(delivery.row) for delivery in deliveries), key=_destination)
    return tuple(_sum(tuple(group)) for _, group in groupby(ordered, key=_destination))


def entity_transactions(contributions: tuple[Contribution, ...]) -> "DBSpendUpdateTransactions":
    charges: Final = _baseline_charges(contributions)
    transactions: Final = {
        "user_list_transactions": dict(_entity_totals(contributions, "LiteLLM_UserTable")),
        "end_user_list_transactions": dict(_entity_totals(contributions, "LiteLLM_EndUserTable")),
        "key_list_transactions": dict(_entity_totals(contributions, "LiteLLM_VerificationToken")),
        "team_list_transactions": dict(_entity_totals(contributions, "LiteLLM_TeamTable")),
        "team_member_list_transactions": dict(_entity_totals(contributions, "LiteLLM_TeamMembership")),
        "org_list_transactions": dict(_entity_totals(contributions, "LiteLLM_OrganizationTable")),
        "org_member_list_transactions": dict(_entity_totals(contributions, "LiteLLM_OrganizationMembership")),
        "project_list_transactions": dict(_entity_totals(contributions, "LiteLLM_ProjectTable")),
        "tag_list_transactions": dict(_entity_totals(contributions, "LiteLLM_TagTable")),
        "agent_list_transactions": dict(_entity_totals(contributions, "LiteLLM_AgentsTable")),
        "model_access_group_list_transactions": dict(
            _entity_totals(contributions, "LiteLLM_ModelAccessGroupBudgetTable")
        ),
        "customer_org_list_transactions": dict(_entity_totals(contributions, "LiteLLM_CustomerOrganizationTable")),
        "customer_team_list_transactions": dict(_entity_totals(contributions, "LiteLLM_CustomerTeamTable")),
        "customer_member_list_transactions": dict(_entity_totals(contributions, "LiteLLM_CustomerMemberTable")),
        **({"baseline_spend": list(charges)} if charges else {}),
    }
    return transactions


def _entity_totals(contributions: tuple[Contribution, ...], table: str) -> tuple[tuple[str, float], ...]:
    """Sum every charge for an entity, including charges that belong to different budget periods."""
    return _summed_by_entity(
        tuple(
            (_entity_key(item), float(dict(item.increments)["spend"])) for item in contributions if item.table == table
        )
    )


def _summed_by_entity(rows: Sequence[tuple[str, float]]) -> tuple[tuple[str, float], ...]:
    ordered: Final = tuple(sorted(rows, key=lambda row: row[0]))
    return tuple(
        _summed_entity(entity_id, tuple(group)) for entity_id, group in groupby(ordered, key=lambda row: row[0])
    )


def _summed_entity(entity_id: str, rows: Sequence[tuple[str, float]]) -> tuple[str, float]:
    return entity_id, math.fsum(amount for _, amount in rows)


def _baseline_charges(contributions: Sequence[Contribution]) -> tuple[BaselineSpendCharge, ...]:
    return summed_baseline_charges(tuple(_baseline_charge_rows(contributions)))


def _baseline_charge_rows(contributions: Sequence[Contribution]) -> Iterator[BaselineSpendCharge]:
    for item in contributions:
        field: Final = BASELINE_SPEND_FIELDS.get(item.table)
        period: Final = dict(item.values).get("budget_reset_at")
        if field is None or not isinstance(period, str) or not period:
            continue
        yield BaselineSpendCharge(
            field=field,
            entity_id=_entity_key(item),
            amount=float(dict(item.increments)["spend"]),
            budget_reset_at=period,
        )


def _entity_key(item: Contribution) -> str:
    fields: Final = dict(item.key)
    if item.table == "LiteLLM_TeamMembership":
        return f"team_id::{fields['team_id']}::user_id::{fields['user_id']}"
    if item.table == "LiteLLM_OrganizationMembership":
        return f"org_id::{quote(fields['organization_id'], safe='')}::user_id::{quote(fields['user_id'], safe='')}"
    return item.key[0][1]


if TYPE_CHECKING:
    from litellm.proxy._types import DBSpendUpdateTransactions
