from collections.abc import Iterable
from itertools import chain, groupby
from typing import TYPE_CHECKING, Final
from urllib.parse import quote

from litellm.accounting.projection import Contribution, project
from litellm.repositories.accounting_delivery_repository import DeliveryProjection


def _destination(contribution: Contribution) -> tuple[str, tuple[tuple[str, str], ...], str]:
    return (
        contribution.table,
        contribution.key,
        (dict(contribution.values).get("window_start") or "")
        if contribution.table == "LiteLLM_BudgetWindowSpend"
        else "",
    )


def _delta(delivery: DeliveryProjection) -> tuple[Contribution, ...]:
    return (
        *project(delivery.row),
        *(
            Contribution(
                item.table, item.key, tuple((column, -value) for column, value in item.increments), item.values
            )
            for item in (() if delivery.previous_payload is None else project(delivery.previous_payload))
        ),
    )


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
    ordered: Final = sorted(chain.from_iterable(_delta(delivery) for delivery in deliveries), key=_destination)
    return tuple(_sum(tuple(group)) for _, group in groupby(ordered, key=_destination))


def entity_transactions(contributions: tuple[Contribution, ...]) -> "DBSpendUpdateTransactions":
    return {
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
    }


def _entity_totals(contributions: tuple[Contribution, ...], table: str) -> tuple[tuple[str, float], ...]:
    return tuple(
        (_entity_key(item), float(dict(item.increments)["spend"])) for item in contributions if item.table == table
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
