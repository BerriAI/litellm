from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.accounting.customer_identity import resolve_customer_delivery
from litellm.accounting.delivery_projection import entity_transactions, project_deliveries
from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction


@dataclass(frozen=True, slots=True)
class _Directory:
    unavailable: bool = False

    async def query(self, query: str, *args: object) -> tuple[Mapping[str, object], ...]:
        if self.unavailable:
            raise OSError("writer database unavailable")
        if query.startswith("SELECT m.customer_member_id"):
            return tuple(
                {
                    "customer_org_id": f"org-{name}",
                    "customer_org_alias": f"alias-{name}",
                    "customer_team_id": args[1] or f"team-{name}",
                    "customer_member_id": f"member-{name}",
                }
                for name in ("a", "b")
                if args[0] == f"member-{name}" and args[1] in (None, f"team-{name}", f"second-team-{name}")
            )
        return tuple(
            {"customer_org_id": f"org-{name}", "customer_org_alias": f"alias-{name}"}
            for name in ("a", "b")
            if args == (f"org-{name}" if "WHERE customer_org_id" in query else f"alias-{name}",)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "alias,member,binding,expected",
    (
        ("alias-a", "member-a", None, ("org-a", "team-a", "member-a")),
        (None, "member-a", None, ("org-a", "team-a", "member-a")),
        ("alias-a", None, None, ("org-a", None, None)),
        ("alias-b", "member-a", "org-a", ("org-a", "team-a", "member-a")),
        ("alias-b", "member-b", "org-a", ("org-a", None, None)),
        ("unknown", None, "org-a", ("org-a", None, None)),
        ("alias-b", "member-a", None, (None, None, None)),
        ("unknown", None, None, (None, None, None)),
        (None, None, None, (None, None, None)),
    ),
)
async def test_recovered_customer_spend_honors_the_authenticated_keys_customer_org(
    alias: str | None, member: str | None, binding: str | None, expected: tuple[str | None, ...]
) -> None:
    transaction: Final = MagicMock(spec=LedgerTransaction)
    transaction.query_raw = AsyncMock(side_effect=_Directory().query)
    delivery: Final = DeliveryProjection(
        {
            "request_id": "outage-request",
            "spend": 0.75,
            "metadata": {
                "accounting_entities": {"api_key": "verified-key"},
                "user_api_key_asserted_customer": {
                    "customer_org_alias": alias,
                    "customer_member_id": member,
                    "key_customer_org_id": binding,
                },
            },
        }
    )
    resolved: Final = await resolve_customer_delivery(transaction, delivery)
    transactions: Final = entity_transactions(project_deliveries((resolved,)))
    assert tuple(transactions[f"customer_{scope}_list_transactions"] for scope in ("org", "team", "member")) == tuple(
        {entity: 0.75} if entity is not None else {} for entity in expected
    )
    assert transactions["key_list_transactions"] == {"verified-key": 0.75}
    assert "accounting_entities" in delivery.row["metadata"]


@pytest.mark.asyncio
async def test_a_request_without_an_assertion_needs_no_customer_directory_read() -> None:
    transaction: Final = MagicMock(spec=LedgerTransaction)
    transaction.query_raw = AsyncMock(side_effect=OSError("must not query"))
    delivery: Final = DeliveryProjection({"spend": 0.75, "metadata": {}})
    assert await resolve_customer_delivery(transaction, delivery) is delivery
    transaction.query_raw.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("team", (None, "second-team-a", "foreign-team"))
async def test_recovery_keeps_the_selected_customer_team_within_the_members_customer_org(team: str | None) -> None:
    transaction: Final = MagicMock(spec=LedgerTransaction)
    transaction.query_raw = AsyncMock(side_effect=_Directory().query)
    delivery: Final = DeliveryProjection(
        {
            "spend": 0.75,
            "metadata": {
                "user_api_key_asserted_customer": {
                    "customer_member_id": "member-a",
                    "customer_team_id": team,
                    "key_customer_org_id": "org-a",
                }
            },
        }
    )
    resolved: Final = await resolve_customer_delivery(transaction, delivery)
    transactions: Final = entity_transactions(project_deliveries((resolved,)))
    assert transactions["customer_org_list_transactions"] == {"org-a": 0.75}
    assert transactions["customer_team_list_transactions"] == (
        {} if team == "foreign-team" else {team or "team-a": 0.75}
    )
    assert transactions["customer_member_list_transactions"] == ({} if team == "foreign-team" else {"member-a": 0.75})


@pytest.mark.asyncio
async def test_a_customer_resolution_outage_propagates_so_the_whole_delivery_retries() -> None:
    transaction: Final = MagicMock(spec=LedgerTransaction)
    transaction.query_raw = AsyncMock(side_effect=_Directory(unavailable=True).query)
    delivery: Final = DeliveryProjection(
        {"metadata": {"user_api_key_asserted_customer": {"customer_member_id": "member-a"}}}
    )
    with pytest.raises(OSError, match="writer database unavailable"):
        await resolve_customer_delivery(transaction, delivery)
