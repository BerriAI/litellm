"""The customer hierarchy and app Python repositories over fake Rust repositories."""

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import BaseModel

from litellm.models.app import LiteLLM_AppCustomerMemberTable, LiteLLM_AppCustomerOrgTable, LiteLLM_AppTable
from litellm.models.customer_hierarchy import (
    LiteLLM_CustomerMemberTable,
    LiteLLM_CustomerOrganizationTable,
    LiteLLM_CustomerTeamTable,
)
from litellm.proxy.management_helpers.customer_hierarchy import customer_row_reset_at
from litellm.repositories import rust_database
from litellm.repositories.app_customer_member_repository import AppCustomerMemberRepository
from litellm.repositories.app_customer_organization_repository import AppCustomerOrganizationRepository
from litellm.repositories.app_repository import AppRepository
from litellm.repositories.customer_member_repository import CustomerMemberRepository
from litellm.repositories.customer_organization_repository import CustomerOrganizationRepository
from litellm.repositories.customer_team_repository import CustomerTeamRepository
from litellm.rust_bridge._native import (
    RustAppCustomerMemberRepository,
    RustAppCustomerOrganizationRepository,
    RustAppRepository,
    RustCustomerMemberRepository,
    RustCustomerOrganizationRepository,
    RustCustomerTeamRepository,
)

_AUDIT: Final = {
    "created_at": "2026-11-01T00:00:00+00:00",
    "created_by": "creator",
    "updated_at": "2026-11-02T00:00:00+00:00",
    "updated_by": "updater",
}
_BUDGET: Final = {
    "budget_id": "budget-1",
    "max_budget": 10.0,
    "budget_duration": "1mo",
    "budget_limits": [{"budget_duration": "12mo", "max_budget": 100.0}],
    "budget_reset_at": "2026-12-01T00:00:00+00:00",
    **_AUDIT,
}


class _FakeRustRepository:
    """Answers every read with one JSON row, recording the arguments of each read."""

    def __init__(self, row: Mapping[str, object] | None) -> None:
        self._row = row
        self.lookups: list[tuple[str, ...]] = []

    async def _answer(self, *args: str) -> str | None:
        self.lookups.append(args)
        return None if self._row is None else json.dumps(self._row)

    async def find_with_budget(self, entity_id: str) -> str | None:
        return await self._answer(entity_id)

    async def find_with_budgets(self, entity_id: str) -> str | None:
        return await self._answer(entity_id)

    async def find(self, *args: str) -> str | None:
        return await self._answer(*args)


Read = Callable[[], Awaitable[BaseModel | None]]


@dataclass(frozen=True)
class _Case:
    kind: object
    read: Read
    row: Mapping[str, object]
    lookup: tuple[str, ...]
    model: type[BaseModel]


_CASES: Final = (
    _Case(
        RustCustomerOrganizationRepository,
        lambda: CustomerOrganizationRepository(object()).find_with_budget("org-1"),
        {"customer_org_id": "org-1", **_AUDIT, "spend": 2.5, "budget_template_ids": ["t-1"], "budget": _BUDGET},
        ("org-1",),
        LiteLLM_CustomerOrganizationTable,
    ),
    _Case(
        RustCustomerTeamRepository,
        lambda: CustomerTeamRepository(object()).find_with_budgets("team-1"),
        {
            "customer_team_id": "team-1",
            **_AUDIT,
            "parent_org_id": "org-1",
            "spend": 4.5,
            "budget_override_expires_at": "2026-12-24T00:00:00+00:00",
            "budget": None,
            "budget_override": _BUDGET,
        },
        ("team-1",),
        LiteLLM_CustomerTeamTable,
    ),
    _Case(
        RustCustomerMemberRepository,
        lambda: CustomerMemberRepository(object()).find_with_budgets("member-1"),
        {
            "customer_member_id": "member-1",
            **_AUDIT,
            "parent_org_id": "org-1",
            "parent_team_ids": ["team-1"],
            "spend": 1.25,
            "budget": _BUDGET,
            "budget_override": None,
        },
        ("member-1",),
        LiteLLM_CustomerMemberTable,
    ),
    _Case(
        RustAppRepository,
        lambda: AppRepository(object()).find("app-1"),
        {
            "app_id": "app-1",
            "alias": "app",
            **_AUDIT,
            "spend": 6.5,
            "auto_create_customers": False,
            "customer_margin": 0.2,
        },
        ("app-1",),
        LiteLLM_AppTable,
    ),
    _Case(
        RustAppCustomerOrganizationRepository,
        lambda: AppCustomerOrganizationRepository(object()).find("app-1", "org-1"),
        {"app_id": "app-1", "customer_org_id": "org-1", "spend": 3.75},
        ("app-1", "org-1"),
        LiteLLM_AppCustomerOrgTable,
    ),
    _Case(
        RustAppCustomerMemberRepository,
        lambda: AppCustomerMemberRepository(object()).find("app-1", "member-1"),
        {"app_id": "app-1", "customer_member_id": "member-1", "spend": 0.5},
        ("app-1", "member-1"),
        LiteLLM_AppCustomerMemberTable,
    ),
)
_IDS: Final = ("customer_org", "customer_team", "customer_member", "app", "app_customer_org", "app_customer_member")


def _install(monkeypatch: pytest.MonkeyPatch, kind: object, fake: _FakeRustRepository | None) -> None:
    def repository(requested: object) -> _FakeRustRepository | None:
        return fake if requested is kind else None

    monkeypatch.setattr(rust_database, "repository", repository)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
async def test_decodes_the_rust_row(monkeypatch: pytest.MonkeyPatch, case: _Case) -> None:
    fake: Final = _FakeRustRepository(case.row)
    _install(monkeypatch, case.kind, fake)
    found: Final = await case.read()
    assert found == case.model.model_validate(case.row)
    assert fake.lookups == [case.lookup]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
async def test_missing_row_is_none(monkeypatch: pytest.MonkeyPatch, case: _Case) -> None:
    _install(monkeypatch, case.kind, _FakeRustRepository(None))
    assert await case.read() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
async def test_without_a_rust_database_raises_no_db_connected(monkeypatch: pytest.MonkeyPatch, case: _Case) -> None:
    _install(monkeypatch, case.kind, None)
    with pytest.raises(rust_database.NoDatabaseConnected):
        await case.read()


@pytest.mark.asyncio
async def test_customer_budgets_decode_with_their_reset_clock_and_concurrent_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(
        monkeypatch,
        RustCustomerTeamRepository,
        _FakeRustRepository({"customer_team_id": "t", "parent_org_id": "o", **_AUDIT, "budget": _BUDGET}),
    )
    team: Final = await CustomerTeamRepository(object()).find_with_budgets("t")
    assert team is not None and team.budget is not None
    assert team.budget.budget_reset_at == datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert team.budget.budget_limits is not None and team.budget.budget_limits[0].budget_duration == "12mo"
    assert customer_row_reset_at(team, datetime(2026, 11, 18, tzinfo=timezone.utc)) == datetime(
        2026, 12, 1, tzinfo=timezone.utc
    )
