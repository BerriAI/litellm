from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litellm.models.person_budget import PersonBudgetSubject
from litellm.proxy.auth.customer_hierarchy_auth import customer_member_cache_key
from litellm.proxy.auth.person_budget import person_budget_assignment_cache_key
from litellm.proxy.common_utils.timezone_utils import BudgetResetSettings
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.budget_assignments import (
    BudgetAssignment,
    BudgetAssignmentPatch,
    BudgetAssignments,
    get_budget_assignments,
    patch_budget_assignment,
    router,
)
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    ErrorCode,
    Failure,
    TransactionSource,
    invalidate_person_budget_members,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    MemberCreate,
    MemberPatch,
    OrganizationCreate,
    TeamCreate,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database


class Fixture:
    def __init__(self) -> None:
        self.db = Database()
        self.evictions: tuple[tuple[tuple[str, ...], str | None], ...] = ()
        self.cache = UserApiKeyCache()
        self.assignments = BudgetAssignments(
            cast(TransactionSource, self.db),
            lambda: NOW,
            self.invalidate,
            invalidate_members=partial(invalidate_person_budget_members, cache=self.cache),
        )
        self.hierarchy = CustomerHierarchy(
            cast(TransactionSource, self.db),
            lambda: NOW,
            self.invalidate_sources,
            invalidate_members=partial(invalidate_person_budget_members, cache=self.cache),
        )

    async def invalidate(self, budgets: tuple[str, ...], user: str | None = None) -> None:
        self.evictions = (*self.evictions, (budgets, user))

    async def invalidate_sources(self, budgets: tuple[str, ...]) -> None:
        await self.invalidate(budgets)

    async def create(self, subject: PersonBudgetSubject) -> BudgetAssignment:
        await self.db.litellm_budgettable.create(
            data={
                "budget_id": "profile",
                "is_profile": True,
                "budget_limits": (
                    {"budget_duration": "1mo", "max_budget": 10},
                    {"budget_duration": "12mo", "max_budget": 100},
                ),
            }
        )
        if subject == "user":
            await self.db.litellm_usertable.create(data={"user_id": "person", "budget_id": "profile"})
        else:
            await self.hierarchy.create_organization(OrganizationCreate(customer_org_id="org"), "admin")
            await self.hierarchy.create_team(TeamCreate(customer_team_id="team", parent_org_id="org"), "admin")
            created: Final = await self.hierarchy.create_member(
                MemberCreate(customer_member_id="person", parent_team_id="team", budget_id="profile"), "admin"
            )
            assert not isinstance(created, Failure)
        response: Final = await self.assignments.get(subject, "person")
        assert isinstance(response, BudgetAssignment)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", ("user", "customer_member"))
async def test_patch_preserves_other_duration_zero_and_null_clear(subject: PersonBudgetSubject) -> None:
    fixture: Final = Fixture()
    original: Final = await fixture.create(subject)
    saved: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": original.revision,
                "budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": 0}]},
                "budget_override_expires_at": NOW + timedelta(days=2),
            }
        ),
        "admin",
    )
    assert isinstance(saved, BudgetAssignment)
    assert {limit.budget_duration: limit.max_budget for limit in saved.effective_budget.budget_limits or ()} == {
        "1mo": 0,
        "12mo": 100,
    }
    assert saved.budget_id == "profile"
    assert saved.budget_override_id != saved.budget_id
    partial: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": saved.revision,
                "budget_override": {"cap_policy": "pause_until_reset"},
            }
        ),
        "admin",
    )
    assert isinstance(partial, BudgetAssignment)
    assert partial.effective_budget.budget_limits == saved.effective_budget.budget_limits
    assert partial.budget_override_expires_at == saved.budget_override_expires_at
    inherited: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": partial.revision,
                "budget_override": {"budget_limits": None},
            }
        ),
        "admin",
    )
    assert isinstance(inherited, BudgetAssignment)
    assert inherited.effective_budget.budget_limits == original.effective_budget.budget_limits
    cleared: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": inherited.revision,
                "budget_override": None,
            }
        ),
        "admin",
    )
    assert isinstance(cleared, BudgetAssignment)
    assert cleared.budget_override is None
    assert cleared.budget_override_id is None
    assert cleared.budget_override_expires_at is None
    assert saved.budget_override_id not in fixture.db.litellm_budgettable.rows
    assert (saved.budget_override_id,) == fixture.evictions[-1][0]
    assert fixture.evictions[-1][1] == ("person" if subject == "user" else None)


@pytest.mark.asyncio
async def test_stale_assignment_and_profile_revisions_fail_without_overwriting() -> None:
    fixture: Final = Fixture()
    original: Final = await fixture.create("user")
    first: Final = await patch_budget_assignment(
        "user",
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": original.revision,
                "budget_override": {"cap_policy": "pause_until_reset"},
            }
        ),
        "admin",
        fixture.assignments,
    )
    with pytest.raises(ManagementProblem) as stale:
        await patch_budget_assignment(
            "user",
            "person",
            BudgetAssignmentPatch(revision=original.revision, budget_id=None),
            "admin",
            fixture.assignments,
        )
    assert stale.value.problem.status == 409
    await fixture.db.litellm_budgettable.update(
        where={"budget_id": "profile"},
        data={
            "budget_limits": ({"budget_duration": "1mo", "max_budget": 20},),
        },
    )
    result: Final = await fixture.assignments.patch(
        "user", "person", BudgetAssignmentPatch(revision=first.revision, budget_id=None), "admin"
    )
    assert result == Failure(ErrorCode.REVISION_CONFLICT)
    fresh: Final = await fixture.assignments.get("user", "person")
    assert isinstance(fresh, BudgetAssignment)
    assert fresh.budget_id == "profile"
    assert fresh.revision != first.revision


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", ("user", "customer_member"))
async def test_expired_increases_do_not_block_unrelated_patch_and_clear_as_group(subject: PersonBudgetSubject) -> None:
    fixture: Final = Fixture()
    original: Final = await fixture.create(subject)
    saved: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": original.revision,
                "budget_override": {
                    "temp_budget_increase": 5,
                    "temp_budget_duration": "1mo",
                    "temp_budget_expiry": NOW + timedelta(days=1),
                },
            }
        ),
        "admin",
    )
    assert isinstance(saved, BudgetAssignment)
    fixture.assignments.clock = lambda: NOW + timedelta(days=2)
    unrelated: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": saved.revision,
                "budget_override": {"cap_policy": "pause_until_reset"},
            }
        ),
        "admin",
    )
    assert isinstance(unrelated, BudgetAssignment)
    assert unrelated.effective_budget.increase_for("1mo", NOW + timedelta(days=2)) == 0
    cleared: Final = await fixture.assignments.patch(
        subject,
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": unrelated.revision,
                "budget_override": {"temp_budget_increase": None},
            }
        ),
        "admin",
    )
    assert isinstance(cleared, BudgetAssignment)
    assert cleared.budget_override is not None
    assert cleared.budget_override.temp_budget_increase is None
    assert cleared.budget_override.temp_budget_expiry is None
    assert cleared.budget_override.temp_budget_duration is None


@pytest.mark.asyncio
async def test_shared_member_owner_rejects_invalid_increases_before_writes() -> None:
    fixture: Final = Fixture()
    original: Final = await fixture.create("customer_member")
    invalid: Final = MemberPatch.model_validate({"budget_override": {"temp_budget_increase": 5}})
    result: Final = await fixture.hierarchy.update_member("person", invalid, "admin")
    assert isinstance(result, Failure) and result.code == ErrorCode.INVALID_BUDGET
    assert await fixture.assignments.get("customer_member", "person") == original
    assert len(fixture.db.litellm_budgettable.rows) == 1
    created: Final = await fixture.hierarchy.create_member(
        MemberCreate.model_validate(
            {
                "parent_team_id": "team",
                "budget_id": "profile",
                "budget_override": {"temp_budget_increase": 5, "temp_budget_duration": "12mo"},
            }
        ),
        "admin",
    )
    assert isinstance(created, Failure) and created.code == ErrorCode.INVALID_BUDGET
    assert len(fixture.db.litellm_customermembertable.rows) == 1


@pytest.mark.parametrize(
    "body",
    (
        {},
        {"revision": ""},
        {"revision": "r", "budget_override": {"is_profile": True}},
        {"revision": "r", "budget_override": {"budget_limits": [{"budget_duration": "30d", "max_budget": 1}]}},
        {"revision": "r", "budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": -1}]}},
        {"revision": "r", "budget_override": {"temp_budget_increase": float("inf")}},
        {"revision": "r", "budget_override": {"temp_budget_expiry": "2026-01-01T00:00:00"}},
    ),
)
def test_precise_assignment_schema_rejects_ambiguous_or_invalid_inputs(body: object) -> None:
    with pytest.raises(ValidationError):
        BudgetAssignmentPatch.model_validate(body)


@pytest.mark.asyncio
async def test_every_member_mutation_evicts_hierarchy_and_assignment_after_success() -> None:
    fixture: Final = Fixture()
    keys: Final = (customer_member_cache_key("person"), person_budget_assignment_cache_key("person"))
    for key in keys:
        await fixture.cache.async_set_cache(key, "previous member", local_only=True)
    original: Final = await fixture.create("customer_member")
    assert all(fixture.cache.get_cache(key, local_only=True) is None for key in keys)
    for key in keys:
        await fixture.cache.async_set_cache(key, "cached member", local_only=True)
    invalid: Final = await fixture.hierarchy.update_member(
        "person",
        MemberPatch.model_validate(
            {
                "budget_override": {"temp_budget_increase": 5},
            }
        ),
        "admin",
    )
    assert isinstance(invalid, Failure)
    assert all(fixture.cache.get_cache(key, local_only=True) == "cached member" for key in keys)
    updated: Final = await fixture.assignments.patch(
        "customer_member",
        "person",
        BudgetAssignmentPatch.model_validate(
            {
                "revision": original.revision,
                "budget_override": {"cap_policy": "pause_until_reset"},
            }
        ),
        "admin",
    )
    assert isinstance(updated, BudgetAssignment)
    assert all(fixture.cache.get_cache(key, local_only=True) is None for key in keys)
    for key in keys:
        await fixture.cache.async_set_cache(key, "cached member", local_only=True)
    assert await fixture.hierarchy.delete_member("person") is None
    assert all(fixture.cache.get_cache(key, local_only=True) is None for key in keys)


@pytest.mark.asyncio
async def test_generic_http_route_returns_conditional_assignment_and_calendar_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as server

    fixture: Final = Fixture()
    await fixture.create("user")
    fixture.assignments.settings = BudgetResetSettings(timezone="America/New_York")
    # The real auth dependency checks this isolated test master key.
    monkeypatch.setattr(server, "master_key", "sk-assignment-route-test")
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def handle_problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    app.dependency_overrides[get_budget_assignments] = lambda: fixture.assignments
    with TestClient(app) as http:
        headers: Final = {"Authorization": "Bearer sk-assignment-route-test"}
        path: Final = "/management/v1/budget_assignments/user/person"
        initial: Final = http.get(path, headers=headers)
        assert initial.status_code == 200, initial.text
        first: Final = BudgetAssignment.model_validate(initial.json())
        assert {window.budget_duration: window.max_budget for window in first.windows} == {"1mo": 10, "12mo": 100}
        assert all(window.reset_at == datetime(2026, 1, 1, 5, tzinfo=timezone.utc) for window in first.windows)
        saved: Final = http.patch(
            path,
            headers=headers,
            json={
                "revision": first.revision,
                "budget_override": {"budget_limits": [{"budget_duration": "1mo", "max_budget": 0}]},
            },
        )
        assert saved.status_code == 200, saved.text
        updated: Final = BudgetAssignment.model_validate(saved.json())
        assert {window.budget_duration: window.max_budget for window in updated.windows} == {"1mo": 0, "12mo": 100}
        stale: Final = http.patch(path, headers=headers, json={"revision": first.revision, "budget_id": None})
        assert stale.status_code == 409, stale.text
        assert http.get("/management/v1/budget_assignments/team/person", headers=headers).status_code == 422
