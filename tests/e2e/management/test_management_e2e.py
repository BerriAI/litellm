"""Live e2e: the key/team/user/organization management routes' lifecycle contract.

Each test creates its resources under unique names (deleted on teardown) and
asserts both halves of the contract: the recorded state (the info route reflects
the write) and the enforced behavior (the data plane serves or refuses traffic
accordingly). Key writes reach the data plane when its auth cache entry expires,
so the traffic-facing read-backs poll to a deadline instead of asserting once.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from management_client import (
    MODEL_ACCESS_DENIED_MARKER,
    PROXY_ADMIN_REQUIRED_MARKER,
    ROUTE_NOT_ALLOWED_MARKER,
    TEAM_ADMIN_REQUIRED_MARKER,
    ManagementClient,
)
from models import (
    BudgetNewBody,
    KeyGenerateBody,
    OrgNewBody,
    TeamMemberAddBody,
    TeamMemberDeleteBody,
    TeamMemberEntry,
    TeamNewBody,
    UserNewBody,
)

pytestmark = pytest.mark.e2e

def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.gateway.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.gateway.poll_interval)
    pytest.fail(failure)


def _generate_key(client: ManagementClient, resources: ResourceManager, body: KeyGenerateBody) -> str:
    key = client.gateway.generate_key(body)
    resources.defer(lambda: client.gateway.delete_key(key))
    return key


def _create_team(client: ManagementClient, resources: ResourceManager, alias: str, models: list[str]) -> str:
    team_id = client.create_team(TeamNewBody(team_alias=alias, models=models))
    resources.defer(lambda: client.delete_team(team_id))
    return team_id


def _create_user(client: ManagementClient, resources: ResourceManager, body: UserNewBody) -> str:
    user_id = client.create_user(body)
    resources.defer(lambda: client.delete_user(user_id))
    return user_id


def _create_budget(client: ManagementClient, resources: ResourceManager, body: BudgetNewBody) -> str:
    budget_id = client.create_budget(body)
    resources.defer(lambda: client.delete_budget(budget_id))
    return budget_id


def _create_nonadmin_key(client: ManagementClient, resources: ResourceManager) -> str:
    """A key bound to a fresh internal_user: it inherits that user's non-admin
    role, so route_checks denies it the proxy-admin-only management writes."""
    user_id = _create_user(
        client,
        resources,
        UserNewBody(user_email=f"e2e-mgmt-nonadmin-{unique_marker()}@example.com", user_role="internal_user"),
    )
    return _generate_key(client, resources, KeyGenerateBody(user_id=user_id))


def _add_team_member(client: ManagementClient, resources: ResourceManager, team_id: str) -> tuple[str, str]:
    """Create an internal user, add them to `team_id` as a plain member (role
    'user', not team admin), and return their (user_id, key scoped to the team)."""
    user_id = _create_user(
        client,
        resources,
        UserNewBody(user_email=f"e2e-mgmt-member-{unique_marker()}@example.com", user_role="internal_user"),
    )
    client.add_team_member(team_id, user_id)
    key = _generate_key(client, resources, KeyGenerateBody(user_id=user_id, team_id=team_id))
    return user_id, key


def _is_model_denial(outcome: StreamingResponse) -> bool:
    return outcome.status_code == 403 and MODEL_ACCESS_DENIED_MARKER in outcome.body


def _assert_model_denied(outcome: StreamingResponse, model: str) -> None:
    assert outcome.status_code == 403, (
        f"chat on {model!r} outside the key's model list must be denied 403, got "
        f"{outcome.status_code}: {outcome.body[:300]}"
    )
    assert MODEL_ACCESS_DENIED_MARKER in outcome.body, (
        f"403 body must be a model-access denial, got: {outcome.body[:300]}"
    )


def _poll_chat_ok(client: ManagementClient, key: str, model: str) -> None:
    def attempt() -> bool | None:
        outcome = client.chat_status(key, model, f"reply with one word {unique_marker()}")
        return True if outcome.ok else None

    _ = _poll(client, attempt, f"chat on {model} never succeeded for the key before the deadline")


def _poll_chat_denied(client: ManagementClient, key: str, model: str) -> None:
    def attempt() -> bool | None:
        return True if _is_model_denial(client.chat_status(key, model, f"say hi {unique_marker()}")) else None

    _ = _poll(
        client,
        attempt,
        f"chat on {model} was never denied with {MODEL_ACCESS_DENIED_MARKER} before the deadline",
    )


def _poll_model_access_granted(client: ManagementClient, key: str, model: str) -> None:
    """The key's model-access check stopped denying `model`: any outcome other than
    the key_model_access_denied 403 (a 200, or an upstream error) proves the flip.
    Requiring a 200 would couple the assertion to `model` being a healthy routable
    upstream, which is not the enforcement contract under test."""

    def attempt() -> bool | None:
        outcome = client.chat_status(key, model, f"say hi {unique_marker()}")
        if _is_model_denial(outcome) or outcome.status_code == 401:
            return None
        return True

    _ = _poll(client, attempt, f"model-access denial on {model} never lifted before the deadline")


class TestKeyRoutes:
    @pytest.mark.covers("mgmt.key.generate.persists")
    @pytest.mark.covers("mgmt.key.info.persists")
    def test_generate_persists_to_key_info_and_scopes_chat(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-mgmt-key-{unique_marker()}"
        key = _generate_key(
            client,
            resources,
            KeyGenerateBody(models=["gemini-2.5-flash"], key_alias=alias, tpm_limit=424242),
        )

        info = client.gateway.key_info(key)
        assert info.key_alias == alias, f"/key/info reports key_alias {info.key_alias!r}, configured {alias!r}"
        assert info.models == ["gemini-2.5-flash"], (
            f"/key/info reports models {info.models}, configured ['gemini-2.5-flash']"
        )
        assert info.tpm_limit == 424242, (
            f"/key/info reports tpm_limit {info.tpm_limit}, configured 424242"
        )

        _poll_chat_ok(client, key, "gemini-2.5-flash")
        _assert_model_denied(
            client.chat_status(key, "gpt-5.5", f"say hi {unique_marker()}"), "gpt-5.5"
        )

    @pytest.mark.covers("mgmt.key.update.persists")
    def test_update_models_persists_and_flips_enforcement(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = _generate_key(client, resources, KeyGenerateBody(models=["gemini-2.5-flash"]))
        _poll_chat_ok(client, key, "gemini-2.5-flash")
        _assert_model_denied(
            client.chat_status(key, "gpt-5.5", f"say hi {unique_marker()}"), "gpt-5.5"
        )

        client.update_key_models(key, ["gpt-5.5"])

        info = client.gateway.key_info(key)
        assert info.models == ["gpt-5.5"], (
            f"/key/info reports models {info.models} after /key/update to ['gpt-5.5']"
        )

        _poll_model_access_granted(client, key, "gpt-5.5")
        _poll_chat_denied(client, key, "gemini-2.5-flash")

    @pytest.mark.covers("mgmt.key.delete.persists")
    def test_delete_revokes_the_key_on_chat(self, client: ManagementClient, resources: ResourceManager) -> None:
        """The teardown's deferred delete fires again on the already-deleted key by
        design: the deferred cleanup must survive this test failing before the
        in-body delete, and a repeat /key/delete is a cheap no-op the warn-only
        teardown absorbs."""
        key = _generate_key(client, resources, KeyGenerateBody(models=["gemini-2.5-flash"]))
        _poll_chat_ok(client, key, "gemini-2.5-flash")

        client.delete_key_strict(key)

        def rejected() -> bool | None:
            outcome = client.chat_status(key, "gemini-2.5-flash", f"say hi {unique_marker()}")
            return True if outcome.status_code == 401 else None

        _ = _poll(client, rejected, "deleted key was still accepted on chat (never rejected 401) at the deadline")


class TestTeamRoutes:
    @pytest.mark.covers("mgmt.team.new.persists")
    def test_new_persists_to_team_info_and_binds_keys(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-mgmt-team-{unique_marker()}"
        team_id = _create_team(client, resources, alias, ["gemini-2.5-flash"])

        info = client.team_info(team_id)
        assert info.team_alias == alias, f"/team/info reports team_alias {info.team_alias!r}, configured {alias!r}"
        assert info.models == ["gemini-2.5-flash"], (
            f"/team/info reports models {info.models}, configured ['gemini-2.5-flash']"
        )

        key = _generate_key(client, resources, KeyGenerateBody(team_id=team_id))
        key_info = client.gateway.key_info(key)
        assert key_info.team_id == team_id, (
            f"key generated under team {team_id} carries team_id {key_info.team_id!r} in /key/info"
        )

    @pytest.mark.covers("mgmt.team.member_add.persists")
    @pytest.mark.covers("mgmt.team.member_delete.persists")
    def test_member_add_and_delete_persist_to_team_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        user_id = _create_user(
            client,
            resources,
            UserNewBody(user_email=f"e2e-mgmt-{unique_marker()}@example.com", user_role="internal_user"),
        )
        team_id = _create_team(client, resources, f"e2e-mgmt-team-{unique_marker()}", ["gemini-2.5-flash"])

        client.add_team_member(team_id, user_id)
        member = next(
            (entry for entry in client.team_info(team_id).members_with_roles if entry.user_id == user_id), None
        )
        assert member is not None, f"/team/info does not list {user_id} after /team/member_add"
        assert member.role == "user", f"member {user_id} added with role 'user' but /team/info reports {member.role!r}"

        client.delete_team_member(team_id, user_id)
        remaining = client.team_info(team_id).members_with_roles
        assert all(entry.user_id != user_id for entry in remaining), (
            f"/team/info still lists {user_id} after /team/member_delete"
        )

    @pytest.mark.covers("mgmt.team.new.admin_only")
    def test_new_forbidden_for_non_admin(self, client: ManagementClient, resources: ResourceManager) -> None:
        key = _create_nonadmin_key(client, resources)
        team_id = f"e2e-mgmt-team-forbidden-{unique_marker()}"

        _assert_admin_only(
            "/team/new", client.team_new_status(key, TeamNewBody(team_alias=team_id, team_id=team_id))
        )

        probe = client.team_info_status(team_id)
        assert probe.status_code == 404, (
            f"team {team_id} was created despite the 401 admin-only denial: "
            f"/team/info returned {probe.status_code}: {probe.body[:300]}"
        )

    @pytest.mark.covers("mgmt.team.member_add.member_forbidden")
    @pytest.mark.covers("mgmt.team.member_delete.member_forbidden")
    def test_member_writes_forbidden_for_non_admin_member(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_id = _create_team(client, resources, f"e2e-mgmt-team-{unique_marker()}", ["gemini-2.5-flash"])
        member_id, member_key = _add_team_member(client, resources, team_id)
        outsider_id = _create_user(
            client,
            resources,
            UserNewBody(user_email=f"e2e-mgmt-outsider-{unique_marker()}@example.com", user_role="internal_user"),
        )

        _assert_team_admin_required(
            "/team/member_add",
            client.team_member_add_status(
                member_key,
                TeamMemberAddBody(team_id=team_id, member=TeamMemberEntry(role="user", user_id=outsider_id)),
            ),
        )
        _assert_team_admin_required(
            "/team/member_delete",
            client.team_member_delete_status(member_key, TeamMemberDeleteBody(team_id=team_id, user_id=member_id)),
        )

        roster = {entry.user_id for entry in client.team_info(team_id).members_with_roles}
        assert outsider_id not in roster, (
            f"outsider {outsider_id} was added to team {team_id} despite the 403 team-admin-required denial"
        )
        assert member_id in roster, (
            f"member {member_id} was removed from team {team_id} despite the 403 team-admin-required denial"
        )


class TestUserRoutes:
    @pytest.mark.covers("mgmt.user.new.happy_path")
    def test_new_persists_to_user_info(self, client: ManagementClient, resources: ResourceManager) -> None:
        email = f"e2e-mgmt-{unique_marker()}@example.com"
        user_id = _create_user(client, resources, UserNewBody(user_email=email, user_role="internal_user"))

        info = client.user_info(user_id).user_info
        assert info.user_email == email, f"/user/info reports user_email {info.user_email!r}, configured {email!r}"
        assert info.user_role == "internal_user", (
            f"/user/info reports user_role {info.user_role!r}, configured 'internal_user'"
        )


class TestOrganizationRoutes:
    @pytest.mark.covers("mgmt.organization.new.happy_path")
    def test_new_persists_to_organization_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-mgmt-org-{unique_marker()}"
        org_id = client.create_org(OrgNewBody(organization_alias=alias, models=["gemini-2.5-flash"]))
        resources.defer(lambda: client.delete_org(org_id))

        info = client.org_info(org_id)
        assert info.organization_alias == alias, (
            f"/organization/info reports alias {info.organization_alias!r}, configured {alias!r}"
        )
        assert info.models == ["gemini-2.5-flash"], (
            f"/organization/info reports models {info.models}, configured ['gemini-2.5-flash']"
        )


class TestBudgetRoutes:
    @pytest.mark.covers("mgmt.budget.new.persists")
    def test_new_persists_to_budget_info(self, client: ManagementClient, resources: ResourceManager) -> None:
        budget_id = f"e2e-mgmt-budget-{unique_marker()}"
        _create_budget(
            client,
            resources,
            BudgetNewBody(budget_id=budget_id, max_budget=42.5, soft_budget=21.25, budget_duration="30d"),
        )

        row = client.budget_row(budget_id)
        assert row is not None, f"/budget/info does not list {budget_id} after /budget/new"
        assert row.max_budget == 42.5, f"/budget/info reports max_budget {row.max_budget}, configured 42.5"
        assert row.soft_budget == 21.25, f"/budget/info reports soft_budget {row.soft_budget}, configured 21.25"
        assert row.budget_duration == "30d", (
            f"/budget/info reports budget_duration {row.budget_duration!r}, configured '30d'"
        )
        assert row.budget_reset_at is not None, (
            "a budget with a 30d duration must persist a computed budget_reset_at, got None"
        )

    @pytest.mark.covers("mgmt.budget.new.admin_only")
    def test_new_forbidden_for_non_admin(self, client: ManagementClient, resources: ResourceManager) -> None:
        key = _create_nonadmin_key(client, resources)
        budget_id = f"e2e-mgmt-budget-forbidden-{unique_marker()}"

        _assert_admin_only(
            "/budget/new", client.budget_new_status(key, BudgetNewBody(budget_id=budget_id, max_budget=1))
        )

        assert client.budget_row(budget_id) is None, (
            f"budget {budget_id} was created despite the 401 admin-only denial"
        )


def _assert_route_forbidden(route: str, outcome: StreamingResponse) -> None:
    assert outcome.status_code == 403, (
        f"llm-only key POSTing {route} must be denied exactly 403, got {outcome.status_code}: {outcome.body[:300]}"
    )
    assert ROUTE_NOT_ALLOWED_MARKER in outcome.body, (
        f"{route} denial body must be a route-permission denial, got: {outcome.body[:300]}"
    )


def _assert_admin_only(route: str, outcome: StreamingResponse) -> None:
    assert outcome.status_code == 401, (
        f"non-admin key POSTing {route} must be denied 401, got {outcome.status_code}: {outcome.body[:300]}"
    )
    assert PROXY_ADMIN_REQUIRED_MARKER in outcome.body, (
        f"{route} denial body must be a proxy-admin-only error, got: {outcome.body[:300]}"
    )


def _assert_team_admin_required(route: str, outcome: StreamingResponse) -> None:
    assert outcome.status_code == 403, (
        f"non-admin team member POSTing {route} must be denied 403, got {outcome.status_code}: {outcome.body[:300]}"
    )
    assert TEAM_ADMIN_REQUIRED_MARKER in outcome.body, (
        f"{route} denial body must be a team-admin-required error, got: {outcome.body[:300]}"
    )


class TestManagementRoutePermissions:
    @pytest.mark.covers("other.auth.virtual_key.route_permission_enforced")
    def test_llm_only_key_forbidden_from_management_writes(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.llm_only_key()
        resources.defer(lambda: client.gateway.delete_key(key))
        marker = unique_marker()
        alias = f"e2e-mgmt-forbidden-key-{marker}"
        team_id = f"e2e-mgmt-forbidden-team-{marker}"
        user_id = f"e2e-mgmt-forbidden-user-{marker}"

        _assert_route_forbidden(
            "/key/generate", client.key_generate_status(key, KeyGenerateBody(models=[], key_alias=alias))
        )
        _assert_route_forbidden(
            "/team/new", client.team_new_status(key, TeamNewBody(team_alias=team_id, team_id=team_id))
        )
        _assert_route_forbidden(
            "/user/new",
            client.user_new_status(
                key,
                UserNewBody(user_email=f"{user_id}@example.com", user_role="internal_user", user_id=user_id),
            ),
        )

        assert client.key_alias_count(alias) == 0, f"key {alias} was created despite the 403 route denial"
        team_probe = client.team_info_status(team_id)
        assert team_probe.status_code == 404, (
            f"team {team_id} was created despite the 403 route denial: "
            f"/team/info returned {team_probe.status_code}: {team_probe.body[:300]}"
        )
        assert client.user_count(user_id) == 0, f"user {user_id} was created despite the 403 route denial"
