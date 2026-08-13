"""Live e2e: the /team/* management routes' block, membership, and admin-only
contract.

Each test creates its team/user/key resources under unique names (deleted on
teardown) and asserts both halves of the contract: the recorded state (the info
route reflects the write) and the enforced behavior (a non-admin key is refused).
Team writes reach the read path once their db/cache entry propagates, so the
read-backs poll to a deadline instead of asserting once.

Everything the shared harness does not already model lives here: the local
request/response models for /team/block, /team/member_update, and the
/team/info fields (blocked flag and per-member budget) these tests assert on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, StreamingResponse, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import (
    KeyGenerateBody,
    TeamInfoParams,
    TeamMemberAddBody,
    TeamMemberDeleteBody,
    TeamMemberEntry,
    TeamNewBody,
    UserNewBody,
)

pytestmark = pytest.mark.e2e

TeamRole = Literal["admin", "user"]


class TeamBlockBody(BaseModel):
    team_id: str


class MemberUpdateBody(BaseModel):
    team_id: str
    user_id: str
    role: TeamRole | None = None
    max_budget_in_team: float | None = None


class MemberRoleEntry(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    role: TeamRole


class MemberBudgetTable(BaseModel):
    max_budget: float | None = None


class TeamMembership(BaseModel):
    user_id: str
    litellm_budget_table: MemberBudgetTable | None = None


class TeamInfoData(BaseModel):
    team_alias: str | None = None
    models: list[str] = []
    blocked: bool | None = None
    members_with_roles: list[MemberRoleEntry] = []


class TeamInfoRead(BaseModel):
    team_id: str
    team_info: TeamInfoData
    team_memberships: list[TeamMembership] = []


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


def _create_team(client: ManagementClient, resources: ResourceManager, alias: str, models: list[str]) -> str:
    team_id = client.create_team(TeamNewBody(team_alias=alias, models=models))
    resources.defer(lambda: client.delete_team(team_id))
    return team_id


def _create_user(client: ManagementClient, resources: ResourceManager, email: str) -> str:
    user_id = client.create_user(UserNewBody(user_email=email, user_role="internal_user"))
    resources.defer(lambda: client.delete_user(user_id))
    return user_id


def _generate_key(client: ManagementClient, resources: ResourceManager, body: KeyGenerateBody) -> str:
    key = client.proxy.generate_key(body)
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _read_team(client: ManagementClient, team_id: str) -> TeamInfoRead:
    return unwrap(
        client.proxy.transport.get(
            "/team/info",
            headers=client.proxy.transport.master,
            params=TeamInfoParams(team_id=team_id),
            response_type=TeamInfoRead,
        )
    )


def _set_blocked(client: ManagementClient, team_id: str, *, blocked: bool) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/team/unblock" if not blocked else "/team/block",
            headers=client.proxy.transport.master,
            json=TeamBlockBody(team_id=team_id),
            response_type=NoBody,
        )
    )


def _member_update(client: ManagementClient, body: MemberUpdateBody) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/team/member_update",
            headers=client.proxy.transport.master,
            json=body,
            response_type=NoBody,
        )
    )


def _member_role(info: TeamInfoRead, user_id: str) -> TeamRole | None:
    return next((m.role for m in info.team_info.members_with_roles if m.user_id == user_id), None)


def _member_max_budget(info: TeamInfoRead, user_id: str) -> float | None:
    membership = next((tm for tm in info.team_memberships if tm.user_id == user_id), None)
    if membership is None or membership.litellm_budget_table is None:
        return None
    return membership.litellm_budget_table.max_budget


def _member_add_status(client: ManagementClient, key: str, team_id: str, user_id: str) -> StreamingResponse:
    return client.proxy.transport.send(
        "/team/member_add",
        headers=client.proxy.transport.bearer(key),
        json=TeamMemberAddBody(team_id=team_id, member=TeamMemberEntry(role="user", user_id=user_id)),
    )


def _member_delete_status(client: ManagementClient, key: str, team_id: str, user_id: str) -> StreamingResponse:
    return client.proxy.transport.send(
        "/team/member_delete",
        headers=client.proxy.transport.bearer(key),
        json=TeamMemberDeleteBody(team_id=team_id, user_id=user_id),
    )


class TestTeamManagementRoutes:
    @pytest.mark.covers("mgmt.team.info.happy_path")
    def test_info_returns_created_team_fields(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-team-info-{unique_marker()}"
        team_id = _create_team(client, resources, alias, ["gemini-2.5-flash"])

        info = _read_team(client, team_id)
        assert info.team_id == team_id, f"/team/info echoed team_id {info.team_id!r}, requested {team_id!r}"
        assert info.team_info.team_alias == alias, (
            f"/team/info reports team_alias {info.team_info.team_alias!r}, configured {alias!r}"
        )
        assert info.team_info.models == ["gemini-2.5-flash"], (
            f"/team/info reports models {info.team_info.models}, configured ['gemini-2.5-flash']"
        )

    @pytest.mark.covers("mgmt.team.block.persists")
    def test_block_then_unblock_persists_to_team_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_id = _create_team(client, resources, f"e2e-team-block-{unique_marker()}", ["gemini-2.5-flash"])
        assert not _read_team(client, team_id).team_info.blocked, "/team/info reports the team blocked before /team/block"

        _set_blocked(client, team_id, blocked=True)
        _ = _poll(
            client,
            lambda: True if _read_team(client, team_id).team_info.blocked else None,
            "/team/info never reflected blocked=True after /team/block",
        )

        _set_blocked(client, team_id, blocked=False)
        _ = _poll(
            client,
            lambda: True if _read_team(client, team_id).team_info.blocked is False else None,
            "/team/info never reflected blocked=False after /team/unblock",
        )

    @pytest.mark.covers("mgmt.team.member_update.persists")
    def test_member_update_persists_role_and_budget(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        user_id = _create_user(client, resources, f"e2e-team-mu-{unique_marker()}@example.com")
        team_id = _create_team(client, resources, f"e2e-team-mu-{unique_marker()}", ["gemini-2.5-flash"])
        client.add_team_member(team_id, user_id)
        assert _member_role(_read_team(client, team_id), user_id) == "user", (
            f"member {user_id} should start as role 'user' after /team/member_add"
        )

        budget = 4242.0
        _member_update(client, MemberUpdateBody(team_id=team_id, user_id=user_id, role="admin", max_budget_in_team=budget))

        def updated() -> bool | None:
            info = _read_team(client, team_id)
            return True if _member_role(info, user_id) == "admin" and _member_max_budget(info, user_id) == budget else None

        _ = _poll(
            client,
            updated,
            f"/team/info never reflected role=admin and max_budget={budget} for {user_id} after /team/member_update",
        )

    @pytest.mark.covers("mgmt.team.member_delete.persists")
    def test_member_delete_persists_to_team_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        user_id = _create_user(client, resources, f"e2e-team-md-{unique_marker()}@example.com")
        team_id = _create_team(client, resources, f"e2e-team-md-{unique_marker()}", ["gemini-2.5-flash"])
        client.add_team_member(team_id, user_id)
        assert _member_role(_read_team(client, team_id), user_id) == "user", (
            f"/team/info does not list {user_id} as a member after /team/member_add"
        )

        client.delete_team_member(team_id, user_id)
        _ = _poll(
            client,
            lambda: True if _member_role(_read_team(client, team_id), user_id) is None else None,
            f"/team/info still lists {user_id} after /team/member_delete",
        )

    @pytest.mark.covers("mgmt.team.new.admin_only")
    def test_new_is_denied_to_non_admin_keys(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        no_role_key = _generate_key(client, resources, KeyGenerateBody(models=[]))
        internal_user_id = _create_user(client, resources, f"e2e-team-adm-{unique_marker()}@example.com")
        internal_user_key = _generate_key(client, resources, KeyGenerateBody(user_id=internal_user_id))

        for key, label in ((no_role_key, "role=None"), (internal_user_key, "internal_user")):
            outcome = client.team_new_status(key, TeamNewBody(team_alias=f"e2e-team-adm-{unique_marker()}"))
            assert outcome.status_code in (401, 403), (
                f"/team/new by a {label} key must be denied 401/403, got {outcome.status_code}: {outcome.body[:300]}"
            )

    @pytest.mark.covers("mgmt.team.member_add.member_forbidden")
    def test_member_add_forbidden_to_plain_member(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        _member_id, other_id, member_key, team_id = self._team_with_member_key(client, resources)

        outcome = _member_add_status(client, member_key, team_id, other_id)
        assert outcome.status_code == 403, (
            f"/team/member_add by a plain team member must be 403, got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert "not allowed" in outcome.body.lower(), (
            f"403 body should say the call is not allowed, got: {outcome.body[:300]}"
        )

    @pytest.mark.covers("mgmt.team.member_delete.member_forbidden")
    def test_member_delete_forbidden_to_plain_member(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        member_id, _other_id, member_key, team_id = self._team_with_member_key(client, resources)

        outcome = _member_delete_status(client, member_key, team_id, member_id)
        assert outcome.status_code == 403, (
            f"/team/member_delete by a plain team member must be 403, got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert "not allowed" in outcome.body.lower(), (
            f"403 body should say the call is not allowed, got: {outcome.body[:300]}"
        )

    @staticmethod
    def _team_with_member_key(
        client: ManagementClient, resources: ResourceManager
    ) -> tuple[str, str, str, str]:
        """A team with a plain member (role user) whose key is scoped to that
        user + team, plus a second user id the member could try to add."""
        member_id = _create_user(client, resources, f"e2e-team-fb-{unique_marker()}@example.com")
        other_id = _create_user(client, resources, f"e2e-team-fb-{unique_marker()}@example.com")
        team_id = _create_team(client, resources, f"e2e-team-fb-{unique_marker()}", ["gemini-2.5-flash"])
        client.add_team_member(team_id, member_id)
        member_key = _generate_key(client, resources, KeyGenerateBody(user_id=member_id, team_id=team_id))
        return member_id, other_id, member_key, team_id
