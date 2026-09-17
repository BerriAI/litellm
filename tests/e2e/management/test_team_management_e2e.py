"""Live e2e: the /team/* management routes' block, membership, and admin-only
contract, plus the team settings a team admin may change on /team/update once a
proxy admin enables them under Settings > UI > Team admin editable fields.

Each test creates its team/user/key resources under unique names (deleted on
teardown) and asserts both halves of the contract: the recorded state (the info
route reflects the write) and the enforced behavior (a non-admin key is refused).
Team writes reach the read path once their db/cache entry propagates, so the
read-backs poll to a deadline instead of asserting once.

Everything the shared harness does not already model lives here: the local
request/response models for /team/block, /team/member_update, the partial
/team/update, the UI settings allow-list, and the /team/info fields (blocked
flag, limits, budgets, per-member budget, the caller's edit access) these tests
assert on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import pytest
from pydantic import BaseModel

from e2e_config import settle_propagation, unique_marker
from e2e_http import NoBody, PartialBody, StreamingResponse, unwrap
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

_TEAM_TPM_LIMIT: Final = 1000
_TEAM_MAX_BUDGET: Final = 10.0


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


class CallerEditAccess(BaseModel):
    kind: Literal["unrestricted", "team_admin", "team_admin_disabled", "none"]
    editable_fields: list[str] = []


class BudgetWindow(BaseModel):
    budget_duration: str
    max_budget: float
    reset_at: str | None = None


class TeamCustomMetadata(BaseModel):
    cost_center: str | None = None


class TeamSettings(BaseModel):
    team_alias: str | None = None
    models: list[str] = []
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_budget: float | None = None
    budget_duration: str | None = None
    budget_limits: list[BudgetWindow] | None = None
    metadata: TeamCustomMetadata | None = None


class TeamInfoData(TeamSettings):
    blocked: bool | None = None
    members_with_roles: list[MemberRoleEntry] = []
    budget_reset_at: datetime | None = None
    caller_edit_access: CallerEditAccess | None = None


class TeamInfoRead(BaseModel):
    team_id: str
    team_info: TeamInfoData
    team_memberships: list[TeamMembership] = []


class TeamWithAdminNewBody(TeamNewBody):
    tpm_limit: int
    max_budget: float | None = None
    members_with_roles: list[TeamMemberEntry]


class TeamSettingsChange(PartialBody, TeamSettings):
    pass


class TeamSettingsUpdate(TeamSettingsChange):
    team_id: str


class TeamAdminEditableFields(BaseModel):
    team_admin_editable_team_fields: list[str] = []


class UiSettingsRead(BaseModel):
    values: TeamAdminEditableFields


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


def _read_team(client: ManagementClient, team_id: str, caller_key: str | None = None) -> TeamInfoRead:
    return unwrap(
        client.proxy.transport.get(
            "/team/info",
            headers=client.proxy.transport.master if caller_key is None else client.proxy.transport.bearer(caller_key),
            params=TeamInfoParams(team_id=team_id),
            response_type=TeamInfoRead,
        )
    )


def _poll_team(
    client: ManagementClient, team_id: str, ready: Callable[[TeamInfoData], bool], failure: str
) -> TeamInfoData:
    def read() -> TeamInfoData | None:
        info = _read_team(client, team_id).team_info
        return info if ready(info) else None

    return _poll(client, read, failure)


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


def _team_admin_editable_fields(client: ManagementClient) -> list[str]:
    return unwrap(
        client.proxy.transport.get(
            "/get/ui_settings",
            headers=client.proxy.transport.master,
            params=NoBody(),
            response_type=UiSettingsRead,
        )
    ).values.team_admin_editable_team_fields


def _set_team_admin_editable_fields(client: ManagementClient, fields: list[str]) -> None:
    _ = unwrap(
        client.proxy.transport.patch(
            "/update/ui_settings",
            headers=client.proxy.transport.master,
            json=TeamAdminEditableFields(team_admin_editable_team_fields=fields),
            response_type=NoBody,
        )
    )


@contextmanager
def _team_admins_may_edit(client: ManagementClient, fields: list[str]) -> Generator[None]:
    """The allow-list is proxy-wide, so restore whatever was there. Other replicas pick a change up on their
    config reload, which the wait covers before any team admin call lands on one of them."""
    original = _team_admin_editable_fields(client)
    _set_team_admin_editable_fields(client, fields)
    settle_propagation(time.monotonic())
    try:
        yield
    finally:
        _set_team_admin_editable_fields(client, original)


@pytest.fixture(scope="class")
def no_team_admin_editable_fields(client: ManagementClient) -> Generator[None]:
    with _team_admins_may_edit(client, []):
        yield


@pytest.fixture(scope="class")
def tpm_limit_editable_by_team_admins(client: ManagementClient) -> Generator[None]:
    with _team_admins_may_edit(client, ["tpm_limit"]):
        yield


@pytest.fixture(scope="class")
def rpm_limit_and_max_budget_editable_by_team_admins(client: ManagementClient) -> Generator[None]:
    with _team_admins_may_edit(client, ["rpm_limit", "max_budget"]):
        yield


def _team_with_admin(
    client: ManagementClient, resources: ResourceManager, max_budget: float | None = None
) -> tuple[str, str]:
    """A team with a tpm_limit, and the key of a user who is an admin of that team."""
    admin_id = _create_user(client, resources, f"e2e-team-admin-{unique_marker()}@example.com")
    team_id = client.create_team(
        TeamWithAdminNewBody(
            team_alias=f"e2e-team-admin-{unique_marker()}",
            tpm_limit=_TEAM_TPM_LIMIT,
            max_budget=max_budget,
            members_with_roles=[TeamMemberEntry(role="admin", user_id=admin_id)],
        )
    )
    resources.defer(lambda: client.delete_team(team_id))
    return team_id, _generate_key(client, resources, KeyGenerateBody(user_id=admin_id))


def _update_team_as(client: ManagementClient, caller_key: str, body: TeamSettingsUpdate) -> StreamingResponse:
    return client.proxy.transport.send("/team/update", headers=client.proxy.transport.bearer(caller_key), json=body)


@pytest.mark.usefixtures("no_team_admin_editable_fields")
class TestTeamAdminWithNoEditableFields:
    """No proxy admin has enabled a team field for team admins, which is how every proxy starts."""

    @pytest.mark.covers("mgmt.team.update.team_admin_forbidden_until_enabled")
    def test_team_admin_cannot_change_any_team_setting(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_id, admin_key = _team_with_admin(client, resources)
        access = _read_team(client, team_id, admin_key).team_info.caller_edit_access
        assert access == CallerEditAccess(kind="team_admin_disabled"), (
            f"/team/info should tell the team admin that editing is disabled, got {access}"
        )

        outcome = _update_team_as(client, admin_key, TeamSettingsUpdate(team_id=team_id, tpm_limit=5000))

        assert outcome.status_code == 403, (
            f"/team/update by a team admin must be 403 while nothing is enabled, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        assert "cannot edit team settings" in outcome.body, f"403 body should say why, got: {outcome.body[:300]}"
        tpm_limit = _read_team(client, team_id).team_info.tpm_limit
        assert tpm_limit == _TEAM_TPM_LIMIT, f"the refused update still changed tpm_limit to {tpm_limit}"


@pytest.mark.usefixtures("tpm_limit_editable_by_team_admins")
class TestTeamAdminWithTpmLimitEnabled:
    """A proxy admin has enabled tpm_limit, so a team admin may change that setting and no other."""

    @pytest.mark.covers("mgmt.team.update.team_admin_limited_to_enabled_fields")
    def test_team_admin_saves_the_settings_form_with_a_new_tpm_limit(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_id, admin_key = _team_with_admin(client, resources)
        access = _read_team(client, team_id, admin_key).team_info.caller_edit_access
        assert access == CallerEditAccess(kind="team_admin", editable_fields=["tpm_limit"]), (
            f"/team/info should list tpm_limit as the team admin's only editable field, got {access}"
        )
        before = _read_team(client, team_id).team_info

        outcome = _update_team_as(
            client,
            admin_key,
            TeamSettingsUpdate(team_id=team_id, team_alias=before.team_alias, models=before.models, tpm_limit=5000),
        )

        assert outcome.status_code == 200, (
            f"a team admin resending the form with only tpm_limit changed must succeed, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        after = _poll_team(
            client, team_id, lambda info: info.tpm_limit == 5000, "/team/info never reflected tpm_limit=5000"
        )
        assert after.model_copy(update={"tpm_limit": _TEAM_TPM_LIMIT}) == before, (
            f"the update changed more than tpm_limit: before {before}, after {after}"
        )

    @pytest.mark.covers("mgmt.team.update.team_admin_limited_to_enabled_fields")
    @pytest.mark.parametrize(
        "change",
        [
            pytest.param(TeamSettingsChange(rpm_limit=10), id="rpm_limit"),
            pytest.param(TeamSettingsChange(max_budget=0.5), id="max_budget"),
            pytest.param(TeamSettingsChange(team_alias="renamed-by-team-admin"), id="team_alias"),
            pytest.param(TeamSettingsChange(models=["gemini-2.5-flash"]), id="models"),
            pytest.param(TeamSettingsChange(budget_duration="1d"), id="budget_duration"),
            pytest.param(TeamSettingsChange(metadata=TeamCustomMetadata(cost_center="team-admin")), id="metadata"),
        ],
    )
    def test_team_admin_cannot_change_a_setting_that_is_not_enabled(
        self, client: ManagementClient, resources: ResourceManager, change: TeamSettingsChange
    ) -> None:
        (field,) = change.model_fields_set
        team_id, admin_key = _team_with_admin(client, resources)
        before = _read_team(client, team_id).team_info

        outcome = _update_team_as(
            client,
            admin_key,
            TeamSettingsUpdate.model_validate(
                {**change.model_dump(exclude_unset=True), "team_id": team_id, "tpm_limit": 5000}
            ),
        )

        assert outcome.status_code == 403, (
            f"a team admin changing {field} must be 403, got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert f"'{field}'" in outcome.body, f"403 body should name {field}, got: {outcome.body[:300]}"
        after = _read_team(client, team_id).team_info
        assert after == before, (
            f"the refused update still wrote to the team, the enabled tpm_limit included: before {before}, "
            f"after {after}"
        )

    @pytest.mark.covers("mgmt.team.update.team_admin_resend_keeps_budget_reset")
    def test_team_admin_resending_the_budget_settings_keeps_the_next_budget_reset(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """A 120s budget resets at the start of the minute after next. Resending it once the next minute has
        started would push that reset a minute later, while the stored reset is still a minute out, so the
        proxy's budget reset job cannot be what moves it."""
        team_id, admin_key = _team_with_admin(client, resources)
        _ = unwrap(
            client.proxy.transport.post(
                "/team/update",
                headers=client.proxy.transport.master,
                json=TeamSettingsUpdate(
                    team_id=team_id,
                    budget_duration="120s",
                    budget_limits=[BudgetWindow(budget_duration="120s", max_budget=5.0)],
                ),
                response_type=NoBody,
            )
        )
        budgeted = _poll_team(
            client,
            team_id,
            lambda info: info.budget_reset_at is not None and bool(info.budget_limits),
            "/team/info never reflected the 120s budget the proxy admin set",
        )
        assert budgeted.budget_reset_at is not None
        next_minute = budgeted.budget_reset_at - timedelta(seconds=58)
        time.sleep(max(0.0, (next_minute - datetime.now(UTC)).total_seconds()))

        outcome = _update_team_as(
            client,
            admin_key,
            TeamSettingsUpdate(
                team_id=team_id,
                tpm_limit=5000,
                budget_duration=budgeted.budget_duration,
                budget_limits=budgeted.budget_limits,
            ),
        )

        assert outcome.status_code == 200, (
            f"resending unchanged budget settings with a new tpm_limit must succeed, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        after = _poll_team(
            client, team_id, lambda info: info.tpm_limit == 5000, "/team/info never reflected tpm_limit=5000"
        )
        assert after.budget_reset_at == budgeted.budget_reset_at, (
            f"the team admin pushed the budget reset from {budgeted.budget_reset_at} to {after.budget_reset_at}"
        )
        assert after.budget_limits == budgeted.budget_limits, (
            f"the team admin pushed the budget window resets from {budgeted.budget_limits} to {after.budget_limits}"
        )


@pytest.mark.usefixtures("rpm_limit_and_max_budget_editable_by_team_admins")
class TestTeamAdminWithRpmLimitAndMaxBudgetEnabled:
    """A proxy admin has enabled rpm_limit and max_budget, so a team admin may change the RPM limit and keep or
    lower the team's budget. Raising or removing the budget stays with the proxy admin."""

    @pytest.mark.covers("mgmt.team.update.team_admin_limited_to_enabled_fields")
    def test_team_admin_saves_a_new_rpm_limit_and_a_lower_budget(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        team_id, admin_key = _team_with_admin(client, resources, max_budget=_TEAM_MAX_BUDGET)
        access = _read_team(client, team_id, admin_key).team_info.caller_edit_access
        assert access == CallerEditAccess(kind="team_admin", editable_fields=["max_budget", "rpm_limit"]), (
            f"/team/info should list max_budget and rpm_limit as the team admin's editable fields, got {access}"
        )
        before = _read_team(client, team_id).team_info

        outcome = _update_team_as(
            client, admin_key, TeamSettingsUpdate(team_id=team_id, rpm_limit=50, max_budget=_TEAM_MAX_BUDGET / 2)
        )

        assert outcome.status_code == 200, (
            f"a team admin setting an RPM limit and lowering the budget must succeed, got {outcome.status_code}: "
            f"{outcome.body[:300]}"
        )
        after = _poll_team(
            client,
            team_id,
            lambda info: info.rpm_limit == 50 and info.max_budget == _TEAM_MAX_BUDGET / 2,
            f"/team/info never reflected rpm_limit=50 and max_budget={_TEAM_MAX_BUDGET / 2}",
        )
        assert after.model_copy(update={"rpm_limit": before.rpm_limit, "max_budget": before.max_budget}) == before, (
            f"the update changed more than rpm_limit and max_budget: before {before}, after {after}"
        )

    @pytest.mark.covers("mgmt.team.update.team_admin_cannot_grow_budget")
    @pytest.mark.parametrize(
        ("max_budget", "refusal"),
        [
            pytest.param(_TEAM_MAX_BUDGET * 2, "Only a proxy admin can raise", id="raise"),
            pytest.param(None, "Only a proxy admin can remove", id="remove"),
        ],
    )
    def test_team_admin_cannot_raise_or_remove_the_budget(
        self, client: ManagementClient, resources: ResourceManager, max_budget: float | None, refusal: str
    ) -> None:
        team_id, admin_key = _team_with_admin(client, resources, max_budget=_TEAM_MAX_BUDGET)
        before = _read_team(client, team_id).team_info

        outcome = _update_team_as(
            client, admin_key, TeamSettingsUpdate(team_id=team_id, rpm_limit=50, max_budget=max_budget)
        )

        assert outcome.status_code == 403, (
            f"a team admin changing max_budget from {_TEAM_MAX_BUDGET} to {max_budget} must be 403, "
            f"got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert refusal in outcome.body, f"403 body should say {refusal!r}, got: {outcome.body[:300]}"
        after = _read_team(client, team_id).team_info
        assert after == before, (
            f"the refused update still wrote to the team, the rpm_limit included: before {before}, after {after}"
        )
