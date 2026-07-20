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
    ROUTE_NOT_ALLOWED_MARKER,
    ManagementClient,
)
from models import KeyGenerateBody, OrgNewBody, TagListEntry, TagNewBody, TeamNewBody, UserNewBody

pytestmark = pytest.mark.e2e

def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


def _generate_key(client: ManagementClient, resources: ResourceManager, body: KeyGenerateBody) -> str:
    key = client.proxy.generate_key(body)
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _create_team(client: ManagementClient, resources: ResourceManager, alias: str, models: list[str]) -> str:
    team_id = client.create_team(TeamNewBody(team_alias=alias, models=models))
    resources.defer(lambda: client.delete_team(team_id))
    return team_id


def _create_user(client: ManagementClient, resources: ResourceManager, body: UserNewBody) -> str:
    user_id = client.create_user(body)
    resources.defer(lambda: client.delete_user(user_id))
    return user_id


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
    def test_generate_persists_to_key_info_and_scopes_chat(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e-mgmt-key-{unique_marker()}"
        key = _generate_key(
            client,
            resources,
            KeyGenerateBody(models=["gemini-2.5-flash"], key_alias=alias, tpm_limit=424242, rpm_limit=424243),
        )

        info = client.proxy.key_info(key)
        assert info.key_alias == alias, f"/key/info reports key_alias {info.key_alias!r}, configured {alias!r}"
        assert info.models == ["gemini-2.5-flash"], (
            f"/key/info reports models {info.models}, configured ['gemini-2.5-flash']"
        )
        assert info.tpm_limit == 424242, (
            f"/key/info reports tpm_limit {info.tpm_limit}, configured 424242"
        )
        assert info.rpm_limit == 424243, (
            f"/key/info reports rpm_limit {info.rpm_limit}, configured 424243"
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

        info = client.proxy.key_info(key)
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


class TestKeyRegeneration:
    @pytest.mark.covers("mgmt.key.regenerate.happy_path")
    def test_regenerate_rotates_to_a_working_new_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        old_key = _generate_key(client, resources, KeyGenerateBody(models=["gpt-5.5"]))

        new_key = client.regenerate_key(old_key)
        resources.defer(lambda: client.proxy.delete_key(new_key))
        assert new_key != old_key, "regenerate returned the same key string, so no rotation happened"

        def new_accepted() -> bool | None:
            outcome = client.chat_status(new_key, "gpt-5.5", f"say hi {unique_marker()}")
            return True if outcome.status_code != 401 else None

        _ = _poll(client, new_accepted, "regenerated key was never accepted at auth (still 401) at the deadline")

        def old_rejected() -> bool | None:
            outcome = client.chat_status(old_key, "gpt-5.5", f"say hi {unique_marker()}")
            return True if outcome.status_code == 401 else None

        _ = _poll(
            client, old_rejected, "old key was still accepted after regeneration (never rejected 401) at the deadline"
        )


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
        key_info = client.proxy.key_info(key)
        assert key_info.team_id == team_id, (
            f"key generated under team {team_id} carries team_id {key_info.team_id!r} in /key/info"
        )

    @pytest.mark.covers("mgmt.team.member_add.persists")
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

    @pytest.mark.covers("mgmt.organization.delete.persists")
    def test_delete_removes_from_organization_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        """The teardown's deferred delete fires again on the already-deleted org by
        design: the deferred cleanup must survive this test failing before the
        in-body delete, and a repeat /organization/delete is a warn-only no-op the
        teardown absorbs."""
        org_id = client.create_org(OrgNewBody(organization_alias=f"e2e-mgmt-org-{unique_marker()}"))
        resources.defer(lambda: client.delete_org(org_id))

        assert client.org_info_status(org_id).status_code == 200, (
            f"/organization/info did not resolve org {org_id} before deletion"
        )

        client.delete_org(org_id)

        def gone() -> bool | None:
            return True if client.org_info_status(org_id).status_code == 404 else None

        _ = _poll(client, gone, f"org {org_id} still resolved on /organization/info after /organization/delete")


class TestTagRoutes:
    @pytest.mark.covers("mgmt.tag.new.happy_path")
    def test_new_persists_to_tag_list(self, client: ManagementClient, resources: ResourceManager) -> None:
        name = f"e2e-mgmt-tag-{unique_marker()}"
        description = "Tag for spend categorization"

        assert all(entry.name != name for entry in client.tag_list()), (
            f"tag {name!r} was already listed by /tag/list before /tag/new created it"
        )

        client.create_tag(TagNewBody(name=name, description=description))
        resources.defer(lambda: client.delete_tag(name))

        def listed() -> TagListEntry | None:
            return next((entry for entry in client.tag_list() if entry.name == name), None)

        entry = _poll(client, listed, f"/tag/list never listed {name!r} after /tag/new")
        assert entry.description == description, (
            f"/tag/list reports description {entry.description!r} for {name!r}, configured {description!r}"
        )


def _assert_route_forbidden(route: str, outcome: StreamingResponse) -> None:
    assert outcome.status_code == 403, (
        f"llm-only key POSTing {route} must be denied exactly 403, got {outcome.status_code}: {outcome.body[:300]}"
    )
    assert ROUTE_NOT_ALLOWED_MARKER in outcome.body, (
        f"{route} denial body must be a route-permission denial, got: {outcome.body[:300]}"
    )


class TestManagementRoutePermissions:
    @pytest.mark.covers("other.auth.virtual_key.route_permission_enforced")
    def test_llm_only_key_forbidden_from_management_writes(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.llm_only_key()
        resources.defer(lambda: client.proxy.delete_key(key))
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
