"""Management writes and tenant isolation under credentials issued by Keycloak."""

from __future__ import annotations

from typing import Final, Literal

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, PROXY_BASE_URL, unique_marker
from e2e_http import UnauthorizedError, UnknownApiError, unwrap
from idp import ADMIN_CLIENT_ID, Identity, Keycloak, token_claims
from lifecycle import ResourceManager
from management.jwt_actors import ActorFactory, ActorRole
from management_client import ManagementClient
from models import KeyGenerateBody, KeyUpdateBody, TeamNewBody, UserInfoParams, UserInfoResponse, UserNewBody
from proxy_client import Caller

pytestmark = pytest.mark.e2e


class TestJwtManagement:
    @pytest.mark.parametrize(
        "role",
        (
            "proxy_admin",
            "proxy_admin_viewer",
            "organization_admin",
            "team_admin",
            "team_member",
            "internal_user",
            "internal_user_viewer",
            "unrelated_user",
        ),
    )
    @pytest.mark.covers("mgmt.user.jwt.database_roles")
    def test_actor_subject_and_database_role(self, actor_factory: ActorFactory, role: ActorRole) -> None:
        tenants: Final = (
            (actor_factory.tenant(),) if role in ("organization_admin", "team_admin", "team_member") else ()
        )
        actor: Final = actor_factory.create(role, tenants=tenants)
        caller: Final = actor.mint_caller(actor_factory.idp)
        claims: Final = token_claims(caller.credential)
        assert claims.sub == actor.identity.user_id
        assert claims.iss == actor_factory.idp.issuer
        assert claims.aud == "litellm-e2e" or "litellm-e2e" in claims.aud
        assert actor.identity.groups == ()
        assert ("litellm_proxy_admin" in claims.scope.split()) == (role == "proxy_admin")
        stored: Final = actor_factory.bootstrap.user_info(actor.identity.user_id)
        assert stored.user_id == actor.identity.user_id
        assert stored.user_info.user_role == actor.global_role
        bound: Final = actor_factory.bootstrap.with_caller(caller)
        own: Final = unwrap(
            bound.proxy.transport.get(
                "/user/info",
                headers=bound.proxy.management_headers(),
                params=UserInfoParams(),
                response_type=UserInfoResponse,
            )
        )
        assert own.user_id == actor.identity.user_id
        assert own.user_info.user_role == actor.global_role
        for tenant in tenants:
            info = actor_factory.bootstrap.team_info(tenant.team_id)
            assert info.organization_id == tenant.organization_id
            assert {(member.user_id, member.role) for member in info.members_with_roles} == {
                (actor.identity.user_id, "admin" if role == "team_admin" else "user")
            }
            assert {
                (member.user_id, member.user_role)
                for member in actor_factory.bootstrap.org_info(tenant.organization_id).members
            } == {(actor.identity.user_id, "org_admin" if role == "organization_admin" else "internal_user")}

    @pytest.mark.covers("mgmt.key.jwt.viewer_denied")
    def test_admin_viewer_reads_but_cannot_update(self, actor_factory: ActorFactory) -> None:
        actor: Final = actor_factory.create("proxy_admin_viewer")
        viewer: Final = actor_factory.bootstrap.with_caller(actor.mint_caller(actor_factory.idp))
        alias: Final = f"e2e-viewer-{unique_marker()}"
        key: Final = actor_factory.key().key
        unwrap(actor_factory.bootstrap.update_key(KeyUpdateBody(key=key, key_alias=alias)))
        assert viewer.proxy.key_info(key).key_alias == alias
        denied: Final = viewer.update_key(KeyUpdateBody(key=key, key_alias="forbidden"))
        assert isinstance(denied, UnknownApiError) and denied.status_code == 403, f"viewer write was accepted: {denied}"
        assert "proxy_admin_viewer" in denied.body and "/key/update" in denied.body
        assert actor_factory.bootstrap.proxy.key_info(key).key_alias == alias

    @pytest.mark.covers("mgmt.user.oidc.identity_mapping")
    def test_oidc_browser_profile_identity_mapping(self, actor_factory: ActorFactory) -> None:
        actor: Final = actor_factory.create("internal_user")
        idp: Final = actor_factory.idp.with_strict_cleanup()
        discovery: Final = idp.discovery()
        assert discovery.issuer == idp.issuer
        assert discovery.jwks_uri == idp.jwks_url
        callback: Final = f"{PROXY_BASE_URL}/sso/callback"
        browser: Final = idp.browser_client(callback_url=callback, defer=actor_factory.resources.defer)
        token: Final = idp.browser_token(actor.identity, browser)
        assert token_claims(token).sub == actor.identity.user_id
        userinfo: Final = idp.userinfo(token)
        assert userinfo.sub == actor.identity.user_id
        assert userinfo.email == f"{actor.identity.username}@example.com"
        assert browser.environment(discovery)["GENERIC_USER_ID_ATTRIBUTE"] == "sub"

    @pytest.mark.covers("mgmt.key.jwt.lifecycle")
    @pytest.mark.parametrize("credential_kind", ("direct_jwt", "virtual_key"))
    def test_admin_creates_reads_updates_clears_and_deletes_a_key(
        self,
        client: ManagementClient,
        idp: Keycloak,
        jwt_identity: Identity,
        resources: ResourceManager,
        actor_factory: ActorFactory,
        credential_kind: Literal["direct_jwt", "virtual_key"],
    ) -> None:
        actor: Final = actor_factory.create("proxy_admin")
        virtual_key: Final = (
            actor_factory.key(user_id=actor.identity.user_id).key if credential_kind == "virtual_key" else None
        )
        admin: Final = (
            virtual_key if virtual_key is not None else idp.access_token(jwt_identity, client_id=ADMIN_CLIENT_ID)
        )
        bound: Final = client.with_caller(Caller(credential=admin, kind=credential_kind, role="proxy_admin"))
        alias: Final = f"e2e-jwt-key-{unique_marker()}"
        created: Final = unwrap(
            bound.generate_key(
                KeyGenerateBody(key_alias=alias, team_id=jwt_identity.group, models=[CHEAP_OPENAI_MODEL]),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(created.key))

        original: Final = unwrap(bound.key_info_as(created.key)).info
        assert original.key_alias == alias and original.team_id == jwt_identity.group
        assert original.models == [CHEAP_OPENAI_MODEL]

        updated_alias: Final = f"{alias}-updated"
        unwrap(bound.update_key(KeyUpdateBody(key=created.key, key_alias=updated_alias, rpm_limit=120)))
        updated: Final = unwrap(bound.key_info_as(created.key)).info
        assert updated.key_alias == updated_alias and updated.rpm_limit == 120
        assert updated.models == [CHEAP_OPENAI_MODEL], "omitted models must preserve the restriction"

        unwrap(bound.update_key(KeyUpdateBody(key=created.key, models=[])))
        cleared: Final = unwrap(bound.key_info_as(created.key)).info
        assert cleared.models == [] and cleared.rpm_limit == 120

        assert unwrap(bound.key_list(updated_alias)).total_count == 1
        bound.delete_key_strict(created.key)
        assert unwrap(bound.key_list(updated_alias)).total_count == 0

    @pytest.mark.covers("mgmt.team.jwt.tenant_isolation")
    def test_two_actor_sets_keep_tenants_and_keys_isolated(self, actor_factory: ActorFactory) -> None:
        first: Final = actor_factory.tenant()
        second: Final = actor_factory.tenant()
        assert first.organization_id != second.organization_id and first.team_id != second.team_id
        actors: Final = tuple(
            actor_factory.create("team_member", tenants=(tenant,), profile="group_scoped") for tenant in (first, second)
        )
        assert actors[0].identity.user_id != actors[1].identity.user_id
        callers: Final = tuple(
            actor_factory.bootstrap.with_caller(actor.mint_caller(actor_factory.idp)) for actor in actors
        )
        keys: Final = tuple(actor_factory.key(tenant) for tenant in (first, second))
        assert keys[0].key != keys[1].key
        assert callers[0].proxy.key_info(keys[0].key).team_id == first.team_id
        assert callers[1].proxy.key_info(keys[1].key).team_id == second.team_id
        for caller, other_key in ((callers[0], keys[1].key), (callers[1], keys[0].key)):
            hidden = caller.key_info_as(other_key)
            assert isinstance(hidden, UnknownApiError) and hidden.status_code == 403
        assert tuple(actor.identity.groups for actor in actors) == ((first.team_id,), (second.team_id,))

    @pytest.mark.covers("mgmt.team.jwt.multiple_memberships")
    def test_multi_group_actor_keeps_exact_memberships(self, actor_factory: ActorFactory) -> None:
        tenants: Final = (actor_factory.tenant(), actor_factory.tenant())
        actor: Final = actor_factory.create("team_member", tenants=tenants, profile="group_scoped")
        claims: Final = token_claims(actor.mint_caller(actor_factory.idp).credential)
        assert set(claims.groups) == {tenant.team_id for tenant in tenants}
        assert "litellm_proxy_admin" not in claims.scope.split()
        assert actor.identity.groups == tuple(tenant.team_id for tenant in tenants)
        for tenant in tenants:
            assert {
                (entry.user_id, entry.role)
                for entry in actor_factory.bootstrap.team_info(tenant.team_id).members_with_roles
            } == {(actor.identity.user_id, "user")}

    @pytest.mark.covers("mgmt.user.jwt.cleanup")
    def test_successful_actor_cleanup_removes_owned_state(self, actor_factory: ActorFactory) -> None:
        resources: Final = ResourceManager(client=actor_factory.bootstrap.proxy, strict_cleanup=True)
        factory: Final = ActorFactory(bootstrap=actor_factory.bootstrap, idp=actor_factory.idp, resources=resources)
        try:
            tenant: Final = factory.tenant()
            actor: Final = factory.create("team_member", tenants=(tenant,), profile="group_scoped")
            key: Final = factory.key(tenant)
            alias: Final = factory.bootstrap.proxy.key_info(key.key).key_alias
            assert alias is not None
        finally:
            resources.teardown()
        assert factory.bootstrap.user_count(actor.identity.user_id) == 0
        assert factory.bootstrap.key_alias_count(alias) == 0
        assert factory.bootstrap.team_info_status(tenant.team_id).status_code == 404
        assert factory.bootstrap.org_info_status(tenant.organization_id).status_code == 404
        factory.idp.assert_absent("users", actor.identity.user_id)
        factory.idp.assert_absent("groups", tenant.group_id)

    @pytest.mark.parametrize("stage", ("group", "user"))
    @pytest.mark.covers("mgmt.user.jwt.partial_cleanup")
    def test_partial_setup_removes_previously_created_identities(
        self,
        actor_factory: ActorFactory,
        stage: Literal["group", "user"],
    ) -> None:
        idp: Final = actor_factory.idp.with_strict_cleanup()
        resources: Final = ResourceManager(client=actor_factory.bootstrap.proxy, strict_cleanup=True)
        marker: Final = unique_marker()
        group_id: Final = idp.create_group(f"e2e-partial-{marker}")
        resources.defer(lambda: idp.delete_group(group_id))
        try:
            identity: Final = (
                idp.provision_user(
                    marker=marker,
                    groups=(f"e2e-partial-{marker}",),
                    group_ids=(group_id,),
                    defer=resources.defer,
                )
                if stage == "user"
                else None
            )
            if identity is None:
                with pytest.raises(pytest.fail.Exception, match="HTTP 409"):
                    idp.create_group(f"e2e-partial-{marker}")
            else:
                with pytest.raises(pytest.fail.Exception, match="HTTP 409"):
                    idp.create_user(
                        username=identity.username,
                        email=f"{identity.username}@example.com",
                        password=identity.password,
                        groups=identity.groups,
                    )
        finally:
            resources.teardown()
        idp.assert_absent("groups", group_id)
        if identity is not None:
            idp.assert_absent("users", identity.user_id)

    @pytest.mark.covers("mgmt.key.jwt.member_denied", "mgmt.key.jwt.other_team_denied")
    def test_member_cannot_write_and_another_team_cannot_read_the_key(
        self, client: ManagementClient, idp: Keycloak, jwt_identity: Identity, resources: ResourceManager
    ) -> None:
        admin: Final = idp.access_token(jwt_identity, client_id=ADMIN_CLIENT_ID)
        bound: Final = client.with_caller(Caller(credential=admin, kind="direct_jwt", role="proxy_admin"))
        member: Final = idp.access_token(jwt_identity)
        member_client: Final = client.with_caller(Caller(credential=member, kind="direct_jwt", role="team_member"))
        alias: Final = f"e2e-jwt-owned-{unique_marker()}"
        created: Final = unwrap(
            client.generate_key(KeyGenerateBody(key_alias=alias, team_id=jwt_identity.group), caller_key=admin)
        )
        resources.defer(lambda: client.proxy.delete_key(created.key))

        client.add_team_member(jwt_identity.group, jwt_identity.user_id)
        assert unwrap(member_client.key_info_as(created.key)).info.key_alias == alias

        refused: Final = member_client.update_key(KeyUpdateBody(key=created.key, key_alias="forbidden"))
        assert isinstance(refused, UnauthorizedError), f"member write was accepted: {refused}"
        assert "does not have permissions for endpoint" in refused.body.lower(), (
            f"expected a permission denial: {refused}"
        )
        assert unwrap(bound.key_info_as(created.key)).info.key_alias == alias

        marker: Final = unique_marker()
        outsider: Final = idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}", defer=resources.defer)
        resources.defer(lambda: client.proxy.delete_user(outsider.user_id))
        client.create_user(
            UserNewBody(
                user_id=outsider.user_id, user_email=f"{outsider.username}@example.com", user_role="internal_user"
            )
        )
        team_id: Final = client.proxy.create_team(TeamNewBody(team_alias=marker, team_id=outsider.group))
        resources.defer(lambda: client.proxy.delete_team(team_id))
        client.add_team_member(outsider.group, outsider.user_id)
        outsider_token: Final = idp.access_token(outsider)
        hidden: Final = client.key_info_as(created.key, caller_key=outsider_token)
        assert isinstance(hidden, UnknownApiError) and hidden.status_code == 403, (
            f"another team must not read this key: {hidden}"
        )
        assert unwrap(bound.key_info_as(created.key)).info.team_id == jwt_identity.group
