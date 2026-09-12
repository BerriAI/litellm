"""Management writes and tenant isolation under credentials issued by Keycloak."""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import UnauthorizedError, UnknownApiError, unwrap
from idp import ADMIN_CLIENT_ID, Identity, Keycloak
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, KeyUpdateBody, TeamNewBody, UserNewBody

pytestmark = pytest.mark.e2e


class TestJwtManagement:
    @pytest.mark.covers("mgmt.key.jwt.lifecycle")
    def test_admin_creates_reads_updates_clears_and_deletes_a_key(
        self, client: ManagementClient, idp: Keycloak, jwt_identity: Identity, resources: ResourceManager
    ) -> None:
        admin: Final = idp.access_token(jwt_identity, client_id=ADMIN_CLIENT_ID)
        alias: Final = f"e2e-jwt-key-{unique_marker()}"
        created: Final = unwrap(
            client.generate_key(
                KeyGenerateBody(key_alias=alias, team_id=jwt_identity.group, models=[CHEAP_OPENAI_MODEL]),
                caller_key=admin,
            )
        )
        resources.defer(lambda: client.proxy.delete_key(created.key))

        original: Final = unwrap(client.key_info_as(created.key, caller_key=admin)).info
        assert original.key_alias == alias and original.team_id == jwt_identity.group
        assert original.models == [CHEAP_OPENAI_MODEL]

        updated_alias: Final = f"{alias}-updated"
        unwrap(
            client.update_key(KeyUpdateBody(key=created.key, key_alias=updated_alias, rpm_limit=120), caller_key=admin)
        )
        updated: Final = unwrap(client.key_info_as(created.key, caller_key=admin)).info
        assert updated.key_alias == updated_alias and updated.rpm_limit == 120
        assert updated.models == [CHEAP_OPENAI_MODEL], "omitted models must preserve the restriction"

        unwrap(client.update_key(KeyUpdateBody(key=created.key, models=[]), caller_key=admin))
        cleared: Final = unwrap(client.key_info_as(created.key, caller_key=admin)).info
        assert cleared.models == [] and cleared.rpm_limit == 120

        assert unwrap(client.key_list(updated_alias, caller_key=admin)).total_count == 1
        client.delete_key_strict(created.key, caller_key=admin)
        assert unwrap(client.key_list(updated_alias, caller_key=admin)).total_count == 0

    @pytest.mark.covers("mgmt.key.jwt.member_denied", "mgmt.key.jwt.other_team_denied")
    def test_member_cannot_write_and_another_team_cannot_read_the_key(
        self, client: ManagementClient, idp: Keycloak, jwt_identity: Identity, resources: ResourceManager
    ) -> None:
        admin: Final = idp.access_token(jwt_identity, client_id=ADMIN_CLIENT_ID)
        member: Final = idp.access_token(jwt_identity)
        alias: Final = f"e2e-jwt-owned-{unique_marker()}"
        created: Final = unwrap(
            client.generate_key(KeyGenerateBody(key_alias=alias, team_id=jwt_identity.group), caller_key=admin)
        )
        resources.defer(lambda: client.proxy.delete_key(created.key))

        client.add_team_member(jwt_identity.group, jwt_identity.user_id)
        assert unwrap(client.key_info_as(created.key, caller_key=member)).info.key_alias == alias

        refused: Final = client.update_key(KeyUpdateBody(key=created.key, key_alias="forbidden"), caller_key=member)
        assert isinstance(refused, UnauthorizedError), f"member write was accepted: {refused}"
        assert "does not have permissions for endpoint" in refused.body.lower(), (
            f"expected a permission denial: {refused}"
        )
        assert unwrap(client.key_info_as(created.key, caller_key=admin)).info.key_alias == alias

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
        assert unwrap(client.key_info_as(created.key, caller_key=admin)).info.team_id == jwt_identity.group
