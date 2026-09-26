"""auto_register with auto_register_map_existing_key binds the JWT claim to the user's existing key.

`unregistered_jwt_client_behavior: auto_register` on `virtual_key_claim_field: sub` mints a fresh
virtual key on the user's first JWT call. With `auto_register_map_existing_key: true` the proxy must
instead point the new JWT mapping at a key the resolved user already owns, and mint only when the
user has none. Each behavior gets its own gateway because the flag lives in `litellm_jwtauth`, so
this file boots two owned proxies against the shared database and Keycloak realm.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import ExitStack
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from idp import Identity, Keycloak
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, JwtKeyMappingRow, KeyGenerateBody, TeamNewBody, UserNewBody
from other_client import OtherClient
from owned_jwt_gateway import MODEL_NAME, OwnedJwtGateway, owned_jwt_gateway

pytestmark = pytest.mark.e2e

_JWT_COMMON: Final = (
    "user_id_jwt_field: sub\n"
    "user_email_jwt_field: email\n"
    "team_ids_jwt_field: groups\n"
    "user_id_upsert: true\n"
    "virtual_key_claim_field: sub\n"
    "unregistered_jwt_client_behavior: auto_register"
)


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _ping() -> ChatBody:
    return ChatBody(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content=f"Reply with the single word ok. {unique_marker()}")],
        max_tokens=5,
    )


def _identity_with_user(idp: Keycloak, client: OtherClient, resources: ResourceManager) -> Identity:
    """An IdP identity plus the litellm user and team its claims resolve to, with
    teardown that also sweeps the user's keys and JWT mapping rows the proxy
    wrote, since those outlive the user row itself."""
    marker: Final = unique_marker()
    identity: Final = idp.provision(marker=marker, group=f"e2e-jwt-team-{marker}", defer=resources.defer)
    resources.defer(lambda: client.proxy.delete_user(identity.user_id))
    team_id: Final = client.proxy.create_team(TeamNewBody(team_alias=f"e2e-jwt-{marker}", team_id=identity.group))
    resources.defer(lambda: client.proxy.delete_team(team_id))
    unwrap(
        client.user_new(
            UserNewBody(
                user_id=identity.user_id,
                user_email=f"{identity.username}@example.com",
                user_role="internal_user",
                auto_create_key=False,
            )
        )
    )

    def delete_user_keys() -> None:
        for row in unwrap(client.user_info(identity.user_id)).keys:
            client.proxy.delete_key(row.token)

    def delete_user_mappings() -> None:
        for mapping in unwrap(client.jwt_mapping_list()).mappings:
            if mapping.jwt_claim_value == identity.user_id:
                _ = client.jwt_mapping_delete(mapping.id)

    resources.defer(delete_user_keys)
    resources.defer(delete_user_mappings)
    return identity


def _mapping_for(client: OtherClient, claim_value: str) -> JwtKeyMappingRow | None:
    return next(
        (row for row in unwrap(client.jwt_mapping_list()).mappings if row.jwt_claim_value == claim_value),
        None,
    )


@pytest.fixture(scope="module")
def mapping_gateway(idp: Keycloak, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedJwtGateway]:
    with ExitStack() as cleanup:
        yield owned_jwt_gateway(
            idp,
            tmp_path_factory.mktemp("jwt-mapping"),
            cleanup,
            litellm_jwtauth=f"{_JWT_COMMON}\nauto_register_map_existing_key: true",
            name="jwt-mapping-gateway",
        )


@pytest.fixture(scope="module")
def minting_gateway(idp: Keycloak, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedJwtGateway]:
    with ExitStack() as cleanup:
        yield owned_jwt_gateway(
            idp,
            tmp_path_factory.mktemp("jwt-minting"),
            cleanup,
            litellm_jwtauth=_JWT_COMMON,
            name="jwt-minting-gateway",
        )


@pytest.mark.owned_gateway
class TestJwtAutoRegisterMapExistingKey:
    @pytest.mark.covers("other.auth.jwt.auto_register_maps_existing_key")
    def test_first_jwt_call_maps_to_the_users_existing_key_and_mints_none(
        self, client: OtherClient, idp: Keycloak, resources: ResourceManager, mapping_gateway: OwnedJwtGateway
    ) -> None:
        identity: Final = _identity_with_user(idp, client, resources)
        existing_key: Final = client.proxy.generate_key(
            KeyGenerateBody(user_id=identity.user_id, key_alias=f"e2e-jwt-existing-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(existing_key))

        response: Final = unwrap(mapping_gateway.proxy.chat(idp.access_token(identity), _ping()))

        keys: Final = unwrap(client.user_info(identity.user_id)).keys
        assert [row.token for row in keys] == [_key_hash(existing_key)], (
            f"map_existing_key must leave the user with only their pre-existing key, got {keys}"
        )
        mapping: Final = _mapping_for(client, identity.user_id)
        assert mapping is not None, (
            f"no JWT mapping row for sub={identity.user_id}: {unwrap(client.jwt_mapping_list())}"
        )
        assert mapping.jwt_claim_name == "sub", f"mapping must bind the sub claim, got {mapping}"
        assert mapping.created_by == "auto_register", f"mapping must be written by auto_register, got {mapping}"
        rows: Final = client.proxy.poll_logs_for_key(existing_key)
        assert any(row.request_id == response.id for row in rows), (
            f"the JWT chat must be billed to the user's existing key, spend rows for it: {rows}"
        )

    @pytest.mark.covers("other.auth.jwt.auto_register_mints_when_keyless")
    def test_first_jwt_call_mints_a_key_when_the_user_has_none(
        self, client: OtherClient, idp: Keycloak, resources: ResourceManager, mapping_gateway: OwnedJwtGateway
    ) -> None:
        identity: Final = _identity_with_user(idp, client, resources)

        response: Final = unwrap(mapping_gateway.proxy.chat(idp.access_token(identity), _ping()))
        assert response.choices, f"JWT chat returned no completion: {response}"

        keys: Final = unwrap(client.user_info(identity.user_id)).keys
        assert len(keys) == 1, f"a keyless user must get exactly one minted key, got {keys}"
        mapping: Final = _mapping_for(client, identity.user_id)
        assert mapping is not None and mapping.jwt_claim_name == "sub", (
            f"the minted key must be recorded as a sub-claim mapping, mappings: {unwrap(client.jwt_mapping_list())}"
        )

    @pytest.mark.covers("other.auth.jwt.auto_register_default_mints")
    def test_default_behavior_still_mints_when_the_user_already_has_a_key(
        self, client: OtherClient, idp: Keycloak, resources: ResourceManager, minting_gateway: OwnedJwtGateway
    ) -> None:
        identity: Final = _identity_with_user(idp, client, resources)
        existing_key: Final = client.proxy.generate_key(
            KeyGenerateBody(user_id=identity.user_id, key_alias=f"e2e-jwt-existing-{unique_marker()}")
        )
        resources.defer(lambda: client.proxy.delete_key(existing_key))

        response: Final = unwrap(minting_gateway.proxy.chat(idp.access_token(identity), _ping()))
        assert response.id is not None, f"JWT chat returned no response id: {response}"

        keys: Final = unwrap(client.user_info(identity.user_id)).keys
        assert len(keys) == 2, (
            f"default auto_register must mint a second key for a user who already has one, got {keys}"
        )
        rows: Final = client.proxy.poll_logs_for_request_id(response.id)
        assert rows and all(row.api_key != _key_hash(existing_key) for row in rows), (
            f"the default path must bill the minted key, not the user's existing one: {rows}"
        )
