import asyncio
import json
from typing import Final

import pytest

from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.proxy._types import NewMCPServerWithProfilesRequest
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.mcp_server.auth_profiles import (
    DuplicateProfileId,
    InvalidProfileId,
    UnsupportedOAuth2Flow,
    UnsupportedProfileHost,
)
from litellm.proxy.mcp_server.auth_profile_db import SubmittedProfiles
from litellm.proxy.mcp_server.db import create_mcp_server_if_identifier_free, update_mcp_server
from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer
from tests.unit.proxy.mcp_prisma_fake import FakeMcpPrisma


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "auth-profile-db-test-salt")


def _bot(profile_id: str, **overrides: object) -> dict[str, object]:
    return {
        "id": profile_id,
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
        "client_id": f"{profile_id}-app",
        "client_secret": f"{profile_id}-secret",
        "token_url": "https://auth.example/token",
        **overrides,
    }


def _new_server(
    server_id: str, *profiles: dict[str, object] | None, **extra: object
) -> NewMCPServerWithProfilesRequest:
    return NewMCPServerWithProfilesRequest(
        server_id=server_id,
        server_name=server_id,
        url=f"https://mcp.example/{server_id}",
        transport=MCPTransport.http,
        additional_auth=[profile for profile in profiles if profile is not None] or None,
        **extra,
    )


def _sent(request: NewMCPServerWithProfilesRequest) -> SubmittedProfiles | None:
    return SubmittedProfiles(request.additional_auth) if "additional_auth" in request.model_fields_set else None


async def _create(store: FakeMcpPrisma, request: NewMCPServerWithProfilesRequest, touched_by: str):  # noqa: ANN202  # mirrors the writer's union
    return await create_mcp_server_if_identifier_free(store, request, touched_by, submitted_profiles=_sent(request))


def _stored_secret(store: FakeMcpPrisma, profile_id: str, key: str) -> str | None:
    row: Final = next(row for row in store.profiles if row["profile_id"] == profile_id)
    ciphertext: Final = json.loads(str(row["credentials"])).get(key)
    return None if ciphertext is None else decrypt_value_helper(ciphertext, key)


def _profile_ids(store: FakeMcpPrisma) -> dict[str, list[str]]:
    return {
        str(server["server_id"]): [
            str(p["profile_id"]) for p in store.profiles if p["server_id"] == server["server_id"]
        ]
        for server in store.servers
    }


@pytest.mark.asyncio
async def test_create_stores_profiles_with_encrypted_secrets_and_a_lowercase_key() -> None:
    store: Final = FakeMcpPrisma()

    created: Final = await _create(store, _new_server("analytics", _bot("Ana-Bots")), "admin")

    assert isinstance(created, LiteLLM_MCPServerTable)
    assert "additional_auth" not in store.servers[0]
    assert len(store.profiles) == 1
    row: Final = store.profiles[0]
    assert {
        key: row[key] for key in ("profile_id", "profile_id_normalized", "server_id", "auth_type", "oauth2_flow")
    } == {
        "profile_id": "Ana-Bots",
        "profile_id_normalized": "ana-bots",
        "server_id": "analytics",
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
    }
    assert "Ana-Bots-secret" not in str(row["credentials"])
    assert _stored_secret(store, "Ana-Bots", "client_secret") == "Ana-Bots-secret"
    assert _stored_secret(store, "Ana-Bots", "client_id") == "Ana-Bots-app"
    assert row["token_url"] == "https://auth.example/token"


@pytest.mark.asyncio
async def test_create_returns_its_profiles_when_the_reader_lags() -> None:
    store: Final = FakeMcpPrisma()
    store.lag_profile_reads()

    created: Final = await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")

    assert isinstance(created, LiteLLM_MCPServerTable)
    assert tuple(profile.id for profile in created.additional_auth or ()) == ("ana-bots",)


@pytest.mark.asyncio
async def test_a_pin_change_returns_the_server_with_its_profiles_when_the_reader_lags() -> None:
    from litellm.proxy.mcp_server.db import set_mcp_server_pinned_tools

    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    store.lag_profile_reads()

    unpinned: Final = await set_mcp_server_pinned_tools(store, "analytics", None, "admin")  # pyright: ignore[reportArgumentType]  # in-memory Prisma fake

    assert unpinned is not None
    assert tuple(profile.id for profile in unpinned.additional_auth or ()) == ("ana-bots",)


class _NoOAuthClients:
    async def find_many(self, **_: object) -> tuple[()]:
        return ()


@pytest.mark.asyncio
async def test_rotating_the_master_key_reencrypts_profile_secrets_and_moves_their_server_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.mcp_server.db import rotate_mcp_server_credentials_master_key

    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("plain"), "admin")
    await _create(
        store,
        _new_server("tickets", {"id": "tickets-key", "auth_type": "bearer_token", "authentication_token": "tok"}),
        "admin",
    )
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    monkeypatch.setattr(store.db, "litellm_mcpserveroauthclient", _NoOAuthClients(), raising=False)
    store.lag_profile_reads()

    await rotate_mcp_server_credentials_master_key(store, "rotator", "rotated-master-key")  # pyright: ignore[reportArgumentType]  # in-memory Prisma fake
    monkeypatch.setenv("LITELLM_SALT_KEY", "rotated-master-key")

    assert (
        _stored_secret(store, "tickets-key", "auth_value"),
        _stored_secret(store, "ana-bots", "client_id"),
        _stored_secret(store, "ana-bots", "client_secret"),
    ) == ("tok", "ana-bots-app", "ana-bots-secret")
    assert [row["updated_by"] for row in store.profiles] == ["rotator", "rotator"]
    assert {row["server_id"]: row["updated_by"] for row in store.servers} == {
        "plain": "admin",
        "tickets": "rotator",
        "analytics": "rotator",
    }


@pytest.mark.asyncio
async def test_create_without_profiles_writes_exactly_what_main_wrote() -> None:
    store: Final = FakeMcpPrisma()

    created: Final = await _create(store, _new_server("plain"), "admin")

    assert isinstance(created, LiteLLM_MCPServerTable)
    assert "auth_profiles" not in store.servers[0]
    assert store.profiles == []
    assert created.additional_auth is None
    assert "additional_auth" not in created.model_dump()


@pytest.mark.asyncio
async def test_create_refuses_an_id_another_server_holds_ignoring_case_and_stores_nothing() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")

    refused: Final = await _create(store, _new_server("github", _bot("ANA-Bots")), "admin")

    assert refused == DuplicateProfileId("ANA-Bots", "github", "ana-bots", "analytics")
    assert _profile_ids(store) == {"analytics": ["ana-bots"]}


@pytest.mark.asyncio
async def test_an_underscore_in_an_id_is_a_literal_not_a_wildcard() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("a", _bot("ANAxbots")), "admin")

    second: Final = await _create(store, _new_server("b", _bot("ANA_Bots")), "admin")
    third: Final = await _create(store, _new_server("c", _bot("ANA_bots")), "admin")

    assert isinstance(second, LiteLLM_MCPServerTable)
    assert third == DuplicateProfileId("ANA_bots", "c", "ANA_Bots", "b")


def _yaml_server_holding(monkeypatch: pytest.MonkeyPatch, profile_id: str) -> None:
    monkeypatch.setitem(
        global_mcp_server_manager.config_mcp_servers,
        "yaml-server",
        MCPServer(
            server_id="yaml-server",
            name="yaml_server",
            server_name="yaml_server",
            transport=MCPTransport.http,
            additional_auth=(MCPAuthProfile(id=profile_id, auth_type="api_key", authentication_token="k"),),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("server_id", ["github", "yaml_server"], ids=["other-name", "server-id-is-the-yaml-name"])
async def test_create_refuses_an_id_that_config_yaml_holds(monkeypatch: pytest.MonkeyPatch, server_id: str) -> None:
    _yaml_server_holding(monkeypatch, "Yaml-Bots")
    store: Final = FakeMcpPrisma()

    refused: Final = await _create(store, _new_server(server_id, _bot("yaml-bots")), "admin")

    assert refused == DuplicateProfileId("yaml-bots", server_id, "Yaml-Bots", "yaml_server")
    assert store.servers == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_profile", "expected"),
    [
        pytest.param(_bot("default"), InvalidProfileId("srv", 1, "default"), id="reserved"),
        pytest.param(
            _bot("p2", oauth2_flow="authorization_code"),
            UnsupportedOAuth2Flow("srv", 1, "p2", "authorization_code"),
            id="interactive",
        ),
        pytest.param(_bot("Good"), DuplicateProfileId("Good", "srv", "good", "srv"), id="same-server-duplicate"),
    ],
)
async def test_one_good_and_one_bad_profile_store_nothing(bad_profile: dict[str, object], expected: object) -> None:
    store: Final = FakeMcpPrisma()

    refused: Final = await _create(store, _new_server("srv", _bot("good"), bad_profile), "admin")

    assert refused == expected
    assert (store.servers, store.profiles) == ([], [])


@pytest.mark.asyncio
async def test_two_concurrent_creates_of_colliding_ids_leave_one_winner_and_one_conflict() -> None:
    store: Final = FakeMcpPrisma()
    store.hold_profile_lookups_until(2)

    first, second = await asyncio.gather(
        _create(store, _new_server("a", _bot("Ana-Bots")), "admin"),
        _create(store, _new_server("b", _bot("ana-bots")), "admin"),
    )

    winner, loser = (first, second) if isinstance(first, LiteLLM_MCPServerTable) else (second, first)
    assert isinstance(winner, LiteLLM_MCPServerTable)
    assert isinstance(loser, DuplicateProfileId)
    assert loser.other_server == winner.server_id
    assert _profile_ids(store) == {winner.server_id: [loser.other_profile_id]}


@pytest.mark.asyncio
async def test_a_create_that_loses_a_profile_race_to_a_write_the_reader_has_not_seen_is_a_conflict() -> None:
    store: Final = FakeMcpPrisma()
    store.lag_profile_reads()
    await _create(store, _new_server("a", _bot("Ana-Bots")), "admin")

    refused: Final = await _create(store, _new_server("b", _bot("ana-bots")), "admin")

    assert refused == DuplicateProfileId("ana-bots", "b", "Ana-Bots", "a")
    assert _profile_ids(store) == {"a": ["Ana-Bots"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("profiles", [(), ("two",)], ids=["without-profiles", "with-profiles"])
async def test_a_losing_server_id_race_is_not_reported_as_a_profile_conflict(profiles: tuple[str, ...]) -> None:
    from prisma.errors import UniqueViolationError

    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("a", _bot("one")), "admin")

    with pytest.raises(UniqueViolationError):
        await _create(store, _new_server("a", *(_bot(profile_id) for profile_id in profiles)), "admin")

    assert _profile_ids(store) == {"a": ["one"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_fields", "host"),
    [
        pytest.param({"is_byok": True}, "BYOK", id="byok"),
        pytest.param({"spec_path": "./openapi.json"}, "OpenAPI", id="openapi"),
    ],
)
async def test_profiles_are_refused_on_a_server_that_never_sends_a_profile_credential(
    host_fields: dict[str, object], host: str
) -> None:
    store: Final = FakeMcpPrisma()

    created: Final = await _create(store, _new_server("tools", _bot("T"), **host_fields), "admin")

    assert created == UnsupportedProfileHost("tools", host)
    assert (store.servers, store.profiles) == ([], [])


def test_routes_outside_post_put_and_register_drop_additional_auth_as_on_main() -> None:
    from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest

    profile: Final = [_bot("A")]

    assert (
        "additional_auth"
        not in NewMCPServerRequest(server_name="s", url="https://x", additional_auth=profile).model_dump()
    )
    assert "additional_auth" not in UpdateMCPServerRequest(server_id="s", additional_auth=profile).model_dump()
    assert "additional_auth" not in _new_server("s", _bot("A")).model_dump()
