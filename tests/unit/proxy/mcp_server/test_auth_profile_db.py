import asyncio
import json
from typing import Final

import pytest

from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.proxy._types import NewMCPServerWithProfilesRequest, UpdateMCPServerWithProfilesRequest
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.mcp_server.auth_profiles import (
    DuplicateProfileId,
    InvalidProfileId,
    MalformedProfile,
    UnsupportedOAuth2Flow,
    UnsupportedProfileHost,
)
from litellm.proxy.mcp_server.auth_profile_db import SubmittedProfiles
from litellm.proxy.mcp_server.db import McpIdentifierConflict, create_mcp_server_if_identifier_free, update_mcp_server
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


def _sent(
    request: NewMCPServerWithProfilesRequest | UpdateMCPServerWithProfilesRequest,
) -> SubmittedProfiles | None:
    return SubmittedProfiles(request.additional_auth) if "additional_auth" in request.model_fields_set else None


async def _create(store: FakeMcpPrisma, request: NewMCPServerWithProfilesRequest, touched_by: str):  # noqa: ANN202  # mirrors the writer's union
    return await create_mcp_server_if_identifier_free(store, request, touched_by, submitted_profiles=_sent(request))


async def _update(store: FakeMcpPrisma, request: UpdateMCPServerWithProfilesRequest, touched_by: str):  # noqa: ANN202  # mirrors the writer's union
    return await update_mcp_server(store, request, touched_by, submitted_profiles=_sent(request))


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


async def _two_profile_server(store: FakeMcpPrisma) -> None:
    created: Final = await _create(store, _new_server("analytics", _bot("A"), _bot("B")), "admin")
    assert isinstance(created, LiteLLM_MCPServerTable)


@pytest.mark.asyncio
async def test_put_without_additional_auth_leaves_the_profiles_alone() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    before: Final = [dict(row) for row in store.profiles]

    updated: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", description="News"), "admin"
    )

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert store.profiles == before
    assert [profile.id for profile in updated.additional_auth or ()] == ["A", "B"]


@pytest.mark.asyncio
async def test_put_of_both_profiles_without_secrets_keeps_both_secrets_and_changes_their_scopes() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    secrets_before: Final = {
        (pid, key): _stored_secret(store, pid, key) for pid in ("A", "B") for key in ("client_id", "client_secret")
    }

    updated: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics",
            additional_auth=[
                {k: v for k, v in _bot("A", scopes=["read"]).items() if "client_" not in k},
                {k: v for k, v in _bot("B").items() if k != "client_secret"} | {"client_id": ""},
            ],
        ),
        "admin",
    )

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert {
        (pid, key): _stored_secret(store, pid, key) for pid in ("A", "B") for key in ("client_id", "client_secret")
    } == secrets_before
    assert {str(row["profile_id"]): row["scopes"] for row in store.profiles} == {"A": ["read"], "B": []}


@pytest.mark.asyncio
async def test_put_that_repoints_the_token_url_needs_the_client_credentials_again() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots", scopes=["read"])), "admin")
    before: Final = [dict(row) for row in store.profiles]
    moved: Final = {
        k: v for k, v in _bot("ana-bots", token_url="https://other.example/token").items() if "client_" not in k
    }

    refused: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[moved]), "admin"
    )
    unchanged: Final = [dict(row) for row in store.profiles]
    accepted: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics", additional_auth=[{**moved, "client_id": "new-app", "client_secret": "new-secret"}]
        ),
        "admin",
    )

    assert refused == MalformedProfile(
        "analytics", 0, "profile 'ana-bots' changes its auth settings, so client_secret must be supplied"
    )
    assert unchanged == before
    assert isinstance(accepted, LiteLLM_MCPServerTable)
    assert (
        store.profiles[0]["token_url"],
        store.profiles[0]["scopes"],
        _stored_secret(store, "ana-bots", "client_id"),
        _stored_secret(store, "ana-bots", "client_secret"),
    ) == ("https://other.example/token", [], "new-app", "new-secret")


@pytest.mark.asyncio
async def test_put_changing_only_the_id_case_keeps_the_stored_spelling_and_secret() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    resent: Final = {k: v for k, v in _bot("ANA-Bots").items() if "client_" not in k}

    updated: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[resent]), "admin"
    )

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert [profile.id for profile in updated.additional_auth or ()] == ["ana-bots"]
    assert _profile_ids(store) == {"analytics": ["ana-bots"]}
    assert _stored_secret(store, "ana-bots", "client_secret") == "ana-bots-secret"


@pytest.mark.asyncio
async def test_put_merges_secrets_from_the_writer_when_the_reader_lags() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    store.lag_profile_reads()
    resent: Final = {k: v for k, v in _bot("ana-bots").items() if "client_" not in k}

    await _update(store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[resent]), "admin")

    assert _stored_secret(store, "ana-bots", "client_secret") == "ana-bots-secret"


@pytest.mark.asyncio
async def test_put_returns_the_new_secret_when_the_reader_lags() -> None:
    store: Final = FakeMcpPrisma()
    static: Final = {"id": "ana-key", "auth_type": "bearer_token"}
    await _create(store, _new_server("analytics", {**static, "authentication_token": "old"}), "admin")
    store.lag_profile_reads()

    updated: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics", additional_auth=[{**static, "authentication_token": "new"}]
        ),
        "admin",
    )

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert updated.additional_auth is not None
    profile: Final = updated.additional_auth[0]
    assert decrypt_value_helper(profile.authentication_token or "", "auth_value") == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resent_settings", "kept"),
    [
        pytest.param({}, ("client_secret_basic", "https://auth.example/token", ["read"]), id="omitted-kept"),
        pytest.param(
            {"token_endpoint_auth_method": None, "token_url": "https://auth.example/token", "scopes": None},
            (None, "https://auth.example/token", []),
            id="sent-replaced",
        ),
    ],
)
async def test_put_keeps_the_stored_settings_a_profile_omits(
    resent_settings: dict[str, object], kept: tuple[object, ...]
) -> None:
    store: Final = FakeMcpPrisma()
    await _create(
        store,
        _new_server("analytics", _bot("ana-bots", token_endpoint_auth_method="client_secret_basic", scopes=["read"])),
        "admin",
    )
    resent: Final = {"id": "ana-bots", "auth_type": "oauth2", "oauth2_flow": "client_credentials", **resent_settings}

    await _update(store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[resent]), "admin")

    row: Final = store.profiles[0]
    assert (
        json.loads(str(row["credentials"])).get("token_endpoint_auth_method"),
        row["token_url"],
        row["scopes"],
    ) == kept
    assert _stored_secret(store, "ana-bots", "client_secret") == "ana-bots-secret"


@pytest.mark.asyncio
async def test_put_refuses_a_new_static_profile_without_a_token_and_keeps_a_stored_one() -> None:
    store: Final = FakeMcpPrisma()
    static: Final = {"id": "ana-key", "auth_type": "bearer_token"}
    await _create(store, _new_server("analytics", {**static, "authentication_token": "tok"}), "admin")

    kept: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[{**static, "id": "ANA-Key"}]),
        "admin",
    )
    refused: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics", additional_auth=[static, {"id": "new-key", "auth_type": "api_key"}]
        ),
        "admin",
    )

    assert isinstance(kept, LiteLLM_MCPServerTable)
    assert refused == MalformedProfile("analytics", 1, "profile 'new-key' needs authentication_token")
    assert _profile_ids(store) == {"analytics": ["ana-key"]}
    assert _stored_secret(store, "ana-key", "auth_value") == "tok"


@pytest.mark.asyncio
async def test_put_keeps_the_stored_client_credentials_a_same_kind_profile_omits() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    echoed: Final = {"id": "ana-bots", "auth_type": "oauth2", "oauth2_flow": "client_credentials"}

    kept: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[echoed]), "admin"
    )

    assert isinstance(kept, LiteLLM_MCPServerTable)
    assert (
        _stored_secret(store, "ana-bots", "client_id"),
        _stored_secret(store, "ana-bots", "client_secret"),
        store.profiles[0]["token_url"],
    ) == ("ana-bots-app", "ana-bots-secret", "https://auth.example/token")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sent", "missing"),
    [
        pytest.param(_bot("ana-bots", token_url=None), "token_url", id="stored-profile-clears-token-url"),
        pytest.param(_bot("ana-key", client_id=None), "client_id", id="kind-change-without-client-id"),
        pytest.param(_bot("ana-key", token_url=None), "token_url", id="kind-change-without-token-url"),
        pytest.param(_bot("new-bots", client_secret=None), "client_secret", id="new-profile-without-secret"),
    ],
)
async def test_put_refuses_a_client_credentials_profile_left_without_a_credential(
    sent: dict[str, object], missing: str
) -> None:
    store: Final = FakeMcpPrisma()
    await _create(
        store,
        _new_server(
            "analytics", _bot("ana-bots"), {"id": "ana-key", "auth_type": "api_key", "authentication_token": "k"}
        ),
        "admin",
    )
    rows_before: Final = [dict(row) for row in store.profiles]

    refused: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[sent]), "admin"
    )

    assert refused == MalformedProfile("analytics", 0, f"profile '{sent['id']}' needs {missing}")
    assert [dict(row) for row in store.profiles] == rows_before


@pytest.mark.asyncio
async def test_put_replaces_the_set_by_id() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)

    await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[_bot("B"), _bot("C")]),
        "admin",
    )

    assert _profile_ids(store) == {"analytics": ["B", "C"]}


@pytest.mark.asyncio
async def test_put_renaming_a_server_replaces_its_profiles_in_the_same_write() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)

    await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", alias="numbers", additional_auth=[_bot("C")]),
        "admin",
    )

    assert (store.servers[0]["alias"], _profile_ids(store)) == ("numbers", {"analytics": ["C"]})


@pytest.mark.asyncio
async def test_put_renaming_a_server_to_a_name_another_server_holds_keeps_its_profiles() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    await _create(store, _new_server("news"), "admin")
    store.identifier_checks = True

    conflict: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", alias="News", additional_auth=[_bot("C")]),
        "admin",
    )

    assert conflict == McpIdentifierConflict(field="alias", value="News", server_id="news")
    assert _profile_ids(store) == {"analytics": ["A", "B"], "news": []}


@pytest.mark.asyncio
async def test_a_put_whose_profile_write_conflict_is_gone_on_recheck_raises_the_database_error() -> None:
    from prisma.errors import UniqueViolationError

    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    store.vanishing_profile_conflicts = 1

    with pytest.raises(UniqueViolationError):
        await _update(
            store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[_bot("C")]), "admin"
        )

    assert _profile_ids(store) == {"analytics": ["A", "B"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("cleared", [[], None], ids=["empty-list", "null"])
async def test_put_of_an_empty_list_or_null_deletes_every_profile(cleared: list[object] | None) -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)

    updated: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=cleared), "admin"
    )

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert store.profiles == []
    assert updated.additional_auth is None


@pytest.mark.asyncio
async def test_put_changing_the_kind_needs_the_new_kinds_secret() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    static: Final = {"id": "A", "auth_type": "bearer_token"}

    refused: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[static, _bot("B")]), "admin"
    )
    accepted: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics", additional_auth=[{**static, "authentication_token": "tok"}, _bot("B")]
        ),
        "admin",
    )

    assert refused == MalformedProfile(
        "analytics", 0, "profile 'A' changes its auth settings, so authentication_token must be supplied"
    )
    assert isinstance(accepted, LiteLLM_MCPServerTable)
    assert _stored_secret(store, "A", "auth_value") == "tok"
    assert _stored_secret(store, "A", "client_secret") is None


@pytest.mark.asyncio
async def test_a_refused_put_leaves_the_stored_profiles_as_they_were() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    await _create(store, _new_server("github", _bot("gh-bots")), "admin")
    before: Final = [dict(row) for row in store.profiles]

    collision: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[_bot("A"), _bot("GH-BOTS")]),
        "admin",
    )
    interactive: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics", additional_auth=[_bot("A"), _bot("Z", oauth2_flow=None)]
        ),
        "admin",
    )

    assert collision == DuplicateProfileId("GH-BOTS", "analytics", "gh-bots", "github")
    assert interactive == UnsupportedOAuth2Flow("analytics", 1, "Z", None)
    assert store.profiles == before


@pytest.mark.asyncio
async def test_a_credentials_only_update_never_touches_the_profiles() -> None:
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    before: Final = [dict(row) for row in store.profiles]

    await _update(
        store,
        UpdateMCPServerWithProfilesRequest(
            server_id="analytics",
            credentials={"client_id": "dcr", "client_secret": "dcr-secret"},
            oauth2_flow="authorization_code",
        ),
        "mcp_oauth_dcr",
    )

    assert store.profiles == before


@pytest.mark.asyncio
async def test_two_concurrent_puts_adding_colliding_ids_leave_one_winner_and_one_conflict() -> None:
    store: Final = FakeMcpPrisma()
    for server_id in ("a", "b"):
        await _create(store, _new_server(server_id), "admin")
    store.hold_profile_lookups_until(2)

    first, second = await asyncio.gather(
        _update(store, UpdateMCPServerWithProfilesRequest(server_id="a", additional_auth=[_bot("Ana-Bots")]), "admin"),
        _update(store, UpdateMCPServerWithProfilesRequest(server_id="b", additional_auth=[_bot("ana-bots")]), "admin"),
    )

    results: Final = {type(first), type(second)}
    assert results == {LiteLLM_MCPServerTable, DuplicateProfileId}
    assert len(store.profiles) == 1


@pytest.mark.asyncio
async def test_a_put_that_loses_a_profile_race_to_a_write_the_reader_has_not_seen_is_a_conflict() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("a", _bot("Ana-Bots")), "admin")
    await _create(store, _new_server("b"), "admin")
    store.lag_profile_reads()

    refused: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="b", additional_auth=[_bot("ana-bots")]), "admin"
    )

    assert refused == DuplicateProfileId("ana-bots", "b", "Ana-Bots", "a")
    assert _profile_ids(store) == {"a": ["Ana-Bots"], "b": []}


@pytest.mark.asyncio
async def test_a_put_resending_a_profile_keeps_who_created_it_and_when() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "alice")
    created: Final = dict(store.profiles[0])

    await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[_bot("ana-bots")]), "bob"
    )

    row: Final = store.profiles[0]
    assert (row["created_by"], row["created_at"], row["updated_by"]) == ("alice", created["created_at"], "bob")


@pytest.mark.asyncio
async def test_a_put_omitting_a_secret_keeps_the_one_a_concurrent_put_just_rotated_in() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("analytics", _bot("ana-bots")), "admin")
    store.hold_profile_lookups_until(2)
    without_secrets: Final = {k: v for k, v in _bot("ana-bots", scopes=["write"]).items() if "client_" not in k}

    await asyncio.gather(
        _update(
            store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[without_secrets]), "admin"
        ),
        _update(
            store,
            UpdateMCPServerWithProfilesRequest(
                server_id="analytics", additional_auth=[_bot("ana-bots", client_secret="rotated")]
            ),
            "admin",
        ),
    )

    assert _stored_secret(store, "ana-bots", "client_secret") == "rotated"
    assert store.profiles[0]["scopes"] == ["write"]


@pytest.mark.asyncio
async def test_a_tokenless_resend_of_a_profile_a_concurrent_put_removed_is_refused() -> None:
    store: Final = FakeMcpPrisma()
    static: Final = {"id": "ana-key", "auth_type": "api_key", "authentication_token": "k1"}
    await _create(store, _new_server("analytics", static), "admin")
    store.hold_profile_lookups_until(2)
    tokenless: Final = {"id": "ana-key", "auth_type": "api_key"}
    replacement: Final = {"id": "ana-other", "auth_type": "api_key", "authentication_token": "k2"}

    resent, replaced = await asyncio.gather(
        _update(store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[tokenless]), "admin"),
        _update(
            store, UpdateMCPServerWithProfilesRequest(server_id="analytics", additional_auth=[replacement]), "admin"
        ),
    )

    assert resent == MalformedProfile("analytics", 0, "profile 'ana-key' needs authentication_token")
    assert isinstance(replaced, LiteLLM_MCPServerTable)
    assert _profile_ids(store) == {"analytics": ["ana-other"]}


@pytest.mark.asyncio
async def test_a_put_refuses_a_config_profile_id_even_when_its_server_id_is_the_config_server_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _yaml_server_holding(monkeypatch, "Yaml-Bots")
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("yaml_server"), "admin")

    refused: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="yaml_server", additional_auth=[_bot("yaml-bots")]), "admin"
    )

    assert refused == DuplicateProfileId("yaml-bots", "yaml_server", "Yaml-Bots", "yaml_server")
    assert store.profiles == []


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
    await _two_profile_server(store)
    before: Final = ([dict(row) for row in store.servers], [dict(row) for row in store.profiles])

    created: Final = await _create(store, _new_server("tools", _bot("T"), **host_fields), "admin")
    flipped: Final = await _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="analytics", **host_fields), "admin"
    )

    assert (created, flipped) == (UnsupportedProfileHost("tools", host), UnsupportedProfileHost("analytics", host))
    assert ([dict(row) for row in store.servers], store.profiles) == before


class _UnreachableProfileTable:
    def __getattr__(self, name: str) -> object:
        async def unreachable(**_: object) -> None:
            raise ConnectionError("profile table unreachable")

        return unreachable


_HOST_MOVES: Final = [
    pytest.param({"transport": "stdio", "command": "npx", "args": ["server"]}, id="stdio"),
    pytest.param({"is_byok": True}, id="byok"),
    pytest.param({"spec_path": "./openapi.json"}, id="openapi"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("host_fields", _HOST_MOVES)
async def test_a_host_edit_of_a_server_without_profiles_succeeds_when_the_profile_table_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, host_fields: dict[str, object]
) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("plain"), "admin")
    monkeypatch.setattr(store.db, "litellm_mcpauthprofiletable", _UnreachableProfileTable())

    updated: Final = await _update(store, UpdateMCPServerWithProfilesRequest(server_id="plain", **host_fields), "admin")

    assert isinstance(updated, LiteLLM_MCPServerTable)


@pytest.mark.asyncio
@pytest.mark.parametrize("host_fields", _HOST_MOVES)
async def test_a_host_edit_of_a_server_this_worker_loaded_with_profiles_fails_when_the_profile_table_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, host_fields: dict[str, object]
) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    loaded: Final = MCPServer(
        server_id="analytics",
        name="analytics",
        server_name="analytics",
        transport=MCPTransport.http,
        additional_auth=(MCPAuthProfile(id="A", auth_type="api_key", authentication_token="k"),),
    )
    monkeypatch.setitem(global_mcp_server_manager.registry, "analytics", loaded)
    before: Final = [dict(row) for row in store.servers]
    monkeypatch.setattr(store.db, "litellm_mcpauthprofiletable", _UnreachableProfileTable())

    with pytest.raises(ConnectionError):
        await _update(store, UpdateMCPServerWithProfilesRequest(server_id="analytics", **host_fields), "admin")
    assert [dict(row) for row in store.servers] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("host_move_first", [True, False], ids=["host-move-reads-first", "profiles-read-first"])
async def test_a_put_adding_profiles_and_a_put_making_the_server_byok_never_both_commit(host_move_first: bool) -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("plain"), "admin")
    store.hold_profile_lookups_until(2, columns=frozenset({"server_id", "profile_id_normalized"}))
    adding: Final = _update(
        store, UpdateMCPServerWithProfilesRequest(server_id="plain", additional_auth=[_bot("plain-bots")]), "admin"
    )
    moving: Final = _update(store, UpdateMCPServerWithProfilesRequest(server_id="plain", is_byok=True), "admin")

    first, second = await asyncio.gather(*((moving, adding) if host_move_first else (adding, moving)))

    refused: Final = [result for result in (first, second) if isinstance(result, UnsupportedProfileHost)]
    assert refused == [UnsupportedProfileHost("plain", "BYOK")], (first, second)
    assert not (store.servers[0]["is_byok"] and store.profiles), (store.servers, store.profiles)


@pytest.mark.asyncio
async def test_a_host_flip_on_a_server_without_profiles_is_written_as_on_main() -> None:
    store: Final = FakeMcpPrisma()
    await _create(store, _new_server("plain"), "admin")

    updated: Final = await _update(store, UpdateMCPServerWithProfilesRequest(server_id="plain", is_byok=True), "admin")

    assert isinstance(updated, LiteLLM_MCPServerTable)
    assert updated.is_byok is True


def test_routes_outside_post_put_and_register_drop_additional_auth_as_on_main() -> None:
    from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest

    profile: Final = [_bot("A")]

    assert (
        "additional_auth"
        not in NewMCPServerRequest(server_name="s", url="https://x", additional_auth=profile).model_dump()
    )
    assert "additional_auth" not in UpdateMCPServerRequest(server_id="s", additional_auth=profile).model_dump()
    assert "additional_auth" not in _new_server("s", _bot("A")).model_dump()


@pytest.mark.asyncio
async def test_a_put_turning_a_profiled_server_into_stdio_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    store: Final = FakeMcpPrisma()
    await _two_profile_server(store)
    before: Final = ([dict(row) for row in store.servers], [dict(row) for row in store.profiles])

    refused: Final = await _update(
        store,
        UpdateMCPServerWithProfilesRequest(server_id="analytics", transport="stdio", command="npx", args=["server"]),
        "admin",
    )

    assert refused == UnsupportedProfileHost("analytics", "stdio")
    assert ([dict(row) for row in store.servers], store.profiles) == before
