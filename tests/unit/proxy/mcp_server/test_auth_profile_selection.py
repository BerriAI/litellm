import asyncio
import logging
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.auth_profile_selection import (
    AmbiguousProfile,
    IncompleteProfile,
    MissingProfile,
    RejectedToken,
    UnreadableProfiles,
    UnsupportedProfileHost,
    UnusableProfile,
    raise_unusable_profile,
    resolve_effective_server,
    resolve_effective_servers,
    resolve_usable_server,
    select_auth_profile,
    shared_grant_memo,
    unusable_profile_detail,
    unusable_profile_status,
    with_auth_profile,
)
from litellm.proxy.mcp_server.auth_profiles import parse_auth_profiles
from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.auth_profile import ClientCredentialsAuthProfile, StaticAuthProfile
from litellm.types.mcp_server.auth_profile_grants import ProfileGrants, UnreadableGrants
from litellm.types.mcp_server.mcp_server_manager import MCPOAuthIdentityBinding, MCPServer

_BUILD: Final = {"id": "Build", "auth_type": "basic", "auth_value": "build-credential"}
_OPS: Final = {
    "id": "ops",
    "auth_type": "oauth2",
    "oauth2_flow": "client_credentials",
    "client_id": "ops-client",
    "client_secret": "ops-secret",
    "token_url": "https://issuer.example/token",
    "scopes": ["read", "write"],
    "token_endpoint_auth_method": "client_secret_basic",
}
_NO_CREDENTIAL: Final = {"id": "empty", "auth_type": "basic"}
_PROFILES: Final = (_BUILD, _OPS, _NO_CREDENTIAL)


@pytest.fixture(autouse=True)
def _tokens_are_obtainable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_mcp_server_manager, "profile_token_failure", AsyncMock(return_value=None))


@dataclass
class _Source:
    outcome: tuple[ProfileGrants, ...] | UnreadableGrants
    calls: list[UserAPIKeyAuth | None] = field(default_factory=list)  # mutable-ok: the test counts reads

    async def __call__(self, caller: UserAPIKeyAuth | None) -> tuple[ProfileGrants, ...] | UnreadableGrants:
        self.calls.append(caller)
        for _ in range(3):  # a read that takes several event-loop turns, so resolutions overlap
            await asyncio.sleep(0)
        return self.outcome


class _ForbiddenSource:
    async def __call__(self, caller: UserAPIKeyAuth | None) -> tuple[ProfileGrants, ...] | UnreadableGrants:
        raise AssertionError("the grants were read")


def _grants(by_server: Mapping[str, str] = MappingProxyType({})) -> ProfileGrants:
    return MappingProxyType(dict(by_server))


def _server(
    server_id: str = "tickets",
    profiles: Sequence[Mapping[str, object]] = _PROFILES,
    overrides: Mapping[str, object] = MappingProxyType({}),
) -> MCPServer:
    parsed: Final = parse_auth_profiles(server_id, tuple(profiles))
    assert isinstance(parsed, tuple)
    return MCPServer.model_validate(
        {
            "server_id": server_id,
            "name": server_id,
            "transport": MCPTransport.http,
            "url": "https://tickets.example.com/mcp",
            "auth_type": MCPAuth.bearer_token,
            "authentication_token": "default-credential",
            "additional_auth": parsed or None,
            **overrides,
        }
    )


def _selected_id(server: MCPServer, carriers: Sequence[ProfileGrants]) -> str | None:
    selected: Final = select_auth_profile(server, carriers)
    return selected.id if isinstance(selected, (StaticAuthProfile, ClientCredentialsAuthProfile)) else None


@pytest.mark.parametrize(
    "carriers",
    (
        (),
        (_grants(),),
        (_grants({"other": "Build"}),),
        (_grants({"tickets": "default"}),),
        (_grants({"tickets": "DeFaUlT"}),),
        (_grants({"tickets": "default"}), _grants({"tickets": "ops"})),
    ),
)
def test_no_grant_or_the_default_grant_selects_the_default_credential(carriers: Sequence[ProfileGrants]) -> None:
    assert select_auth_profile(_server(), carriers) is None


@pytest.mark.parametrize(
    ("granted", "expected"),
    (("Build", "Build"), ("build", "Build"), ("BUILD", "Build"), ("OPS", "ops"), ("ops", "ops")),
)
def test_a_named_grant_matches_the_profile_ignoring_case(granted: str, expected: str) -> None:
    assert _selected_id(_server(), (_grants({"tickets": granted}),)) == expected


@pytest.mark.parametrize(
    "granted",
    (
        pytest.param("missing", id="unknown"),
        pytest.param("Buil", id="prefix"),
        pytest.param("\u212aey", id="kelvin-sign"),
        pytest.param("bu\u0131ld", id="dotless-i"),
        pytest.param("", id="empty"),
    ),
)
def test_a_grant_to_an_unknown_profile_is_missing_never_the_default(granted: str) -> None:
    profiles: Final = ({"id": "key", "auth_type": "basic", "auth_value": "v"}, {"id": "build", "auth_type": "basic"})
    assert select_auth_profile(_server(profiles=profiles), (_grants({"tickets": granted}),)) == MissingProfile(
        "tickets", granted
    )


@pytest.mark.parametrize("granted", ("Build", "ops", ""))
def test_a_named_grant_on_a_server_with_no_profiles_is_missing_never_the_default(granted: str) -> None:
    server: Final = _server(profiles=())
    assert server.additional_auth is None
    assert select_auth_profile(server, (_grants({"tickets": granted}),)) == MissingProfile("tickets", granted)
    assert select_auth_profile(server, (_grants({"tickets": "default"}),)) is None


_INCOMPLETE_CLIENT_CREDENTIALS: Final = {
    "id": "half",
    "auth_type": "oauth2",
    "oauth2_flow": "client_credentials",
    "client_id": "half-client",
    "token_url": "https://issuer.example/token",
}


@pytest.mark.parametrize(
    "granted", ("EMPTY", "empty", "HALF", "half"), ids=("static-upper", "static", "oauth-upper", "oauth")
)
def test_a_selected_profile_without_its_secret_is_incomplete_never_the_default(granted: str) -> None:
    server: Final = _server(profiles=(*_PROFILES, _INCOMPLETE_CLIENT_CREDENTIALS))
    assert select_auth_profile(server, (_grants({"tickets": granted}),)) == IncompleteProfile("tickets", granted)


@pytest.mark.parametrize(
    ("carriers", "expected"),
    (
        ((_grants({"tickets": "Build"}), _grants({"tickets": "ops"}), _grants({"tickets": "empty"})), "Build"),
        ((_grants({"other": "ops"}), _grants({"tickets": "ops"}), _grants({"tickets": "Build"})), "ops"),
        ((_grants(), _grants(), _grants({"tickets": "Build"})), "Build"),
        ((_grants({"tickets": "default"}), _grants({"tickets": "ops"})), None),
    ),
)
def test_the_first_carrier_naming_the_server_wins(carriers: Sequence[ProfileGrants], expected: str | None) -> None:
    assert _selected_id(_server(), carriers) == expected


def test_unreadable_stored_profiles_make_a_named_grant_unusable_and_leave_other_callers_alone() -> None:
    server: Final = _server(profiles=(), overrides={"auth_profiles_unreadable": True})
    assert select_auth_profile(server, (_grants({"tickets": "Build"}),)) == UnreadableProfiles("tickets", "Build")
    assert select_auth_profile(server, (_grants({"tickets": "default"}),)) is None
    assert select_auth_profile(server, ()) is None


_UNSUPPORTED_HOSTS: Final = {
    "stdio": {"transport": MCPTransport.stdio, "url": None, "command": "run-tickets"},
    "bring your own key": {"is_byok": True},
    "openapi": {"spec_path": "https://tickets.example.com/openapi.json"},
}


@pytest.mark.parametrize("host", _UNSUPPORTED_HOSTS.values(), ids=_UNSUPPORTED_HOSTS.keys())
def test_a_named_grant_on_a_host_that_cannot_apply_profiles_is_unusable_never_the_default(
    host: Mapping[str, object],
) -> None:
    server: Final = _server().model_copy(update=host)
    assert select_auth_profile(server, (_grants({"tickets": "Build"}),)) == UnsupportedProfileHost("tickets", "Build")
    assert select_auth_profile(server, (_grants({"tickets": "missing"}),)) == UnsupportedProfileHost(
        "tickets", "missing"
    )
    unreadable: Final = server.model_copy(update={"auth_profiles_unreadable": True})
    assert select_auth_profile(unreadable, (_grants({"tickets": "Build"}),)) == UnsupportedProfileHost(
        "tickets", "Build"
    )


@pytest.mark.parametrize("host", _UNSUPPORTED_HOSTS.values(), ids=_UNSUPPORTED_HOSTS.keys())
@pytest.mark.parametrize("carriers", ((), (_grants({"tickets": "default"}),), (_grants({"tickets": "DEFAULT"}),)))
def test_no_grant_or_the_default_grant_on_such_a_host_selects_the_default_credential(
    host: Mapping[str, object], carriers: Sequence[ProfileGrants]
) -> None:
    assert select_auth_profile(_server().model_copy(update=host), carriers) is None


def test_two_profiles_whose_ids_match_the_grant_ignoring_case_are_ambiguous_never_first_wins() -> None:
    first: Final = StaticAuthProfile.model_validate({"id": "Ops", "auth_type": "basic", "auth_value": "one"})
    second: Final = StaticAuthProfile.model_validate({"id": "ops", "auth_type": "basic", "auth_value": "two"})
    server: Final = _server(profiles=(), overrides={"additional_auth": (first, second)})
    assert select_auth_profile(server, (_grants({"tickets": "OPS"}),)) == AmbiguousProfile("tickets", "OPS")
    assert select_auth_profile(server, (_grants({"tickets": "default"}),)) is None


_DEFAULT_TEXT_FIELDS: Final = (
    "client_id",
    "client_secret",
    "token_url",
    "configured_token_url",
    "authorization_url",
    "configured_authorization_url",
    "registration_url",
    "configured_registration_url",
    "audience",
    "upstream_resource",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_service_name",
    "aws_role_name",
    "aws_session_name",
    "client_private_key",
    "client_private_key_id",
    "token_exchange_endpoint",
    "id_jag_resource_token_endpoint",
    "id_jag_resource_client_id",
    "id_jag_resource_client_secret",
)
_DEFAULT_CREDENTIALS: Final = MappingProxyType(
    {
        **{name: f"default-{name}" for name in _DEFAULT_TEXT_FIELDS},
        "oauth2_flow": "client_credentials",
        "scopes": ["default-scope"],
        "token_endpoint_auth_method": "client_secret_post",
        "delegate_auth_to_upstream": True,
        "static_headers": {"Authorization": "Bearer default-static-header"},
        "extra_headers": ["Authorization"],
        "env_vars": [{"name": "TOKEN", "value": "default-env-var", "scope": "global"}],
    }
)
_CLEARED_BY_EVERY_PROFILE: Final = (
    *(name for name in _DEFAULT_TEXT_FIELDS if name not in ("client_id", "client_secret", "token_url")),
    "static_headers",
    "extra_headers",
    "env_vars",
)


@pytest.mark.parametrize("granted", ("Build", "ops"), ids=("static", "client-credentials"))
def test_a_selected_profile_leaves_no_default_credential_on_the_applied_server(granted: str) -> None:
    server: Final = _server(overrides=_DEFAULT_CREDENTIALS)
    selected: Final = select_auth_profile(server, (_grants({"tickets": granted}),))
    assert isinstance(selected, (StaticAuthProfile, ClientCredentialsAuthProfile))
    applied: Final = with_auth_profile(server, selected)
    assert {name: getattr(applied, name) for name in _CLEARED_BY_EVERY_PROFILE} == dict.fromkeys(
        _CLEARED_BY_EVERY_PROFILE
    )
    assert applied.delegate_auth_to_upstream is False
    assert applied.auth_profile_id == selected.id
    assert applied.uses_auth_profile and not server.uses_auth_profile
    assert "default-" not in str(applied.model_dump())
    assert applied.model_fields_set <= set(MCPServer.model_fields)


def test_a_static_profile_replaces_the_default_credential_and_clears_the_oauth_client() -> None:
    server: Final = _server(overrides=_DEFAULT_CREDENTIALS)
    selected: Final = select_auth_profile(server, (_grants({"tickets": "Build"}),))
    assert isinstance(selected, StaticAuthProfile)
    applied: Final = with_auth_profile(server, selected)
    assert (applied.auth_type, applied.authentication_token) == (MCPAuth.basic, "build-credential")
    oauth_fields: Final = (
        "oauth2_flow",
        "client_id",
        "client_secret",
        "token_url",
        "scopes",
        "token_endpoint_auth_method",
    )
    assert {name: getattr(applied, name) for name in oauth_fields} == dict.fromkeys(oauth_fields)


def test_a_client_credentials_profile_supplies_the_oauth_fields_and_drops_the_default_token() -> None:
    server: Final = _server(overrides={"delegate_auth_to_upstream": True, "audience": "default-audience"})
    selected: Final = select_auth_profile(server, (_grants({"tickets": "OPS"}),))
    assert isinstance(selected, ClientCredentialsAuthProfile)
    applied: Final = with_auth_profile(server, selected)
    assert (
        applied.auth_type,
        applied.oauth2_flow,
        applied.client_id,
        applied.client_secret,
        applied.effective_token_url,
        applied.scopes,
        applied.token_endpoint_auth_method,
        applied.authentication_token,
        applied.audience,
        applied.auth_profile_id,
    ) == (
        MCPAuth.oauth2,
        "client_credentials",
        "ops-client",
        "ops-secret",
        "https://issuer.example/token",
        ["read", "write"],
        "client_secret_basic",
        None,
        None,
        "ops",
    )
    assert applied.has_client_credentials


@pytest.mark.parametrize("granted", ("Build", "ops"), ids=("static", "client-credentials"))
def test_an_applied_server_still_satisfies_the_model_validators(granted: str) -> None:
    binding: Final = MCPOAuthIdentityBinding(
        mode="enforce", issuer="https://issuer.example", audiences=["gateway"], principal_claim="email"
    )
    server: Final = _server(overrides={"auth_type": MCPAuth.oauth2, "oauth_identity_binding": binding})
    selected: Final = select_auth_profile(server, (_grants({"tickets": granted}),))
    assert isinstance(selected, (StaticAuthProfile, ClientCredentialsAuthProfile))
    applied: Final = with_auth_profile(server, selected)
    assert applied.oauth_identity_binding is None
    MCPServer.model_validate(applied.model_dump())


def test_applying_no_profile_returns_the_identical_server() -> None:
    server: Final = _server()
    assert with_auth_profile(server, None) is server


_REFUSAL_CASES: Final = (
    MissingProfile("tickets-server-id", "granted-profile-id"),
    IncompleteProfile("tickets-server-id", "granted-profile-id"),
    UnreadableProfiles("tickets-server-id", "granted-profile-id"),
    UnsupportedProfileHost("tickets-server-id", "granted-profile-id"),
    AmbiguousProfile("tickets-server-id", "granted-profile-id"),
    RejectedToken("tickets-server-id", "granted-profile-id"),
)


@pytest.mark.parametrize(
    ("unusable", "expected_status"),
    (
        (MissingProfile("tickets-server-id", "granted-profile-id"), 500),
        (IncompleteProfile("tickets-server-id", "granted-profile-id"), 500),
        (UnsupportedProfileHost("tickets-server-id", "granted-profile-id"), 500),
        (AmbiguousProfile("tickets-server-id", "granted-profile-id"), 500),
        (RejectedToken("tickets-server-id", "granted-profile-id"), 500),
        (RejectedToken("tickets-server-id", "granted-profile-id", is_outage=True), 503),
        (UnreadableProfiles("tickets-server-id", "granted-profile-id"), 503),
        (UnreadableGrants(), 503),
    ),
    ids=lambda case: case if isinstance(case, int) else f"{type(case).__name__}{'_outage' if getattr(case, 'is_outage', False) else ''}",
)
def test_every_unusable_profile_raises_its_status_and_detail(
    unusable: UnusableProfile, expected_status: int
) -> None:
    with pytest.raises(HTTPException) as raised:
        raise_unusable_profile(unusable)
    assert raised.value.status_code == expected_status
    assert raised.value.detail == unusable_profile_detail(unusable)


def _logged_refusal(unusable: UnusableProfile, caplog: pytest.LogCaptureFixture) -> str:
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"), pytest.raises(HTTPException):
        raise_unusable_profile(unusable)
    return " ".join(record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)


@pytest.mark.parametrize("unusable", _REFUSAL_CASES, ids=lambda case: type(case).__name__)
def test_the_refusal_log_names_the_server_and_profile_while_the_503_does_not(
    unusable: UnusableProfile, caplog: pytest.LogCaptureFixture
) -> None:
    logged: Final = _logged_refusal(unusable, caplog)
    assert "tickets-server-id" in logged and "granted-profile-id" in logged


def test_the_log_for_unreadable_grants_says_why(caplog: pytest.LogCaptureFixture) -> None:
    assert "grants could not be read" in _logged_refusal(UnreadableGrants(), caplog)


@pytest.mark.asyncio
async def test_resolving_a_server_applies_the_granted_profile() -> None:
    resolved: Final = await resolve_effective_server(_server(), None, _Source((_grants({"tickets": "Build"}),)))
    assert isinstance(resolved, MCPServer) and resolved.auth_profile_id == "Build"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tag", "is_outage"),
    (("misconfigured", False), ("upstream_unavailable", True)),
)
async def test_a_client_credentials_profile_whose_token_cannot_be_obtained_is_unusable(
    monkeypatch: pytest.MonkeyPatch, tag: str, is_outage: bool
) -> None:
    monkeypatch.setattr(global_mcp_server_manager, "profile_token_failure", AsyncMock(return_value=tag))
    resolved: Final = await resolve_effective_server(_server(), None, _Source((_grants({"tickets": "ops"}),)))
    assert resolved == RejectedToken("tickets", "ops", is_outage=is_outage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("granted", "refusal"),
    (("missing", MissingProfile("tickets", "missing")), ("empty", IncompleteProfile("tickets", "empty"))),
)
async def test_resolving_a_server_reports_why_the_granted_profile_is_unusable(
    granted: str, refusal: MissingProfile | IncompleteProfile
) -> None:
    resolved: Final = await resolve_effective_server(_server(), None, _Source((_grants({"tickets": granted}),)))
    assert resolved == refusal


@pytest.mark.asyncio
async def test_unreadable_grants_fail_only_the_servers_that_have_profiles(caplog: pytest.LogCaptureFixture) -> None:
    source: Final = _Source(UnreadableGrants())
    plain: Final = _server("plain", profiles=())
    profiled: Final = _server("profiled")
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        resolutions: Final = await resolve_effective_servers((plain, profiled), None, source)
    assert resolutions.usable == (plain,) and resolutions.usable[0] is plain
    assert resolutions.left_out == ((profiled, UnreadableGrants()),)
    warnings: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "profiled" in warnings[0] and "plain" not in warnings[0]


@pytest.mark.asyncio
async def test_every_left_out_server_is_named_in_one_warning_and_the_outcomes_keep_the_input_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    servers: Final = (_server("a"), _server("plain", profiles=()), _server("b"), _server("c"))
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        resolutions: Final = await resolve_effective_servers(servers, None, _Source(UnreadableGrants()))
    assert [resolution.server for resolution in resolutions.outcomes] == list(servers)
    warnings: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and all(f"{name}:" in warnings[0] for name in ("a", "b", "c"))


@pytest.mark.asyncio
async def test_servers_without_profiles_never_cause_a_grant_read() -> None:
    plain: Final = (_server("a", profiles=()), _server("b", profiles=()))
    assert (await resolve_effective_server(plain[0], None, _ForbiddenSource())) is plain[0]
    resolutions: Final = await resolve_effective_servers(plain, None, _ForbiddenSource())
    assert resolutions.usable == plain and resolutions.left_out == ()
    assert all(resolved is original for resolved, original in zip(resolutions.usable, plain, strict=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("no_profiles", (None, ()), ids=("none", "empty"))
async def test_a_server_without_profiles_is_returned_unchanged_whatever_grants_the_caller_holds(
    no_profiles: tuple[()] | None,
) -> None:
    server: Final = _server(profiles=(), overrides={"additional_auth": no_profiles})
    source: Final = _Source((_grants({"tickets": "Build"}),))
    assert await resolve_effective_server(server, UserAPIKeyAuth(user_id="first"), source) is server
    assert source.calls == []


@pytest.mark.asyncio
async def test_the_default_grant_source_leaves_a_caller_without_grants_on_the_default_credential() -> None:
    server: Final = _server()
    assert await resolve_effective_server(server, None) is server


@pytest.mark.asyncio
async def test_a_server_that_already_has_a_profile_applied_is_not_resolved_again() -> None:
    server: Final = _server()
    selection: Final = select_auth_profile(server, (_grants({"tickets": "Build"}),))
    assert isinstance(selection, StaticAuthProfile)
    applied: Final = with_auth_profile(server, selection)
    assert await resolve_effective_server(applied, None, _ForbiddenSource()) is applied


@pytest.mark.asyncio
async def test_a_batch_of_servers_reads_the_grants_once() -> None:
    source: Final = _Source((_grants({"a": "Build", "b": "ops", "c": "default"}),))
    servers: Final = tuple(_server(name) for name in ("a", "b", "c", "d"))
    resolutions: Final = await resolve_effective_servers(servers, None, source)
    assert len(source.calls) == 1
    assert [server.auth_profile_id for server in resolutions.usable] == ["Build", "ops", None, None]


@pytest.mark.asyncio
async def test_the_memo_serves_one_scope_and_is_dropped_at_its_end() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    other: Final = UserAPIKeyAuth(user_id="second")
    source: Final = _Source((_grants({"a": "Build"}),))
    with shared_grant_memo():
        await resolve_effective_server(_server("a"), caller, source)
        await resolve_effective_servers((_server("a"), _server("b")), caller, source)
        assert source.calls == [caller]
        await resolve_effective_server(_server("a"), other, source)
        await resolve_effective_server(_server("a"), caller, source)
        assert source.calls == [caller, other]
    with shared_grant_memo():
        await resolve_effective_server(_server("a"), caller, source)
    assert source.calls == [caller, other, caller]


@pytest.mark.asyncio
async def test_one_scope_reads_the_grants_of_each_source_separately() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    granting: Final = _Source((_grants({"a": "Build"}),))
    silent: Final = _Source(())
    with shared_grant_memo():
        built: Final = await resolve_effective_server(_server("a"), caller, granting)
        default: Final = await resolve_effective_server(_server("a"), caller, silent)
    assert isinstance(built, MCPServer) and built.auth_profile_id == "Build"
    assert isinstance(default, MCPServer) and default.auth_profile_id is None
    assert len(granting.calls) == len(silent.calls) == 1


@pytest.mark.asyncio
async def test_a_read_that_fails_is_raised_to_every_resolution_in_its_scope_and_not_retried() -> None:
    calls: Final[list[UserAPIKeyAuth | None]] = []  # mutable-ok: the test counts reads

    async def failing(caller: UserAPIKeyAuth | None) -> tuple[ProfileGrants, ...] | UnreadableGrants:
        calls.append(caller)
        raise RuntimeError("the source broke")

    with shared_grant_memo():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="the source broke"):
                await resolve_effective_server(_server("a"), None, failing)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_without_a_request_memo_each_resolution_reads_the_grants() -> None:
    source: Final = _Source(())
    await resolve_effective_server(_server("a"), None, source)
    await resolve_effective_server(_server("a"), None, source)
    assert len(source.calls) == 2


@pytest.mark.asyncio
async def test_stored_profiles_that_could_not_be_read_refuse_a_named_grant() -> None:
    server: Final = _server(profiles=(), overrides={"auth_profiles_unreadable": True})
    source: Final = _Source((_grants({"tickets": "Build"}),))
    assert await resolve_effective_server(server, None, source) == UnreadableProfiles("tickets", "Build")
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_stored_profiles_that_could_not_be_read_keep_the_default_credential_for_the_default_grant() -> None:
    server: Final = _server(profiles=(), overrides={"auth_profiles_unreadable": True})
    source: Final = _Source((_grants({"tickets": "default"}),))
    assert await resolve_effective_server(server, None, source) is server
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_resolving_a_named_grant_on_a_host_that_cannot_apply_profiles_is_unusable() -> None:
    server: Final = _server(overrides={"is_byok": True})
    resolved: Final = await resolve_effective_server(server, None, _Source((_grants({"tickets": "Build"}),)))
    assert resolved == UnsupportedProfileHost("tickets", "Build")


@pytest.mark.asyncio
async def test_resolving_a_grant_that_matches_two_profiles_is_unusable() -> None:
    twins: Final = tuple(
        StaticAuthProfile.model_validate({"id": name, "auth_type": "basic", "auth_value": name})
        for name in ("Ops", "ops")
    )
    resolved: Final = await resolve_effective_server(
        _server(profiles=(), overrides={"additional_auth": twins}), None, _Source((_grants({"tickets": "ops"}),))
    )
    assert resolved == AmbiguousProfile("tickets", "ops")


@pytest.mark.asyncio
async def test_concurrent_resolutions_in_one_request_scope_share_one_grant_read() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build", "b": "ops"}),))
    with shared_grant_memo():
        resolved: Final = await asyncio.gather(
            resolve_effective_server(_server("a"), caller, source),
            resolve_effective_server(_server("b"), caller, source),
        )
    assert len(source.calls) == 1
    assert [server.auth_profile_id for server in resolved if isinstance(server, MCPServer)] == ["Build", "ops"]


@pytest.mark.asyncio
async def test_concurrent_batches_outside_a_request_scope_each_read_the_grants_once() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build", "b": "ops", "c": "Build"}),))
    batch: Final = tuple(_server(name) for name in ("a", "b", "c"))
    first, second = await asyncio.gather(
        resolve_effective_servers(batch, caller, source), resolve_effective_servers(batch, caller, source)
    )
    assert len(source.calls) == 2
    assert first.usable == second.usable and len(first.usable) == 3


@pytest.mark.asyncio
async def test_a_cancelled_reader_does_not_cancel_the_read_another_resolution_shares() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build"}),))
    with shared_grant_memo():
        impatient: Final = asyncio.create_task(resolve_effective_server(_server("a"), caller, source))
        await asyncio.sleep(0)
        patient: Final = asyncio.create_task(resolve_effective_server(_server("a"), caller, source))
        await asyncio.sleep(0)
        impatient.cancel()
        resolved: Final = await patient
    assert isinstance(resolved, MCPServer) and resolved.auth_profile_id == "Build"
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_a_cancelled_joiner_does_not_cancel_the_read_it_joined() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build"}),))
    with shared_grant_memo():
        patient: Final = asyncio.create_task(resolve_effective_server(_server("a"), caller, source))
        await asyncio.sleep(0)
        impatient: Final = asyncio.create_task(resolve_effective_server(_server("a"), caller, source))
        await asyncio.sleep(0)
        impatient.cancel()
        resolved: Final = await patient
    assert isinstance(resolved, MCPServer) and resolved.auth_profile_id == "Build"
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_a_child_task_that_runs_after_its_scope_ended_reads_the_current_grants() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build"}),))
    scope_ended: Final = asyncio.Event()

    async def after_the_scope() -> tuple[MCPServer | UnusableProfile, ...]:
        await scope_ended.wait()
        first: Final = await resolve_effective_server(_server("a"), caller, source)
        second: Final = await resolve_effective_server(_server("a"), caller, source)
        batched: Final = await resolve_effective_servers((_server("a"), _server("b")), caller, source)
        return (first, second, batched.outcomes[0].outcome)

    with shared_grant_memo():
        within: Final = await resolve_effective_server(_server("a"), caller, source)
        child: Final = asyncio.create_task(after_the_scope())
    assert isinstance(within, MCPServer) and within.auth_profile_id == "Build"
    source.outcome = ()
    scope_ended.set()
    resolved: Final = await child
    assert len(source.calls) == 4
    assert [r.auth_profile_id for r in resolved if isinstance(r, MCPServer)] == [None, None, None]


@pytest.mark.asyncio
async def test_a_scope_that_ends_in_another_task_is_closed_without_raising() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build"}),))
    scope: Final = ExitStack()
    scope.enter_context(shared_grant_memo())
    await resolve_effective_server(_server("a"), caller, source)
    await resolve_effective_server(_server("a"), caller, source)
    assert len(source.calls) == 1

    async def end_scope() -> None:
        scope.close()

    await asyncio.create_task(end_scope())
    await resolve_effective_server(_server("a"), caller, source)
    await resolve_effective_server(_server("a"), caller, source)
    assert len(source.calls) == 3


@pytest.mark.asyncio
async def test_a_scope_that_ends_in_another_task_leaves_that_tasks_own_scope_alone() -> None:
    caller: Final = UserAPIKeyAuth(user_id="first")
    source: Final = _Source((_grants({"a": "Build"}),))
    scope: Final = ExitStack()
    own_scope_read: Final = asyncio.Event()
    foreign_scope_open: Final = asyncio.Event()

    async def other_task() -> None:
        with shared_grant_memo():
            await resolve_effective_server(_server("a"), caller, source)
            own_scope_read.set()
            await foreign_scope_open.wait()
            scope.close()
            await resolve_effective_server(_server("a"), caller, source)

    task: Final = asyncio.create_task(other_task())
    await own_scope_read.wait()
    scope.enter_context(shared_grant_memo())
    foreign_scope_open.set()
    await task
    assert len(source.calls) == 1


@pytest.mark.asyncio
async def test_equal_but_distinct_callers_in_one_scope_read_separately() -> None:
    first: Final = UserAPIKeyAuth(user_id="same")
    copy: Final = UserAPIKeyAuth(user_id="same")
    assert first == copy and first is not copy
    source: Final = _Source((_grants({"a": "Build"}),))
    with shared_grant_memo():
        await resolve_effective_server(_server("a"), first, source)
        await resolve_effective_server(_server("a"), copy, source)
    assert source.calls == [first, copy]


@pytest.mark.asyncio
async def test_a_usable_server_resolves_to_the_server_the_grant_selects() -> None:
    resolved: Final = await resolve_usable_server(
        _server(), UserAPIKeyAuth(), _Source((_grants({"tickets": "Build"}),))
    )

    assert (resolved.auth_profile_id, resolved.authentication_token) == ("Build", "build-credential")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (_Source((_grants({"tickets": "missing"}),)), MissingProfile("tickets", "missing")),
        (_Source((_grants({"tickets": "empty"}),)), IncompleteProfile("tickets", "empty")),
        (_Source(UnreadableGrants()), UnreadableGrants()),
    ),
    ids=("missing-profile", "profile-without-a-credential", "unreadable-grants"),
)
async def test_an_unusable_server_raises_its_status_and_detail(source: _Source, expected: UnusableProfile) -> None:
    with pytest.raises(HTTPException) as unavailable:
        await resolve_usable_server(_server(), UserAPIKeyAuth(), source)

    assert (unavailable.value.status_code, unavailable.value.detail) == (
        unusable_profile_status(expected),
        unusable_profile_detail(expected),
    )
