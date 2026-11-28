import logging
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy.mcp_server.auth_profile_selection import (
    PROFILE_UNAVAILABLE,
    AmbiguousProfile,
    IncompleteProfile,
    MissingProfile,
    UnreadableProfiles,
    UnsupportedProfileHost,
    UnusableProfile,
    raise_unusable_profile,
    select_auth_profile,
    with_auth_profile,
)
from litellm.proxy.mcp_server.auth_profiles import parse_auth_profiles
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
)


@pytest.mark.parametrize("unusable", (*_REFUSAL_CASES, UnreadableGrants()), ids=lambda case: type(case).__name__)
def test_every_unusable_profile_becomes_one_503_that_names_no_server_or_profile(unusable: UnusableProfile) -> None:
    with pytest.raises(HTTPException) as raised:
        raise_unusable_profile(unusable)
    assert raised.value.status_code == 503
    assert raised.value.detail == PROFILE_UNAVAILABLE == "The MCP server's selected auth profile is unavailable."
    assert "tickets-server-id" not in str(raised.value.detail) + str(raised.value.headers)


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
