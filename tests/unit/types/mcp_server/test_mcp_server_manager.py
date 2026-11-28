import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.auth_profile import MCPAuthProfile
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_PROFILE: Final[TypeAdapter[MCPAuthProfile]] = TypeAdapter(MCPAuthProfile)
_PROFILE_FIELDS: Final = frozenset({"additional_auth", "auth_profiles_unreadable"})
_SECRETS: Final = ("static-profile-secret", "oauth-profile-id", "oauth-profile-secret")
_PROFILES: Final = (
    {"id": "build", "auth_type": "bearer_token", "authentication_token": _SECRETS[0]},
    {
        "id": "ops",
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
        "client_id": _SECRETS[1],
        "client_secret": _SECRETS[2],
        "token_url": "https://issuer.example/token",
        "scopes": ["read"],
    },
)
_RENDERINGS: Final[dict[str, Callable[[MCPServer], str]]] = {
    "repr": repr,
    "str": str,
    "repr of the profiles": lambda server: repr(server.additional_auth),
    "json": lambda server: server.model_dump_json(),
    "python dump": lambda server: str(server.model_dump()),
    "json dump": lambda server: json.dumps(server.model_dump(mode="json")),
}
_REJECTED_PROFILES: Final = {
    "not a list": {"id": "ok", "auth_type": "bearer_token", "authentication_token": "LEAKED-1"},
    "unknown field": [{"id": "ok", "auth_type": "bearer_token", "authentication_token": "LEAKED-1", "LEAKED-2": "x"}],
    "unknown auth type": [{"id": "ok", "auth_type": "LEAKED-1", "authentication_token": "LEAKED-2"}],
}


def _server(**fields: object) -> MCPServer:
    return MCPServer.model_validate(
        {
            "server_id": "tickets-1",
            "name": "tickets",
            "transport": MCPTransport.http,
            "url": "https://tickets.example.com/mcp",
            "auth_type": MCPAuth.bearer_token,
            "authentication_token": "example-token",
            **fields,
        }
    )


def _with_profiles() -> MCPServer:
    return _server(additional_auth=tuple(_PROFILE.validate_python(profile) for profile in _PROFILES))


@pytest.mark.parametrize(
    "fields", ({}, {"additional_auth": None}, {"additional_auth": ()}, {"auth_profiles_unreadable": False})
)
def test_a_server_without_profiles_dumps_none_of_the_profile_fields(fields: Mapping[str, object]) -> None:
    dumped: Final = _server(**fields).model_dump(mode="json")
    assert dumped.keys().isdisjoint(_PROFILE_FIELDS)
    assert dumped == _server().model_dump(mode="json")


@pytest.mark.parametrize("secret", _SECRETS)
@pytest.mark.parametrize("render", _RENDERINGS.values(), ids=_RENDERINGS.keys())
def test_no_rendering_of_a_server_with_profiles_contains_a_profile_secret(
    secret: str, render: Callable[[MCPServer], str]
) -> None:
    assert secret not in render(_with_profiles())


@pytest.mark.parametrize("profiles", _REJECTED_PROFILES.values(), ids=_REJECTED_PROFILES.keys())
def test_a_rejected_profile_list_never_echoes_what_it_was_given(profiles: object) -> None:
    with pytest.raises(ValidationError) as rejected:
        _server(additional_auth=profiles)
    assert "LEAKED" not in f"{rejected.value}{rejected.value.errors()}{rejected.value.json()}"


def test_a_dump_keeps_the_public_profile_settings() -> None:
    assert _with_profiles().model_dump(mode="json")["additional_auth"] == [
        {"id": "build", "auth_type": "bearer_token"},
        {
            "id": "ops",
            "auth_type": "oauth2",
            "oauth2_flow": "client_credentials",
            "token_url": "https://issuer.example/token",
            "scopes": ["read"],
        },
    ]


def test_a_dump_keeps_the_unreadable_marker() -> None:
    dumped: Final = _server(additional_auth=None, auth_profiles_unreadable=True).model_dump(mode="json")
    assert dumped["auth_profiles_unreadable"] is True
    assert "additional_auth" not in dumped


def test_a_server_cannot_hold_readable_profiles_and_report_them_unreadable() -> None:
    with pytest.raises(ValidationError, match="unreadable"):
        _server(additional_auth=_with_profiles().additional_auth, auth_profiles_unreadable=True)


def test_rejecting_profiles_beside_the_unreadable_marker_never_echoes_a_secret() -> None:
    with pytest.raises(ValidationError) as rejected:
        _server(additional_auth=_PROFILES, auth_profiles_unreadable=True)
    rendered: Final = f"{rejected.value}{rejected.value.errors()}{rejected.value.json()}"
    assert not any(secret in rendered for secret in (*_SECRETS, "example-token"))
