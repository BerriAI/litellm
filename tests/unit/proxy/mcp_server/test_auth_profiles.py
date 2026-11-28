from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy.mcp_server.auth_profiles import (
    InvalidAuthProfiles,
    parse_auth_profiles,
    profile_credentials_error,
)


def test_profile_ids_keep_their_declared_spelling() -> None:
    profile: Final = {"id": "Build", "auth_type": "basic", "auth_value": "credential"}
    profiles: Final = parse_auth_profiles("tickets", [profile, {**profile, "id": "read"}])
    assert isinstance(profiles, tuple)
    assert [p.id for p in profiles] == ["Build", "read"]


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        (
            ("Build", "BUILD"),
            "additional_auth[1] of MCP server 'tickets' repeats profile id 'BUILD' "
            "(first 'Build' at additional_auth[0])",
        ),
        (
            ("A", "B", "b", "a"),
            "additional_auth[2] of MCP server 'tickets' repeats profile id 'b' (first 'B' at additional_auth[1])",
        ),
    ],
)
def test_profile_ids_are_unique_ignoring_case_and_the_first_collision_is_reported(
    ids: tuple[str, ...], expected: str
) -> None:
    entries: Final = [{"id": profile_id, "auth_type": "basic", "auth_value": "credential"} for profile_id in ids]
    assert parse_auth_profiles("tickets", entries) == InvalidAuthProfiles(expected)


@pytest.mark.parametrize("raw", (None, False, {}, "[]"))
def test_a_supplied_non_list_is_invalid(raw: object) -> None:
    assert parse_auth_profiles("tickets", raw) == InvalidAuthProfiles(
        "additional_auth of MCP server 'tickets' must be a list"
    )


def test_an_empty_list_has_no_profiles() -> None:
    assert parse_auth_profiles("tickets", []) == ()


def test_the_profile_count_is_limited() -> None:
    entries: Final = [{"id": f"p{i}", "auth_type": "basic", "auth_value": "credential"} for i in range(257)]
    within: Final = parse_auth_profiles("tickets", entries[:256])
    assert isinstance(within, tuple), within
    assert len(within) == 256
    assert parse_auth_profiles("tickets", entries) == InvalidAuthProfiles(
        "additional_auth of MCP server 'tickets' has more than 256 profiles"
    )


def test_a_tuple_of_profiles_parses_like_a_list() -> None:
    profile: Final = {"id": "Read", "auth_type": "basic", "auth_value": "example-token"}
    assert parse_auth_profiles("tickets", (profile,)) == parse_auth_profiles("tickets", [profile])


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (
            {"id": "default", "auth_type": "basic", "auth_value": "secret-value"},
            "additional_auth[1].id of MCP server 'tickets' is invalid: "
            "Value error, the default credential has no profile id",
        ),
        (
            {
                "id": "reader",
                "auth_type": "oauth2",
                "oauth2_flow": "authorization_code",
                "client_secret": "secret-value",
            },
            "additional_auth[1].oauth2_flow of MCP server 'tickets' is invalid: Input should be 'client_credentials'",
        ),
        (
            {"id": "reader", "auth_type": "secret-value"},
            "additional_auth[1] of MCP server 'tickets' is invalid: Unknown auth type",
        ),
        (
            {"id": "reader", "auth_type": "basic", "secret-value": "x"},
            "additional_auth[1] of MCP server 'tickets' is invalid: Extra inputs are not permitted",
        ),
        (
            {"auth_type": "basic", "auth_value": "secret-value"},
            "additional_auth[1].id of MCP server 'tickets' is invalid: Field required",
        ),
        (
            {"id": "reader", "auth_type": "oauth2", "client_id": "secret-value"},
            "additional_auth[1].oauth2_flow of MCP server 'tickets' is invalid: Field required",
        ),
        (
            {"id": "reader", "auth_type": "oauth2", "oauth2_flow": "client_credentials", "token_url": 5},
            "additional_auth[1].token_url of MCP server 'tickets' is invalid: Input should be a valid string",
        ),
    ],
)
def test_invalid_entry_reports_its_server_position_and_field_without_echoing_input(
    entry: Mapping[str, str | int], expected: str
) -> None:
    assert parse_auth_profiles("tickets", [{"id": "good", "auth_type": "basic"}, entry]) == InvalidAuthProfiles(
        expected
    )


def test_the_first_invalid_entry_is_reported() -> None:
    bad: Final = {"id": "default", "auth_type": "basic"}
    assert parse_auth_profiles("tickets", [bad, {"id": "ok", "auth_type": "basic"}, {"auth_type": "basic"}]) == (
        InvalidAuthProfiles(
            "additional_auth[0].id of MCP server 'tickets' is invalid: "
            "Value error, the default credential has no profile id"
        )
    )


def test_complete_static_and_client_credentials_profiles_can_be_used() -> None:
    profiles: Final = parse_auth_profiles(
        "tickets",
        [
            {"id": "a", "auth_type": "basic", "auth_value": "value"},
            {
                "id": "b",
                "auth_type": "oauth2",
                "oauth2_flow": "client_credentials",
                "client_id": "id",
                "client_secret": "secret",
                "token_url": "https://issuer.example/token",
            },
        ],
    )
    assert isinstance(profiles, tuple)
    assert [p.id for p in profiles] == ["a", "b"]
    assert profile_credentials_error("tickets", profiles) is None


@pytest.mark.parametrize("credential", [{}, {"authentication_token": ""}])
def test_a_static_profile_without_a_token_parses_but_is_incomplete(
    credential: Mapping[str, str],
) -> None:
    profiles: Final = parse_auth_profiles(
        "tickets",
        [
            {"id": "first", "auth_type": "basic", "auth_value": "value"},
            {"id": "build", "auth_type": "basic", **credential},
        ],
    )
    assert isinstance(profiles, tuple)
    assert profile_credentials_error("tickets", profiles) == InvalidAuthProfiles(
        "Profile 'build' of MCP server 'tickets' requires authentication_token"
    )


def test_the_first_incomplete_profile_is_the_one_named() -> None:
    profiles: Final = parse_auth_profiles(
        "tickets", [{"id": "a", "auth_type": "basic"}, {"id": "b", "auth_type": "basic"}]
    )
    assert isinstance(profiles, tuple)
    assert profile_credentials_error("tickets", profiles) == InvalidAuthProfiles(
        "Profile 'a' of MCP server 'tickets' requires authentication_token"
    )


@pytest.mark.parametrize("field", ["client_id", "client_secret", "token_url"])
@pytest.mark.parametrize("supply_empty", [False, True], ids=["omitted", "empty"])
def test_oauth_profile_needs_all_three_credentials_before_use(field: str, supply_empty: bool) -> None:
    credentials: Final = {"client_id": "id", "client_secret": "secret", "token_url": "https://issuer"}
    supplied: Final = {key: value for key, value in credentials.items() if key != field}
    replacement: Final = {field: ""} if supply_empty else {}
    profiles: Final = parse_auth_profiles(
        "tickets",
        [{"id": "build", "auth_type": "oauth2", "oauth2_flow": "client_credentials", **supplied, **replacement}],
    )
    assert isinstance(profiles, tuple)
    assert profile_credentials_error("tickets", profiles) == InvalidAuthProfiles(
        f"Profile 'build' of MCP server 'tickets' requires {field}"
    )


def test_every_missing_credential_is_named() -> None:
    profiles: Final = parse_auth_profiles(
        "tickets", [{"id": "build", "auth_type": "oauth2", "oauth2_flow": "client_credentials", "client_id": "id"}]
    )
    assert isinstance(profiles, tuple)
    assert profile_credentials_error("tickets", profiles) == InvalidAuthProfiles(
        "Profile 'build' of MCP server 'tickets' requires client_secret, token_url"
    )
