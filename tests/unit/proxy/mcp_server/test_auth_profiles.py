from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.mcp_server.auth_profiles import (
    DuplicateProfileId,
    InvalidProfileId,
    MalformedProfile,
    StoredProfileId,
    UnsupportedAuthType,
    UnsupportedOAuth2Flow,
    UnsupportedProfileHost,
    ValidatedProfiles,
    describe_refusal,
    profile_host_refusal,
    refusal_http_status,
    validate_auth_profiles,
)
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile


def _bot_profile(profile_id: object = "ana-bots", **overrides: object) -> dict[str, object]:
    return {
        "id": profile_id,
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
        "client_id": "bot-app",
        "client_secret": "bot-secret",
        "token_url": "https://auth.example/token",
        **overrides,
    }


@pytest.mark.parametrize("profile_id", ["a", "ANA-Bots", "ana_bots", "123", "x" * 64, "A-1_b"])
def test_an_id_inside_the_grammar_is_accepted_as_written(profile_id: str) -> None:
    result: Final = validate_auth_profiles({"analytics": [_bot_profile(profile_id)]}, ())

    assert result == ValidatedProfiles(
        {
            "analytics": (
                MCPAuthProfile(
                    id=profile_id,
                    auth_type="oauth2",
                    oauth2_flow="client_credentials",
                    client_id="bot-app",
                    client_secret="bot-secret",
                    token_url="https://auth.example/token",
                ),
            )
        }
    ), f"{profile_id!r} must be stored with its original case"


@pytest.mark.parametrize(
    "profile_id",
    ["", "x" * 65, "a b", "bötS", "default", "DEFAULT", "Default", "İ-bot", "botK", "bot\n", False, True, 123, None],
)
def test_an_id_outside_the_grammar_is_refused_without_coercion(profile_id: object) -> None:
    result: Final = validate_auth_profiles({"analytics": [_bot_profile(profile_id)]}, ())

    assert result == InvalidProfileId("analytics", 0, profile_id)
    assert refusal_http_status(result) == 400


def test_a_missing_id_is_refused() -> None:
    profile: Final = {key: value for key, value in _bot_profile().items() if key != "id"}

    assert validate_auth_profiles({"analytics": [profile]}, ()) == InvalidProfileId("analytics", 0, None)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"oauth2_flow": "authorization_code"}, id="authorization-code"),
        pytest.param({"oauth2_flow": None}, id="flow-null"),
        pytest.param({"oauth2_flow": "client_credential"}, id="flow-misspelled"),
        pytest.param(
            {"authorization_url": "https://auth.example/authorize", "oauth2_flow": None}, id="interactive-shape"
        ),
    ],
)
def test_an_oauth2_profile_without_an_explicit_client_credentials_flow_is_refused(overrides: dict[str, object]) -> None:
    profile: Final = {**_bot_profile(), **overrides}

    result: Final = validate_auth_profiles({"analytics": [profile]}, ())

    assert result == UnsupportedOAuth2Flow("analytics", 0, "ana-bots", profile["oauth2_flow"])
    assert refusal_http_status(result) == 400


def test_an_oauth2_profile_that_omits_the_flow_key_is_refused_even_with_machine_shaped_fields() -> None:
    profile: Final = {key: value for key, value in _bot_profile().items() if key != "oauth2_flow"}

    assert validate_auth_profiles({"analytics": [profile]}, ()) == UnsupportedOAuth2Flow(
        "analytics", 0, "ana-bots", None
    )


def test_an_oauth2_profile_read_without_the_client_credentials_flow_is_refused_by_its_type() -> None:
    with pytest.raises(ValidationError, match="requires oauth2_flow client_credentials"):
        MCPAuthProfile.model_validate({key: value for key, value in _bot_profile().items() if key != "oauth2_flow"})


@pytest.mark.parametrize(
    "auth_type",
    ["oauth2_token_exchange", "oauth2_id_jag", "true_passthrough", "oauth_delegate", "aws_sigv4", "none"],
)
def test_a_kind_that_needs_a_caller_is_not_supported_in_an_additional_profile(auth_type: str) -> None:
    result: Final = validate_auth_profiles({"analytics": [{"id": "p1", "auth_type": auth_type}]}, ())

    assert result == UnsupportedAuthType("analytics", 0, "p1", auth_type)
    assert "not supported in an additional profile" in describe_refusal(result)
    assert refusal_http_status(result) == 400


@pytest.mark.parametrize("auth_type", ["api_key", "bearer_token", "basic", "token", "authorization"])
def test_a_static_credential_profile_is_accepted(auth_type: str) -> None:
    result: Final = validate_auth_profiles(
        {"gh": [{"id": "gh-bots", "auth_type": auth_type, "authentication_token": "k"}]}, ()
    )

    assert result == ValidatedProfiles(
        {"gh": (MCPAuthProfile(id="gh-bots", auth_type=auth_type, authentication_token="k"),)}
    )


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("not-a-list", id="string"),
        pytest.param({"id": "p1"}, id="mapping"),
        pytest.param(7, id="number"),
    ],
)
def test_additional_auth_must_be_a_list(raw: object) -> None:
    assert validate_auth_profiles({"analytics": raw}, ()) == MalformedProfile(
        "analytics", None, "additional_auth must be a list of profiles"
    )


@pytest.mark.parametrize(
    ("profile", "named_in_detail"),
    [
        pytest.param("p1", "a profile must be a mapping", id="not-a-mapping"),
        pytest.param(
            {"id": "p1", "auth_type": "bearer_token", "client_secert": "x"}, "client_secert", id="unknown-key"
        ),
        pytest.param(
            {"id": "p1", "auth_type": "bearer_token", "oauth2_flow": "client_credentials"},
            "oauth2_flow is only valid for an oauth2 profile",
            id="flow-on-static-kind",
        ),
        pytest.param({"id": "p1", "auth_type": "bogus"}, "auth_type", id="unknown-kind"),
    ],
)
def test_a_malformed_profile_is_refused_with_its_position(profile: object, named_in_detail: str) -> None:
    result: Final = validate_auth_profiles({"analytics": [_bot_profile("first"), profile]}, ())

    assert isinstance(result, MalformedProfile), result
    assert (result.server, result.position) == ("analytics", 1)
    assert named_in_detail in result.detail
    assert refusal_http_status(result) == 400


def test_two_servers_whose_ids_differ_by_case_collide_naming_both_ids_and_servers() -> None:
    result: Final = validate_auth_profiles(
        {"analytics": [_bot_profile("Ana-Bots")], "news": [_bot_profile("ana-bots")]}, ()
    )

    assert result == DuplicateProfileId("ana-bots", "news", "Ana-Bots", "analytics")
    assert describe_refusal(result) == (
        "Profile id 'ana-bots' of MCP server 'news' collides with profile id 'Ana-Bots' of MCP server "
        "'analytics'. Profile ids are unique across all servers, ignoring case."
    )
    assert refusal_http_status(result) == 409


def test_two_profiles_of_one_server_collide() -> None:
    assert validate_auth_profiles({"analytics": [_bot_profile("bots-1"), _bot_profile("Bots-1")]}, ()) == (
        DuplicateProfileId("Bots-1", "analytics", "bots-1", "analytics")
    )


def test_a_submitted_id_collides_with_a_stored_profile_of_another_server() -> None:
    result: Final = validate_auth_profiles(
        {"github": [_bot_profile("ANA_Bots")]}, (StoredProfileId("ana_bots", "analytics"),)
    )

    assert result == DuplicateProfileId("ANA_Bots", "github", "ana_bots", "analytics")


def test_an_underscore_is_a_literal_so_ana_bots_does_not_collide_with_anaxbots() -> None:
    stored: Final = (StoredProfileId("ANAxbots", "a"),)

    first: Final = validate_auth_profiles({"b": [_bot_profile("ANA_Bots")]}, stored)
    second: Final = validate_auth_profiles(
        {"b": [_bot_profile("ANA_Bots")]}, (*stored, StoredProfileId("ANA_Bots", "b"))
    )
    third: Final = validate_auth_profiles(
        {"c": [_bot_profile("ANA_bots")]}, (*stored, StoredProfileId("ANA_Bots", "b"))
    )

    assert isinstance(first, ValidatedProfiles)
    assert isinstance(second, ValidatedProfiles), "a submitted server replaces its own stored profiles"
    assert third == DuplicateProfileId("ANA_bots", "c", "ANA_Bots", "b")


def test_a_submitted_list_replaces_the_stored_profiles_of_its_server() -> None:
    stored: Final = (StoredProfileId("old-bots", "analytics"), StoredProfileId("keep", "news"))

    assert validate_auth_profiles({"analytics": [_bot_profile("OLD-BOTS")]}, stored) == ValidatedProfiles(
        {"analytics": (MCPAuthProfile.model_validate(_bot_profile("OLD-BOTS")),)}
    )
    assert validate_auth_profiles({"analytics": []}, stored) == ValidatedProfiles({"analytics": ()})
    assert validate_auth_profiles({"analytics": [_bot_profile("KEEP")]}, stored) == DuplicateProfileId(
        "KEEP", "analytics", "keep", "news"
    )


def test_a_parse_refusal_wins_over_a_collision() -> None:
    result: Final = validate_auth_profiles(
        {"analytics": [_bot_profile("bots"), {"id": "x", "auth_type": "aws_sigv4"}]},
        (StoredProfileId("BOTS", "news"),),
    )

    assert result == UnsupportedAuthType("analytics", 1, "x", "aws_sigv4")


@pytest.mark.parametrize(
    ("transport", "is_byok", "spec_path", "expected"),
    [
        pytest.param("http", False, None, None, id="http"),
        pytest.param("sse", False, "", None, id="sse"),
        pytest.param("stdio", False, None, UnsupportedProfileHost("s", "stdio"), id="stdio"),
        pytest.param("http", True, None, UnsupportedProfileHost("s", "BYOK"), id="byok"),
        pytest.param("http", False, "./openapi.json", UnsupportedProfileHost("s", "OpenAPI"), id="openapi"),
    ],
)
def test_profiles_are_hosted_only_by_servers_that_send_their_own_credential(
    transport: str, is_byok: bool, spec_path: str | None, expected: UnsupportedProfileHost | None
) -> None:
    assert profile_host_refusal("s", transport=transport, is_byok=is_byok, spec_path=spec_path) == expected
