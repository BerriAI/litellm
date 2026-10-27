from collections.abc import Mapping
from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server.client_allowlist import (
    MCP_ALLOWED_CLIENTS_SETTING,
    MCP_CLIENT_ID_HEADER_SETTING,
    MCP_CLIENT_ID_JWT_FIELD_SETTING,
    MCPClientAllowlist,
    MCPClientIdentity,
    MCPClientRejection,
    check_mcp_client_allowed,
    load_mcp_client_allowlist,
    parse_allowed_mcp_clients,
    resolve_mcp_client_identity,
)

ANTIGRAVITY: Final = {"alias": "Antigravity CLI", "value": "antigravity-cli"}
CODEX: Final = {"alias": "Codex", "value": "codex-mcp-client"}
ANTIGRAVITY_ONLY: Final[Mapping[str, str]] = {"antigravity-cli": "Antigravity CLI"}
JWT_ONLY: Final = MCPClientAllowlist(aliases_by_value=ANTIGRAVITY_ONLY, jwt_field="azp", header=None)
HEADER_ONLY: Final = MCPClientAllowlist(aliases_by_value=ANTIGRAVITY_ONLY, jwt_field=None, header="x-mcp-client")
JWT_AND_HEADER: Final = MCPClientAllowlist(aliases_by_value=ANTIGRAVITY_ONLY, jwt_field="azp", header="x-mcp-client")
NO_SOURCE: Final = MCPClientAllowlist(aliases_by_value=ANTIGRAVITY_ONLY, jwt_field=None, header=None)
NO_HEADERS: Final[Mapping[str, str]] = {}


_ALLOWLIST_SETTING_CASES: Final[tuple[tuple[object, Mapping[str, str] | None], ...]] = (
    (None, None),
    ([], {}),
    ([ANTIGRAVITY], ANTIGRAVITY_ONLY),
    ([ANTIGRAVITY, CODEX], {"antigravity-cli": "Antigravity CLI", "codex-mcp-client": "Codex"}),
    (
        [ANTIGRAVITY, {"alias": "Antigravity (prod)", "value": "antigravity-cli"}],
        {"antigravity-cli": "Antigravity (prod)"},
    ),
    (
        [ANTIGRAVITY, {"alias": "Antigravity CLI", "value": "antigravity-prod"}],
        {**ANTIGRAVITY_ONLY, "antigravity-prod": "Antigravity CLI"},
    ),
    (["antigravity-cli"], {}),
    ("antigravity-cli", {}),
    ([ANTIGRAVITY, 1], {}),
    ([{"alias": "Antigravity CLI"}], {}),
    ([{"value": "antigravity-cli"}], {}),
    ([{"alias": "", "value": "antigravity-cli"}], {}),
    ([{"alias": "Antigravity CLI", "value": ""}], {}),
    ([{"alias": "Antigravity CLI", "value": ["antigravity-cli"]}], {}),
    (ANTIGRAVITY, {}),
)


@pytest.mark.parametrize(("raw_setting", "expected"), _ALLOWLIST_SETTING_CASES)
def test_parse_allowed_mcp_clients(raw_setting: object, expected: Mapping[str, str] | None) -> None:
    assert parse_allowed_mcp_clients(raw_setting) == expected


def test_load_returns_none_when_the_allowlist_setting_is_absent_even_if_identity_sources_are_set() -> None:
    settings: Final = {"litellm_jwtauth": {"mcp_client_id_jwt_field": "azp"}, "mcp_client_id_header": "x-mcp-client"}
    assert load_mcp_client_allowlist(settings) is None


def test_load_reads_the_jwt_field_from_litellm_jwtauth_and_lowercases_the_header_name() -> None:
    settings: Final = {
        "mcp_allowed_clients": [ANTIGRAVITY, CODEX],
        "litellm_jwtauth": {"user_id_jwt_field": "sub", "mcp_client_id_jwt_field": "resource_access.mcp.client"},
        "mcp_client_id_header": "X-MCP-Client",
    }
    assert load_mcp_client_allowlist(settings) == MCPClientAllowlist(
        aliases_by_value={"antigravity-cli": "Antigravity CLI", "codex-mcp-client": "Codex"},
        jwt_field="resource_access.mcp.client",
        header="x-mcp-client",
    )


@pytest.mark.parametrize(
    "settings",
    (
        {"mcp_allowed_clients": [ANTIGRAVITY]},
        {"mcp_allowed_clients": [ANTIGRAVITY], "litellm_jwtauth": {}, "mcp_client_id_header": ""},
        {"mcp_allowed_clients": [ANTIGRAVITY], "litellm_jwtauth": {"mcp_client_id_jwt_field": ""}},
        {"mcp_allowed_clients": [ANTIGRAVITY], "litellm_jwtauth": "azp", "mcp_client_id_header": ["x"]},
    ),
)
def test_load_without_a_usable_identity_source_keeps_the_allowlist_but_no_source(
    settings: Mapping[str, object],
) -> None:
    assert load_mcp_client_allowlist(settings) == NO_SOURCE


@pytest.mark.parametrize("raw_setting", ("antigravity-cli", ["antigravity-cli"], [{"alias": "Antigravity CLI"}]))
def test_load_malformed_allowlist_admits_nobody(raw_setting: object) -> None:
    loaded: Final = load_mcp_client_allowlist({"mcp_allowed_clients": raw_setting})
    assert loaded is not None
    assert loaded.aliases_by_value == {}
    assert check_mcp_client_allowed(loaded, {"azp": "antigravity-cli"}, {"x-mcp-client": "antigravity-cli"}) is not None


def test_only_the_value_identifies_a_client_never_its_alias() -> None:
    assert check_mcp_client_allowed(JWT_ONLY, {"azp": "Antigravity CLI"}, NO_HEADERS) is not None
    assert check_mcp_client_allowed(HEADER_ONLY, None, {"x-mcp-client": "Antigravity CLI"}) is not None


def test_two_clients_may_share_an_alias_and_both_are_admitted() -> None:
    settings: Final = {
        "mcp_allowed_clients": [
            {"alias": "Coding CLI", "value": "cli-dev"},
            {"alias": "Coding CLI", "value": "cli-prod"},
        ],
        "litellm_jwtauth": {"mcp_client_id_jwt_field": "azp"},
    }
    loaded: Final = load_mcp_client_allowlist(settings)
    assert check_mcp_client_allowed(loaded, {"azp": "cli-dev"}, NO_HEADERS) is None
    assert check_mcp_client_allowed(loaded, {"azp": "cli-prod"}, NO_HEADERS) is None
    assert check_mcp_client_allowed(loaded, {"azp": "Coding CLI"}, NO_HEADERS) is not None


def test_unconfigured_allowlist_admits_callers_with_no_identity_at_all() -> None:
    assert check_mcp_client_allowed(None, None, NO_HEADERS) is None
    assert check_mcp_client_allowed(None, {"azp": "claude-code"}, {"x-mcp-client": "claude-code"}) is None


def test_jwt_claim_identifies_the_client() -> None:
    assert resolve_mcp_client_identity(JWT_ONLY, {"azp": "antigravity-cli"}, NO_HEADERS) == MCPClientIdentity(
        client_id="antigravity-cli", source="jwt", source_name="azp"
    )
    assert check_mcp_client_allowed(JWT_ONLY, {"azp": "antigravity-cli"}, NO_HEADERS) is None


def test_nested_jwt_claim_path_is_resolved_with_dot_notation() -> None:
    nested: Final = MCPClientAllowlist(
        aliases_by_value=ANTIGRAVITY_ONLY, jwt_field="resource_access.mcp.client", header=None
    )
    claims: Final = {"resource_access": {"mcp": {"client": "antigravity-cli"}}}
    assert check_mcp_client_allowed(nested, claims, NO_HEADERS) is None


def test_unlisted_jwt_client_is_rejected_and_the_rejection_names_it() -> None:
    rejection: Final = check_mcp_client_allowed(JWT_ONLY, {"azp": "claude-code"}, NO_HEADERS)
    assert isinstance(rejection, MCPClientRejection)
    assert "'claude-code'" in rejection.details
    assert "azp" in rejection.details
    assert MCP_ALLOWED_CLIENTS_SETTING in rejection.details
    assert rejection.response_body == {"error": "Forbidden", "details": rejection.details}


@pytest.mark.parametrize("claims", ({"sub": "user-1"}, {"azp": ""}, {"azp": 42}, {"azp": ["antigravity-cli"]}))
def test_jwt_without_a_usable_client_claim_is_rejected(claims: Mapping[str, object]) -> None:
    rejection: Final = check_mcp_client_allowed(JWT_ONLY, claims, NO_HEADERS)
    assert isinstance(rejection, MCPClientRejection)
    assert "azp" in rejection.details


def test_matching_is_exact_not_prefix_or_case_insensitive() -> None:
    for spoof in ("Antigravity-Cli", "antigravity-cli-sdk", " antigravity-cli"):
        assert check_mcp_client_allowed(JWT_ONLY, {"azp": spoof}, NO_HEADERS) is not None
        assert check_mcp_client_allowed(HEADER_ONLY, None, {"x-mcp-client": spoof}) is not None


def test_configured_header_identifies_callers_without_a_jwt() -> None:
    headers: Final = {"x-mcp-client": "antigravity-cli"}
    assert resolve_mcp_client_identity(HEADER_ONLY, None, headers) == MCPClientIdentity(
        client_id="antigravity-cli", source="header", source_name="x-mcp-client"
    )
    assert check_mcp_client_allowed(HEADER_ONLY, None, headers) is None
    assert check_mcp_client_allowed(HEADER_ONLY, {}, headers) is None


def test_unlisted_or_missing_header_is_rejected() -> None:
    unlisted: Final = check_mcp_client_allowed(HEADER_ONLY, None, {"x-mcp-client": "claude-code"})
    assert isinstance(unlisted, MCPClientRejection)
    assert "'claude-code'" in unlisted.details
    for headers in (NO_HEADERS, {"x-mcp-client": ""}, {"x-other": "antigravity-cli"}):
        missing = check_mcp_client_allowed(HEADER_ONLY, None, headers)
        assert isinstance(missing, MCPClientRejection)
        assert "x-mcp-client" in missing.details


def test_header_is_not_consulted_when_it_is_not_configured() -> None:
    rejection: Final = check_mcp_client_allowed(JWT_ONLY, None, {"x-mcp-client": "antigravity-cli"})
    assert isinstance(rejection, MCPClientRejection)
    assert MCP_CLIENT_ID_HEADER_SETTING in rejection.details
    assert MCP_CLIENT_ID_JWT_FIELD_SETTING in rejection.details


def test_jwt_caller_is_judged_by_its_claim_even_when_the_header_would_pass() -> None:
    spoofed_header: Final = {"x-mcp-client": "antigravity-cli"}
    assert check_mcp_client_allowed(JWT_AND_HEADER, {"azp": "claude-code"}, spoofed_header) is not None
    assert check_mcp_client_allowed(JWT_AND_HEADER, {"sub": "user-1"}, spoofed_header) is not None
    assert check_mcp_client_allowed(JWT_AND_HEADER, {"azp": "antigravity-cli"}, {"x-mcp-client": "claude-code"}) is None


def test_jwt_caller_with_an_empty_claim_set_cannot_fall_back_to_the_header() -> None:
    rejection: Final = check_mcp_client_allowed(JWT_AND_HEADER, {}, {"x-mcp-client": "antigravity-cli"})
    assert isinstance(rejection, MCPClientRejection)
    assert "azp" in rejection.details


def test_non_jwt_caller_falls_back_to_the_header_when_both_sources_are_configured() -> None:
    assert check_mcp_client_allowed(JWT_AND_HEADER, None, {"x-mcp-client": "antigravity-cli"}) is None
    assert check_mcp_client_allowed(JWT_AND_HEADER, None, {"x-mcp-client": "claude-code"}) is not None


def test_allowlist_with_no_identity_source_rejects_everyone_and_says_what_to_configure() -> None:
    rejection: Final = check_mcp_client_allowed(
        NO_SOURCE, {"azp": "antigravity-cli"}, {"x-mcp-client": "antigravity-cli"}
    )
    assert isinstance(rejection, MCPClientRejection)
    assert MCP_CLIENT_ID_JWT_FIELD_SETTING in rejection.details
    assert MCP_CLIENT_ID_HEADER_SETTING in rejection.details


def test_empty_allowlist_rejects_an_identified_client() -> None:
    empty: Final = MCPClientAllowlist(aliases_by_value={}, jwt_field="azp", header="x-mcp-client")
    assert check_mcp_client_allowed(empty, {"azp": "antigravity-cli"}, NO_HEADERS) is not None
    assert check_mcp_client_allowed(empty, None, {"x-mcp-client": "antigravity-cli"}) is not None
