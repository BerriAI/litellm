import json
from typing import Final

import pytest

from litellm.proxy._experimental.mcp_server.client_allowlist import (
    MCP_ALLOWED_CLIENTS_SETTING,
    MCPClientRejection,
    check_mcp_client_allowed,
    extract_mcp_client_name,
    parse_allowed_mcp_clients,
)


def _initialize_body(client_info: object) -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": client_info},
        }
    ).encode()


CLAUDE_CODE: Final = _initialize_body({"name": "claude-code", "version": "2.1.274"})
ANTIGRAVITY: Final = _initialize_body({"name": "antigravity-cli", "version": "1.0.0"})


@pytest.mark.parametrize(
    ("raw_setting", "expected"),
    (
        (None, None),
        ([], frozenset()),
        (["antigravity-cli"], frozenset({"antigravity-cli"})),
        (["antigravity-cli", "codex-mcp-client"], frozenset({"antigravity-cli", "codex-mcp-client"})),
        ("antigravity-cli", frozenset()),
        ([1, "antigravity-cli"], frozenset()),
        ({"name": "antigravity-cli"}, frozenset()),
    ),
)
def test_parse_allowed_mcp_clients(raw_setting: object, expected: frozenset[str] | None) -> None:
    assert parse_allowed_mcp_clients(raw_setting) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    (
        (CLAUDE_CODE, "claude-code"),
        (_initialize_body({"name": "", "version": "1"}), None),
        (_initialize_body({"version": "1"}), None),
        (_initialize_body({"name": 7}), None),
        (_initialize_body("claude-code"), None),
        (b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}', None),
        (b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":[]}', None),
        (b'["not", "an", "object"]', None),
        (b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"clientInfo":{"name":"clau', None),
        (b"\xff\xfe", None),
        (b"", None),
    ),
)
def test_extract_mcp_client_name(body: bytes, expected: str | None) -> None:
    assert extract_mcp_client_name(body) == expected


def test_unconfigured_allowlist_admits_every_client_including_unidentified_ones() -> None:
    assert check_mcp_client_allowed(CLAUDE_CODE, None) is None
    assert check_mcp_client_allowed(b'{"method":"initialize","params":{}}', None) is None
    assert check_mcp_client_allowed(b"garbage", None) is None


def test_listed_client_is_admitted_and_unlisted_client_is_rejected_by_name() -> None:
    allowed: Final = frozenset({"antigravity-cli"})
    assert check_mcp_client_allowed(ANTIGRAVITY, allowed) is None
    assert check_mcp_client_allowed(CLAUDE_CODE, allowed) == MCPClientRejection(client_name="claude-code")


def test_matching_is_exact_not_prefix_or_case_insensitive() -> None:
    allowed: Final = frozenset({"claude-code"})
    assert check_mcp_client_allowed(_initialize_body({"name": "Claude-Code"}), allowed) is not None
    assert check_mcp_client_allowed(_initialize_body({"name": "claude-code-sdk"}), allowed) is not None
    assert check_mcp_client_allowed(_initialize_body({"name": " claude-code"}), allowed) is not None


def test_empty_allowlist_rejects_every_client() -> None:
    assert check_mcp_client_allowed(ANTIGRAVITY, frozenset()) == MCPClientRejection(client_name="antigravity-cli")
    assert check_mcp_client_allowed(CLAUDE_CODE, frozenset()) == MCPClientRejection(client_name="claude-code")


def test_missing_or_malformed_client_metadata_is_rejected_when_allowlist_is_set() -> None:
    allowed: Final = frozenset({"antigravity-cli"})
    assert check_mcp_client_allowed(_initialize_body({"version": "1"}), allowed) == MCPClientRejection(None)
    assert check_mcp_client_allowed(b'{"method":"initialize","params":{}}', allowed) == MCPClientRejection(None)
    assert check_mcp_client_allowed(b"{not json", allowed) == MCPClientRejection(None)


def test_rejection_details_name_the_setting_and_the_offending_client() -> None:
    named: Final = MCPClientRejection(client_name="claude-code").details
    assert "claude-code" in named
    assert MCP_ALLOWED_CLIENTS_SETTING in named

    anonymous: Final = MCPClientRejection(client_name=None).details
    assert "clientInfo.name" in anonymous
    assert MCP_ALLOWED_CLIENTS_SETTING in anonymous
    assert "None" not in anonymous
