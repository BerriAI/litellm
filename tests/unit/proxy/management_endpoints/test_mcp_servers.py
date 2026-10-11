from datetime import datetime
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    MCPTransport,
    NewMCPServerRequest,
)
from litellm.proxy._experimental.mcp_server.db import McpIdentifierConflict
from litellm.proxy._experimental.mcp_server.utils import validate_mcp_server_name
from litellm.proxy.management_endpoints.mcp_management_endpoints import (
    _redact_mcp_credentials,
    does_mcp_server_exist,
    raise_mcp_identifier_conflict,
    validate_and_normalize_mcp_server_payload,
)


def test_does_mcp_server_exist() -> None:
    created_at: Final = datetime(2025, 1, 1)
    records: Final = (
        LiteLLM_MCPServerTable(
            server_id="server-1",
            alias="first",
            url="https://first.example.com/mcp",
            transport=MCPTransport.sse,
            created_at=created_at,
            updated_at=created_at,
        ),
        LiteLLM_MCPServerTable(
            server_id="server-2",
            alias="second",
            url="https://second.example.com/mcp",
            transport=MCPTransport.http,
            created_at=created_at,
            updated_at=created_at,
        ),
    )

    assert does_mcp_server_exist(records, "server-1")
    assert does_mcp_server_exist(records, "server-2")
    assert not does_mcp_server_exist(records, "missing-server")


def test_create_duplicate_mcp_server_returns_conflict_error() -> None:
    conflict: Final = McpIdentifierConflict(
        field="alias",
        value="duplicate-server",
        server_id="existing-server",
    )

    with pytest.raises(HTTPException) as error:
        raise_mcp_identifier_conflict(conflict)

    assert error.value.status_code == 400
    assert error.value.detail == {
        "error": "An MCP server with alias 'duplicate-server' already exists "
        "(server_id=existing-server). MCP server names and aliases must be unique, case-insensitive."
    }


def test_create_mcp_server_direct_normalizes_alias() -> None:
    payload: Final = NewMCPServerRequest(
        server_id="server-1",
        alias="Test Server",
        url="https://server.example.com/mcp",
        transport=MCPTransport.sse,
    )

    validate_and_normalize_mcp_server_payload(payload)

    assert payload.alias == "Test_Server"
    assert payload.url == "https://server.example.com/mcp"
    assert payload.transport == MCPTransport.sse


def test_edit_mcp_server_redacts_credentials() -> None:
    created_at: Final = datetime(2025, 1, 1)
    server: Final = LiteLLM_MCPServerTable(
        server_id="server-1",
        alias="server",
        url="https://server.example.com/mcp",
        transport=MCPTransport.http,
        created_at=created_at,
        updated_at=created_at,
        credentials={"auth_value": "secret-value"},
    )

    redacted: Final = _redact_mcp_credentials(server)

    assert redacted.credentials is None
    assert server.credentials == {"auth_value": "secret-value"}


def test_create_mcp_server_invalid_alias() -> None:
    payload: Final = NewMCPServerRequest(
        alias="invalid-alias",
        url="https://server.example.com/mcp",
        transport=MCPTransport.sse,
    )

    with pytest.raises(HTTPException) as error:
        validate_and_normalize_mcp_server_payload(payload)

    assert error.value.status_code == 400
    assert error.value.detail == {
        "error": "Server name cannot contain '-'. Use an alternative character instead Found: invalid-alias"
    }


def test_validate_mcp_server_name_direct() -> None:
    validate_mcp_server_name("valid_name")
    validate_mcp_server_name("valid name")

    with pytest.raises(Exception, match="Server name cannot contain"):
        validate_mcp_server_name("invalid-name")

    with pytest.raises(HTTPException) as error:
        validate_mcp_server_name("invalid-name", raise_http_exception=True)

    assert error.value.status_code == 400
    assert error.value.detail == {
        "error": "Server name cannot contain '-'. Use an alternative character instead Found: invalid-name"
    }
