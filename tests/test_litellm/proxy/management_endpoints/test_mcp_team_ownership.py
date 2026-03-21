"""Tests for MCP server team ownership via MCPServerTable.team_id."""

import uuid

import pytest


def test_prepare_mcp_server_data_includes_team_id():
    """team_id should be included in data dict when set."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import NewMCPServerRequest

    team_id = str(uuid.uuid4())
    request = NewMCPServerRequest(
        server_name="test_server",
        transport="http",
        url="https://example.com/mcp",
        team_id=team_id,
    )
    data_dict = _prepare_mcp_server_data(request)
    assert data_dict["team_id"] == team_id


def test_prepare_mcp_server_data_excludes_none_team_id():
    """team_id=None should not be in the data dict (exclude_none=True)."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import NewMCPServerRequest

    request = NewMCPServerRequest(
        server_name="test_server",
        transport="http",
        url="https://example.com/mcp",
    )
    data_dict = _prepare_mcp_server_data(request)
    assert "team_id" not in data_dict


def test_prepare_mcp_server_data_update_includes_team_id():
    """UpdateMCPServerRequest with team_id should include it in data dict."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import UpdateMCPServerRequest

    team_id = str(uuid.uuid4())
    request = UpdateMCPServerRequest(
        server_id="test-server-id",
        transport="http",
        url="https://example.com/mcp",
        team_id=team_id,
    )
    data_dict = _prepare_mcp_server_data(request)
    assert data_dict["team_id"] == team_id


def test_prepare_mcp_server_data_update_excludes_none_team_id():
    """UpdateMCPServerRequest without team_id should not have it in data dict."""
    from litellm.proxy._experimental.mcp_server.db import _prepare_mcp_server_data
    from litellm.proxy._types import UpdateMCPServerRequest

    request = UpdateMCPServerRequest(
        server_id="test-server-id",
        transport="http",
        url="https://example.com/mcp",
    )
    data_dict = _prepare_mcp_server_data(request)
    assert "team_id" not in data_dict


def test_update_request_model_fields_set_detects_explicit_null():
    """When team_id is explicitly set to None in JSON, model_fields_set should contain it."""
    from litellm.proxy._types import UpdateMCPServerRequest

    # Simulate JSON: {"server_id": "x", "transport": "http", "url": "...", "team_id": null}
    request = UpdateMCPServerRequest.model_validate(
        {"server_id": "x", "transport": "http", "url": "https://example.com/mcp", "team_id": None}
    )
    assert "team_id" in request.model_fields_set

    # Simulate JSON: {"server_id": "x", "transport": "http", "url": "..."} — team_id absent
    request2 = UpdateMCPServerRequest.model_validate(
        {"server_id": "x", "transport": "http", "url": "https://example.com/mcp"}
    )
    assert "team_id" not in request2.model_fields_set
