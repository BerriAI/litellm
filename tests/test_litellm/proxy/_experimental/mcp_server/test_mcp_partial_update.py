"""
Tests for partial-update semantics of PUT /v1/mcp/server.

A partial update must only write the fields the caller explicitly provided.
Omitting a field must NOT reset it to its Pydantic schema default (e.g.
``transport=sse``, ``mcp_access_groups=[]``, ``allow_all_keys=False``), which
would silently overwrite the existing DB row.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy._experimental.mcp_server.db import (
    create_mcp_server,
    update_mcp_server,
)
from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest


def _mock_prisma():
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_mcpservertable = AsyncMock()
    mock_prisma.db.litellm_mcpservertable.update = AsyncMock(return_value=MagicMock())
    mock_prisma.db.litellm_mcpservertable.create = AsyncMock(return_value=MagicMock())
    return mock_prisma


async def _run_update(data: UpdateMCPServerRequest, fields_set=None) -> dict:
    mock_prisma = _mock_prisma()
    await update_mcp_server(mock_prisma, data, "test-user", fields_set=fields_set)
    return mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]


@pytest.mark.asyncio
async def test_partial_update_omits_unset_defaultful_fields():
    """
    A PUT touching only allowed_tools must not write transport,
    mcp_access_groups, allow_all_keys, available_on_public_internet,
    delegate_auth_to_upstream, is_byok, args, env or byok_description.
    """
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        allowed_tools=["foo"],
    )

    data_dict = await _run_update(data)

    # The intended change is present.
    assert data_dict["allowed_tools"] == ["foo"]

    # Fields the caller did not provide must not be in the write payload, so the
    # existing DB value is preserved.
    for trapped_field in (
        "transport",
        "mcp_access_groups",
        "allow_all_keys",
        "available_on_public_internet",
        "delegate_auth_to_upstream",
        "is_byok",
        "args",
        "env",
        "byok_description",
    ):
        assert trapped_field not in data_dict, (
            f"{trapped_field} should not be written on a partial update that "
            f"omitted it (would reset the row to a schema default)"
        )


@pytest.mark.asyncio
async def test_partial_update_preserves_http_transport():
    """The reported prod incident: a PUT without transport must not flip http->sse."""
    data = UpdateMCPServerRequest(
        server_id="atlassian_url",
        allowed_tools=[],
    )

    data_dict = await _run_update(data)

    assert "transport" not in data_dict
    assert data_dict["allowed_tools"] == []


@pytest.mark.asyncio
async def test_partial_update_writes_explicitly_provided_fields():
    """Explicitly provided fields are written, including falsy/default-equal values."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        url="https://example.com/mcp",
        transport="http",
        allow_all_keys=False,
        mcp_access_groups=["mcp-dev-sandbox"],
        available_on_public_internet=True,
    )

    data_dict = await _run_update(data)

    assert data_dict["transport"] == "http"
    # Explicitly provided False must still be written.
    assert data_dict["allow_all_keys"] is False
    assert data_dict["mcp_access_groups"] == ["mcp-dev-sandbox"]
    assert data_dict["available_on_public_internet"] is True


@pytest.mark.asyncio
async def test_partial_update_can_explicitly_reset_allow_all_keys():
    """Caller can still reset a field to its default by sending it explicitly."""
    enabled = await _run_update(
        UpdateMCPServerRequest(server_id="s", allow_all_keys=True)
    )
    assert enabled["allow_all_keys"] is True

    disabled = await _run_update(
        UpdateMCPServerRequest(server_id="s", allow_all_keys=False)
    )
    assert disabled["allow_all_keys"] is False


@pytest.mark.asyncio
async def test_partial_update_does_not_clear_alias_when_unset():
    """alias is force-normalized on the payload; an unset/None alias must not be written."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        allowed_tools=["foo"],
    )
    fields_set = set(data.fields_set())
    # Simulate validate_and_normalize_mcp_server_payload assigning alias=None.
    data.alias = None

    data_dict = await _run_update(data, fields_set=fields_set)

    assert "alias" not in data_dict


@pytest.mark.asyncio
async def test_partial_update_can_explicitly_clear_alias():
    """Caller can clear an existing alias by explicitly sending alias=None."""
    data = UpdateMCPServerRequest(
        server_id="my-test-server",
        alias=None,
    )
    fields_set = set(data.fields_set())
    # Simulate validate_and_normalize_mcp_server_payload preserving alias=None.
    data.alias = None

    data_dict = await _run_update(data, fields_set=fields_set)

    assert "alias" in data_dict
    assert data_dict["alias"] is None


@pytest.mark.asyncio
async def test_create_still_writes_defaults():
    """
    Regression guard: create (POST) must keep writing defaults so DB columns
    without a default get populated. exclude_unset is update-only.
    """
    mock_prisma = _mock_prisma()
    data = NewMCPServerRequest(
        server_id="new-server",
        url="https://example.com/mcp",
        transport="http",
    )

    await create_mcp_server(mock_prisma, data, "test-user")

    data_dict = mock_prisma.db.litellm_mcpservertable.create.call_args[1]["data"]

    assert data_dict["transport"] == "http"
    # is_byok is force-written on create.
    assert data_dict["is_byok"] is False
    # alias key is always present on create (even if None).
    assert "alias" in data_dict
    # audit fields set by create_mcp_server.
    assert data_dict["created_by"] == "test-user"
    assert data_dict["updated_by"] == "test-user"
