"""
Regression tests for LIT-3423 / LIT-3424.

Bug summary
-----------
PUT /v1/mcp/server with a partial body was silently overwriting omitted
non-Optional fields with their Pydantic defaults because
_prepare_mcp_server_data called model_dump(exclude_none=True).
This caused fields like transport, mcp_access_groups, allow_all_keys,
available_on_public_internet, delegate_auth_to_upstream and is_byok
to be reset on the DB row when the caller only intended to update one field
(e.g. allowed_tools). A related symptom (LIT-3424) was that
allowed_tools: null sent from the UI to clear the list was silently dropped
by exclude_none=True and the existing value stuck.

Fix
---
The update path now uses model_dump(exclude_unset=True) so only fields
the caller explicitly set on the request participate in the dict, and explicit
None survives so the UI can clear nullable fields.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.proxy._experimental.mcp_server.db import (
    _prepare_mcp_server_data,
    update_mcp_server,
)
from litellm.proxy._experimental.mcp_server.utils import (
    validate_and_normalize_mcp_server_payload,
)
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    NewMCPServerRequest,
    UpdateMCPServerRequest,
)


# ---------------------------------------------------------------------------
# _prepare_mcp_server_data — pure-function level
# ---------------------------------------------------------------------------


class TestPrepareUpdateOmitsUnsetFields:
    """LIT-3423: partial UpdateMCPServerRequest must not pull in Pydantic defaults."""

    def test_partial_update_omits_transport_default(self):
        req = UpdateMCPServerRequest(
            server_id="srv-1", allowed_tools=["read"]
        )
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "transport" not in out
        assert out.get("allowed_tools") == ["read"]

    def test_partial_update_omits_mcp_access_groups_default(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "mcp_access_groups" not in out

    def test_partial_update_omits_allow_all_keys_default(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "allow_all_keys" not in out

    def test_partial_update_omits_available_on_public_internet_default(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "available_on_public_internet" not in out

    def test_partial_update_omits_delegate_auth_to_upstream_default(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "delegate_auth_to_upstream" not in out

    def test_partial_update_omits_is_byok_default(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "is_byok" not in out

    def test_partial_update_omits_args_env_byok_description_defaults(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        out = _prepare_mcp_server_data(req, is_update=True)
        for field in ("args", "env", "byok_description"):
            assert field not in out, f"{field} must not be set from default on a partial update"

    def test_partial_update_keeps_only_explicit_fields(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read", "write"])
        out = _prepare_mcp_server_data(req, is_update=True)
        # Only the caller-provided fields plus nothing else should show up.
        assert set(out.keys()) == {"server_id", "allowed_tools"}

    def test_explicit_field_in_update_passes_through(self):
        req = UpdateMCPServerRequest(
            server_id="srv-1", transport="http", url="https://example.com/mcp"
        )
        out = _prepare_mcp_server_data(req, is_update=True)
        assert out.get("transport") == "http"
        assert out.get("url") == "https://example.com/mcp"


class TestPrepareUpdateClearsExplicitNone:
    """LIT-3424: explicit allowed_tools: null must reach the DB as None."""

    def test_allowed_tools_explicit_null_clears(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=None)
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "allowed_tools" in out
        assert out["allowed_tools"] is None


class TestPrepareCreateUnchanged:
    """Create path must keep the historical default-applying behaviour."""

    def test_create_includes_transport_default(self):
        # NewMCPServerRequest requires url/spec_path for http/sse transport.
        req = NewMCPServerRequest(
            server_name="created",
            url="https://example.com/mcp",
            transport="http",
        )
        out = _prepare_mcp_server_data(req, is_update=False)
        assert out.get("transport") == "http"

    def test_create_force_includes_is_byok_false(self):
        req = NewMCPServerRequest(
            server_name="created",
            url="https://example.com/mcp",
            transport="http",
        )
        out = _prepare_mcp_server_data(req, is_update=False)
        # Historical contract: is_byok is force-included on creates so the
        # column always has a deterministic False even when the caller omits it.
        assert "is_byok" in out
        assert out["is_byok"] is False


# ---------------------------------------------------------------------------
# alias normalization — must not clobber stored alias on partial update
# ---------------------------------------------------------------------------


class TestValidatorDoesNotClobberAliasOnPartialUpdate:
    """LIT-3423: validator used to assign payload.alias = None when neither
    alias nor server_name was sent, marking the field as set and erasing
    the stored alias on a partial update."""

    def test_partial_update_without_alias_or_server_name_does_not_set_alias(self):
        req = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=["read"])
        validate_and_normalize_mcp_server_payload(req)
        # The validator must not have written alias back, so it stays absent
        # from the exclude_unset dump and the DB column is untouched.
        out = _prepare_mcp_server_data(req, is_update=True)
        assert "alias" not in out

    def test_partial_update_with_server_name_defaults_alias(self):
        req = UpdateMCPServerRequest(server_id="srv-1", server_name="my_server")
        validate_and_normalize_mcp_server_payload(req)
        out = _prepare_mcp_server_data(req, is_update=True)
        # The normalization should backfill alias when server_name is provided.
        assert out.get("alias") == "my_server"
        assert out.get("server_name") == "my_server"


# ---------------------------------------------------------------------------
# update_mcp_server — integration with mocked Prisma
# ---------------------------------------------------------------------------


def _existing_row(**overrides):
    row = MagicMock()
    row.server_id = overrides.get("server_id", "srv-1")
    row.alias = overrides.get("alias", "stored_alias")
    row.url = overrides.get("url", "https://upstream.example.com/mcp")
    row.transport = overrides.get("transport", "http")
    row.auth_type = overrides.get("auth_type", None)
    row.credentials = overrides.get("credentials", None)
    row.allowed_tools = overrides.get("allowed_tools", ["read", "write"])
    row.mcp_access_groups = overrides.get("mcp_access_groups", ["dev-sandbox"])
    row.allow_all_keys = overrides.get("allow_all_keys", True)
    row.available_on_public_internet = overrides.get("available_on_public_internet", False)
    row.is_byok = overrides.get("is_byok", True)
    return row


def _mock_prisma(existing, captured):
    prisma = MagicMock()
    prisma.db = MagicMock()
    prisma.db.litellm_mcpservertable = MagicMock()
    prisma.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)

    async def _update(*, where, data):
        captured["where"] = where
        captured["data"] = data
        return existing

    prisma.db.litellm_mcpservertable.update = _update
    return prisma


@pytest.mark.asyncio
async def test_partial_update_does_not_overwrite_omitted_fields():
    """LIT-3423 root repro: PUT with only allowed_tools must not send
    transport, mcp_access_groups, allow_all_keys,
    available_on_public_internet or is_byok to Prisma."""
    captured: dict = {}
    existing = _existing_row()
    prisma = _mock_prisma(existing, captured)

    payload = UpdateMCPServerRequest(
        server_id="srv-1", allowed_tools=["read"]
    )
    validate_and_normalize_mcp_server_payload(payload)

    await update_mcp_server(prisma, payload, touched_by="admin")

    data = captured["data"]
    # Caller-set fields propagate.
    assert data["server_id"] == "srv-1"
    assert data["allowed_tools"] == ["read"]
    assert data["updated_by"] == "admin"
    # The whole point of the fix:
    for forbidden in (
        "transport",
        "mcp_access_groups",
        "allow_all_keys",
        "available_on_public_internet",
        "delegate_auth_to_upstream",
        "is_byok",
        "args",
        "env",
        "byok_description",
        "alias",
    ):
        assert forbidden not in data, (
            f"{forbidden} leaked into partial-update payload — would clobber DB"
        )


@pytest.mark.asyncio
async def test_partial_update_allowed_tools_null_clears():
    """LIT-3424: explicit allowed_tools: null must reach Prisma as None."""
    captured: dict = {}
    existing = _existing_row()
    prisma = _mock_prisma(existing, captured)

    payload = UpdateMCPServerRequest(server_id="srv-1", allowed_tools=None)
    validate_and_normalize_mcp_server_payload(payload)

    await update_mcp_server(prisma, payload, touched_by="admin")

    data = captured["data"]
    assert "allowed_tools" in data, "explicit null must propagate"
    assert data["allowed_tools"] is None
