"""
Regression tests for LIT-3423: Partial PUT /v1/mcp/server must not silently
overwrite omitted fields with Pydantic defaults.

Before the fix, ``_prepare_mcp_server_data`` used ``model_dump(exclude_none=True)``,
which kept the schema defaults (transport=sse, mcp_access_groups=[],
allow_all_keys=False, available_on_public_internet=True,
delegate_auth_to_upstream=False, is_byok=False, args=[], env={},
byok_description=[]). The update path then wrote those defaults to the DB
row, silently clobbering the existing values. The fix switches the update
path to ``model_dump(exclude_unset=True)`` so only fields the caller
actually sent end up in the UPDATE statement.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestPartialUpdatePreservesOmittedFields:
    """LIT-3423: omitted fields must NOT appear in the update dict."""

    @pytest.mark.asyncio
    async def test_partial_update_omits_transport_when_caller_did_not_set_it(self):
        """A PUT updating only ``description`` must NOT include transport in
        the dict passed to prisma.update — otherwise an HTTP-configured server
        would silently flip to SSE (the production incident this fixes)."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            description="Updated description",
        )

        await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert "transport" not in data_dict
        for field in (
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
            assert field not in data_dict, (
                f"{field!r} was silently included in the update; "
                "this would overwrite the existing DB row (LIT-3423)."
            )
        assert data_dict["description"] == "Updated description"
        assert data_dict["updated_by"] == "test-user"

    @pytest.mark.asyncio
    async def test_partial_update_only_includes_explicitly_set_fields(self):
        """Update with allowed_tools=["tool_a"] must touch only allowed_tools."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            allowed_tools=["tool_a"],
        )

        await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert data_dict.get("allowed_tools") == ["tool_a"]
        assert "transport" not in data_dict
        assert "mcp_access_groups" not in data_dict
        assert "allow_all_keys" not in data_dict
        assert "is_byok" not in data_dict
        assert "env" not in data_dict

    @pytest.mark.asyncio
    async def test_partial_update_explicit_field_value_is_included(self):
        """Explicitly setting transport=http must be persisted."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import MCPTransport, UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            transport=MCPTransport.http,
            url="https://example.com/mcp",
        )

        await update_mcp_server(mock_prisma, data, "test-user")

        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert data_dict["transport"] in (MCPTransport.http, "http")
        assert data_dict["url"] == "https://example.com/mcp"

    @pytest.mark.asyncio
    async def test_partial_update_does_not_reset_allow_all_keys(self):
        """A row with allow_all_keys=True must NOT be reset by an omitting PUT."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            description="new description",
        )

        await update_mcp_server(mock_prisma, data, "test-user")
        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert "allow_all_keys" not in data_dict

    @pytest.mark.asyncio
    async def test_partial_update_does_not_clear_mcp_access_groups(self):
        """A row with mcp_access_groups=["x"] must NOT be cleared by an omitting PUT."""
        from litellm.proxy._experimental.mcp_server.db import update_mcp_server
        from litellm.proxy._types import UpdateMCPServerRequest

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_mcpservertable.update = AsyncMock(
            return_value=MagicMock()
        )

        data = UpdateMCPServerRequest(
            server_id="test-server",
            url="https://example.com/mcp",
        )

        await update_mcp_server(mock_prisma, data, "test-user")
        data_dict = mock_prisma.db.litellm_mcpservertable.update.call_args[1]["data"]
        assert "mcp_access_groups" not in data_dict

    def test_create_path_still_writes_defaults(self):
        """Create path (NewMCPServerRequest) must still include defaults so
        non-nullable columns are populated on insert. Use exclude_unset=False."""
        from litellm.proxy._experimental.mcp_server.db import (
            _prepare_mcp_server_data,
        )
        from litellm.proxy._types import NewMCPServerRequest

        data = NewMCPServerRequest(
            server_id="srv-1",
            server_name="srv",
            url="https://example.com/mcp",
        )

        prepared = _prepare_mcp_server_data(data, exclude_unset=False)

        assert "transport" in prepared
        assert "is_byok" in prepared
        assert prepared["is_byok"] is False
        assert "alias" in prepared

    def test_update_explicit_null_clears_allowed_tools(self):
        """LIT-3424 bonus: caller sending allowed_tools=None must clear the
        field. Previously exclude_none=True dropped it so the row stuck."""
        from litellm.proxy._experimental.mcp_server.db import (
            _prepare_mcp_server_data,
        )
        from litellm.proxy._types import UpdateMCPServerRequest

        data = UpdateMCPServerRequest(
            server_id="srv-1",
            allowed_tools=None,
        )

        prepared = _prepare_mcp_server_data(data, exclude_unset=True)

        assert "allowed_tools" in prepared
        assert prepared["allowed_tools"] is None

    def test_validate_and_normalize_does_not_force_alias_when_omitted(self):
        """validate_and_normalize_mcp_server_payload must NOT mark alias as
        set on the model when the caller did not provide one — otherwise our
        exclude_unset payload would include alias=None and clear the row
        (LIT-3423 secondary fix)."""
        from litellm.proxy._experimental.mcp_server.utils import (
            validate_and_normalize_mcp_server_payload,
        )
        from litellm.proxy._types import UpdateMCPServerRequest

        payload = UpdateMCPServerRequest(
            server_id="srv-1",
            description="d",
        )
        assert "alias" not in payload.model_fields_set
        validate_and_normalize_mcp_server_payload(payload)
        assert "alias" not in payload.model_fields_set, (
            "validate_and_normalize_mcp_server_payload should not assign "
            "alias when caller omitted both alias and server_name (LIT-3423)."
        )


    def test_validate_and_normalize_assigns_alias_when_normalization_changes_it(self):
        """When the caller provides an alias with spaces, validate_and_normalize
        must reassign payload.alias to the normalized form (covers the True
        branch of the new equality guard added for LIT-3423)."""
        from litellm.proxy._experimental.mcp_server.utils import (
            validate_and_normalize_mcp_server_payload,
        )
        from litellm.proxy._types import NewMCPServerRequest

        payload = NewMCPServerRequest(
            server_name="srv",
            alias="My Server",  # space normalizes to underscore
            url="https://example.com/mcp",
        )
        validate_and_normalize_mcp_server_payload(payload)
        assert payload.alias == "My_Server"
        # alias should be marked as set (we DID change it).
        assert "alias" in payload.model_fields_set

