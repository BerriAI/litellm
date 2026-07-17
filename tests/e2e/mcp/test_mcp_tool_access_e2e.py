"""Live e2e: the MCP gateway's ingress key gate under both documented headers.

Covers mcp.list_tools.{api_key,bearer}.rejects_unknown_key against the
deterministic mcp-stub compose service (tests/e2e/mcp/stub/): a key the proxy
does not recognize must be refused with an explicit 401 at session
establishment, whichever documented header carries it. The list-and-call
happy path itself lives in test_mcp_oauth_interactive_e2e.py, where it runs
against a real OAuth server; these tests keep the plain ingress gate pinned
with a deliberately minimal setup.

Each test body spells out every step a human QA run would take, in order:
create the server over the management API, mint a virtual key over
/key/generate, build the exact wire header, settle the server with the real
key (so the later refusal can only be the ingress auth, never record
propagation), then present an unknown key over the same header and require
the explicit 401. Teardown is the only thing delegated (resources.defer),
because it must run even when an assertion fails.
"""

from __future__ import annotations

import pytest

from e2e_config import MCP_STUB_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient, McpDenied
from models import KeyGenerateBody, McpServerCreateBody

pytestmark = pytest.mark.e2e

class TestMcpToolAccess:
    """A key the proxy does not recognize is turned away at the door, under
    either documented auth header."""

    @pytest.mark.covers("mcp.list_tools.api_key.rejects_unknown_key")
    def test_unrecognized_key_in_x_litellm_api_key_header_is_turned_away(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        """Settle the server with a real key first, then present an unknown
        key over the same header: the refusal must be an explicit 401 from
        session establishment, and can only be the ingress auth because the
        server was just proven servable."""
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_URL
        assert stored.allow_all_keys is True

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))
        _ = client.poll_tool_names(alias, {"x-litellm-api-key": f"Bearer {key}"})

        denied = client.list_tools_once(alias, {"x-litellm-api-key": "Bearer sk-not-a-real-key"})
        assert isinstance(denied, McpDenied), f"unrecognized key was served tools: {denied}"
        assert denied.status_code == 401, f"expected 401 for an unrecognized key, got {denied}"

    @pytest.mark.covers("mcp.list_tools.bearer.rejects_unknown_key")
    def test_unrecognized_key_in_authorization_bearer_header_is_turned_away(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        """The Authorization form of the same rejection: an unknown key sent
        as `Authorization: Bearer ...` must be refused identically."""
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_URL
        assert stored.allow_all_keys is True

        key = client.gateway.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.gateway.delete_key(key))
        _ = client.poll_tool_names(alias, {"Authorization": f"Bearer {key}"})

        denied = client.list_tools_once(alias, {"Authorization": "Bearer sk-not-a-real-key"})
        assert isinstance(denied, McpDenied), f"unrecognized key was served tools: {denied}"
        assert denied.status_code == 401, f"expected 401 for an unrecognized key, got {denied}"
