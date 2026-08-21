"""Live e2e: the MCP REST endpoints return 401 for an invalid key, not a
flattened 500.

Before PR #31011 the MCP protocol path flattened auth errors to 500; the REST
path shares the same user_api_key_auth dependency as /chat/completions, so a
401 here proves the gateway's auth error mapping is intact. Budget enforcement
(429) is the same dependency as chat and is covered by the quota_management
suite.
"""

from __future__ import annotations

import pytest

from e2e_http import UnauthorizedError
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e

GARBAGE_KEY = "sk-deadbeef-not-a-real-key"


class TestMcpAuthStatusCodes:
    @pytest.mark.covers("mcp.auth.api_key.returns_401_not_500")
    def test_invalid_key_returns_401_not_500(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        result = client.list_tools(GARBAGE_KEY)
        assert isinstance(result, UnauthorizedError), (
            f"invalid key on /mcp-rest/tools/list must return 401, not 500; "
            f"got: {result}"
        )
