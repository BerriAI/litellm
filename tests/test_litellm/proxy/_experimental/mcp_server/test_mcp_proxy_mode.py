import pytest
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from litellm.proxy._experimental.mcp_server import server
from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode
from litellm.proxy._types import UserAPIKeyAuth

AUTH = UserAPIKeyAuth(api_key="key")


@pytest.fixture
def proxy_mode():
    token = _mcp_proxy_mode.set(True)
    try:
        yield
    finally:
        _mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_call_rejects_non_proxy_tool_names() -> None:
    result = await server._dispatch_virtual_mcp_tool(
        name="math_stdio-add", arguments={"a": 1, "b": 2}, user_api_key_auth=AUTH, client_ip=None
    )

    assert result is not None
    assert result.isError is True
    assert "unavailable on /mcp/proxy" in result.content[0].text


@pytest.mark.asyncio
@pytest.mark.usefixtures("proxy_mode")
async def test_proxy_rejects_non_tool_protocol_operations() -> None:
    options = server.server.create_initialization_options()
    assert options.capabilities.prompts is None
    assert options.capabilities.resources is None
    assert options.capabilities.tools is not None

    with pytest.raises(McpError):
        await server.list_prompts()
    with pytest.raises(McpError):
        await server.get_prompt("prompt", {})
    with pytest.raises(McpError):
        await server.list_resources()
    with pytest.raises(McpError):
        await server.list_resource_templates()
    with pytest.raises(McpError):
        await server.read_resource(AnyUrl("https://example.com/resource"))
