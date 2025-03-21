from typing import List, Literal, Union

from mcp import ClientSession
from mcp.types import Tool as MCPTool

from litellm.types.llms.openai import Tool


def transform_mcp_tool_to_openai_tool(tool: MCPTool) -> Tool:
    """Convert an MCP tool to an OpenAI tool."""
    raise NotImplementedError("Not implemented")


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> Union[List[MCPTool], List[Tool]]:
    """
    Load all available MCP tools

    Args:
        session: The MCP session to use
        format: The format to convert the tools to
    By default, the tools are returned in MCP format.

    If format is set to "openai", the tools are converted to OpenAI tools.
    """
    tools = await session.list_tools()
    if format == "openai":
        return [transform_mcp_tool_to_openai_tool(tool) for tool in tools.tools]
    return tools.tools
