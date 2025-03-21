import json
from typing import List, Literal, Union

from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.types import Tool as MCPTool
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition


def transform_mcp_tool_to_openai_tool(mcp_tool: MCPTool) -> ChatCompletionToolParam:
    """Convert an MCP tool to an OpenAI tool."""
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=mcp_tool.inputSchema,
            strict=False,
        ),
    )


def _get_function_arguments(function: FunctionDefinition) -> dict:
    """Helper to safely get and parse function arguments."""
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def transform_openai_tool_to_mcp_tool(openai_tool: ChatCompletionToolParam) -> MCPTool:
    """Convert an OpenAI tool to an MCP tool."""
    function = openai_tool["function"]
    return MCPTool(
        name=function["name"],
        description=function.get("description", ""),
        inputSchema=_get_function_arguments(function),
    )


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> Union[List[MCPTool], List[ChatCompletionToolParam]]:
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
        return [
            transform_mcp_tool_to_openai_tool(mcp_tool=tool) for tool in tools.tools
        ]
    return tools.tools


async def call_mcp_tool(
    session: ClientSession,
    name: str,
    arguments: dict,
) -> CallToolResult:
    """Call an MCP tool."""
    tool_result = await session.call_tool(
        name=name,
        arguments=arguments,
    )
    return tool_result


async def call_openai_tool(
    session: ClientSession,
    openai_tool: ChatCompletionToolParam,
) -> CallToolResult:
    """Call an OpenAI tool using MCP client."""
    mcp_tool = transform_openai_tool_to_mcp_tool(
        openai_tool=openai_tool,
    )
    return await call_mcp_tool(
        session=session,
        name=mcp_tool.name,
        arguments=mcp_tool.inputSchema,
    )
