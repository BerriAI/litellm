import json
from typing import List, Literal, Union

from mcp import ClientSession
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from litellm.types.utils import ChatCompletionMessageToolCall


########################################################
# List MCP Tool functions
########################################################
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


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> Union[List[MCPTool], List[ChatCompletionToolParam]]:
    """
    Load all available MCP tools

    Args:
        session: The MCP session to use
        format: The format to convert the tools to
    By default, the tools are returned in MCP format.

    If format is set to "openai", the tools are converted to OpenAI API compatible tools.
    """
    tools = await session.list_tools()
    if format == "openai":
        return [
            transform_mcp_tool_to_openai_tool(mcp_tool=tool) for tool in tools.tools
        ]
    return tools.tools


########################################################
# Call MCP Tool functions
########################################################


async def call_mcp_tool(
    session: ClientSession,
    call_tool_request_params: MCPCallToolRequestParams,
) -> MCPCallToolResult:
    """Call an MCP tool."""
    tool_result = await session.call_tool(
        name=call_tool_request_params.name,
        arguments=call_tool_request_params.arguments,
    )
    return tool_result


def _get_function_arguments(function: FunctionDefinition) -> dict:
    """Helper to safely get and parse function arguments."""
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def _transform_openai_tool_call_to_mcp_tool_call_request(
    openai_tool: ChatCompletionMessageToolCall,
) -> MCPCallToolRequestParams:
    """Convert an OpenAI ChatCompletionMessageToolCall to an MCP CallToolRequestParams."""
    function = openai_tool["function"]
    return MCPCallToolRequestParams(
        name=function["name"],
        arguments=_get_function_arguments(function),
    )


async def call_openai_tool(
    session: ClientSession,
    openai_tool: ChatCompletionMessageToolCall,
) -> MCPCallToolResult:
    """
    Call an OpenAI tool using MCP client.

    Args:
        session: The MCP session to use
        openai_tool: The OpenAI tool to call. You can get this from the `choices[0].message.tool_calls[0]` of the response from the OpenAI API.
    Returns:
        The result of the MCP tool call.
    """
    mcp_tool_call_request_params = _transform_openai_tool_call_to_mcp_tool_call_request(
        openai_tool=openai_tool,
    )
    return await call_mcp_tool(
        session=session,
        call_tool_request_params=mcp_tool_call_request_params,
    )
