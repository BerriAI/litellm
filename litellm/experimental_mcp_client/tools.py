import json
from typing import Dict, List, Literal, Union

from mcp import ClientSession
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import Tool as MCPTool
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from litellm.types.utils import ChatCompletionMessageToolCall


########################################################
# List MCP Tool functions
########################################################
def transform_mcp_tool_to_openai_tool(mcp_tool: MCPTool) -> ChatCompletionToolParam:
    """Convert an MCP tool to an OpenAI tool."""
    normalized_parameters = _normalize_mcp_input_schema(mcp_tool.inputSchema)
    
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            parameters=normalized_parameters,
            strict=False,
        ),
    )


def _normalize_mcp_input_schema(input_schema: dict) -> dict:
    """
    Normalize MCP input schema to ensure it's valid for OpenAI function calling.
    
    OpenAI requires that function parameters have:
    - type: 'object'
    - properties: dict (can be empty)
    - additionalProperties: false (recommended)
    """
    if not input_schema:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        }
    
    # Make a copy to avoid modifying the original
    normalized_schema = dict(input_schema)
    
    # Ensure type is 'object'
    if "type" not in normalized_schema:
        normalized_schema["type"] = "object"
    
    # Ensure properties exists (can be empty)
    if "properties" not in normalized_schema:
        normalized_schema["properties"] = {}
    
    # Add additionalProperties if not present (recommended by OpenAI)
    if "additionalProperties" not in normalized_schema:
        normalized_schema["additionalProperties"] = False
    
    return normalized_schema


def transform_mcp_tool_to_openai_responses_api_tool(mcp_tool: MCPTool) -> FunctionToolParam:
    """Convert an MCP tool to an OpenAI Responses API tool."""
    normalized_parameters = _normalize_mcp_input_schema(mcp_tool.inputSchema)
    
    return FunctionToolParam(
        name=mcp_tool.name,
        parameters=normalized_parameters,
        strict=False,
        type="function",
        description=mcp_tool.description or "",
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


def transform_openai_tool_call_request_to_mcp_tool_call_request(
    openai_tool: Union[ChatCompletionMessageToolCall, Dict],
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
    mcp_tool_call_request_params = (
        transform_openai_tool_call_request_to_mcp_tool_call_request(
            openai_tool=openai_tool,
        )
    )
    return await call_mcp_tool(
        session=session,
        call_tool_request_params=mcp_tool_call_request_params,
    )
