import json
from typing import Final, Literal

import anyio
from mcp import ClientSession
from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import PaginatedRequestParams
from mcp.types import Tool as MCPTool
from openai.types.chat import ChatCompletionToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.shared_params.function_definition import FunctionDefinition

from litellm._logging import verbose_logger
from litellm.constants import (
    MCP_CLIENT_TIMEOUT,
    MCP_TOOL_LISTING_MAX_PAGES,
    MCP_TOOL_LISTING_TIMEOUT,
)
from litellm.types.llms.anthropic import AnthropicMessagesTool
from litellm.types.utils import ChatCompletionMessageToolCall


########################################################
# List MCP Tool functions
########################################################
def transform_mcp_tool_to_openai_tool(mcp_tool: MCPTool) -> ChatCompletionToolParam:
    """Convert an MCP tool to an OpenAI tool."""
    normalized_parameters: Final = _normalize_mcp_input_schema(mcp_tool.inputSchema)

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
        return {"type": "object", "properties": {}, "additionalProperties": False}

    # Make a copy to avoid modifying the original
    normalized_schema: Final = dict(input_schema)

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


def transform_mcp_tool_to_openai_responses_api_tool(
    mcp_tool: MCPTool,
) -> FunctionToolParam:
    """Convert an MCP tool to an OpenAI Responses API tool."""
    normalized_parameters: Final = _normalize_mcp_input_schema(mcp_tool.inputSchema)

    return FunctionToolParam(
        name=mcp_tool.name,
        parameters=normalized_parameters,
        strict=False,
        type="function",
        description=mcp_tool.description or "",
    )


def transform_mcp_tool_to_anthropic_tool(mcp_tool: MCPTool) -> AnthropicMessagesTool:
    """Convert an MCP tool to an Anthropic Messages API tool."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        sanitize_input_schema_for_anthropic,
    )

    return AnthropicMessagesTool(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        input_schema=sanitize_input_schema_for_anthropic(mcp_tool.inputSchema),
        type="custom",
    )


async def list_tools_with_pagination(session: ClientSession) -> list[MCPTool]:  # mutable-ok: list return contract
    """Collect tools from every tools/list page by following nextCursor.

    Stops and returns the tools collected so far when the upstream repeats a
    cursor, the page cap is reached, or the whole-walk deadline expires, so a
    buggy or slow upstream yields a partial catalog instead of an error.
    """
    tools: Final[list[MCPTool]] = []  # mutable-ok: accumulates each page's tools
    seen_cursors: Final[set[str]] = set()  # mutable-ok: guards against cursor loops
    cursor: str | None = None  # rebind-ok: advances to each page's nextCursor
    # The per-request session read timeout restarts on every page, so a multi-page
    # walk needs its own overall deadline. max() keeps the pre-pagination guarantee
    # that a single page slower than the listing timeout but within the client
    # timeout still succeeds.
    listing_deadline: Final = max(MCP_CLIENT_TIMEOUT, MCP_TOOL_LISTING_TIMEOUT)

    with anyio.move_on_after(listing_deadline):
        for _ in range(MCP_TOOL_LISTING_MAX_PAGES):
            result = (
                await session.list_tools()
                if cursor is None
                else await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
            )
            tools.extend(result.tools)

            next_cursor = getattr(result, "nextCursor", None)
            if not isinstance(next_cursor, str) or not next_cursor:
                return tools
            if next_cursor in seen_cursors:
                verbose_logger.warning(
                    "MCP server repeated a tools/list cursor while listing tools; returning %s tools collected so far",
                    len(tools),
                )
                return tools
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        verbose_logger.warning(
            "MCP server tools/list pagination exceeded the maximum of %s pages; returning %s tools collected so far",
            MCP_TOOL_LISTING_MAX_PAGES,
            len(tools),
        )
        return tools

    verbose_logger.warning(
        "MCP server tools/list pagination exceeded the %s second listing deadline; returning %s tools collected so far",
        listing_deadline,
        len(tools),
    )
    return tools


async def load_mcp_tools(
    session: ClientSession, format: Literal["mcp", "openai"] = "mcp"
) -> list[MCPTool] | list[ChatCompletionToolParam]:
    """
    Load all available MCP tools

    Args:
        session: The MCP session to use
        format: The format to convert the tools to
    By default, the tools are returned in MCP format.

    If format is set to "openai", the tools are converted to OpenAI API compatible tools.
    """
    tools: Final = await list_tools_with_pagination(session)
    if format == "openai":
        return [  # mutable-ok: public API returns a list
            transform_mcp_tool_to_openai_tool(mcp_tool=tool) for tool in tools
        ]
    return tools


########################################################
# Call MCP Tool functions
########################################################


async def call_mcp_tool(
    session: ClientSession,
    call_tool_request_params: MCPCallToolRequestParams,
) -> MCPCallToolResult:
    """Call an MCP tool."""
    tool_result: Final = await session.call_tool(
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
    openai_tool: ChatCompletionMessageToolCall | dict,
) -> MCPCallToolRequestParams:
    """Convert an OpenAI ChatCompletionMessageToolCall to an MCP CallToolRequestParams."""
    function: Final = openai_tool["function"]
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
    mcp_tool_call_request_params: Final = transform_openai_tool_call_request_to_mcp_tool_call_request(
        openai_tool=openai_tool,
    )
    return await call_mcp_tool(
        session=session,
        call_tool_request_params=mcp_tool_call_request_params,
    )
