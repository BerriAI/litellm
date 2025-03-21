"""
LiteLLM MCP Server Routes
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from litellm.proxy._experimental.mcp_server.tool_registry import (
    global_mcp_tool_registry,
)
from litellm.types.mcp_server.tool_registry import *

router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
)


@router.get("/")
async def root():
    return {"message": "MCP Server is running"}


@router.get("/tools/list", response_model=ListToolsResponse)
async def list_tools(cursor: Optional[str] = None):
    """
    List all available tools
    """
    tools = []
    for tool in global_mcp_tool_registry.list_tools():
        tools.append(
            ToolSchema(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
        )

    return ListToolsResponse(tools=tools)


@router.post("/tools/call", response_model=CallToolResponse)
async def call_tool(request: CallToolRequest):
    """
    Call a specific tool with the provided arguments
    """
    if request.method != "tools/call":
        raise HTTPException(status_code=400, detail="Invalid method")

    if "name" not in request.params:
        raise HTTPException(status_code=400, detail="Tool name is required")

    tool_name = request.params["name"]
    arguments = request.params.get("arguments", {})

    tool = global_mcp_tool_registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        result = tool.handler(**arguments)

        # Convert result to text content
        if isinstance(result, str):
            content = [ContentItem(type="text", text=result)]
        elif isinstance(result, dict):
            content = [ContentItem(type="text", text=str(result))]
        else:
            content = [ContentItem(type="text", text=str(result))]

        return CallToolResponse(content=content)
    except Exception as e:
        return CallToolResponse(
            content=[ContentItem(type="text", text=f"Error: {str(e)}")], isError=True
        )
