# Create server parameters for stdio connection
import asyncio
import os

import pytest
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm.experimental_mcp_client.tools import (
    transform_mcp_tool_to_openai_tool,
    _transform_openai_tool_call_to_mcp_tool_call_request,
)
import json


@pytest.mark.asyncio
async def test_mcp_routes():
    model = ChatOpenAI(
        model="gpt-4o", api_key="sk-1234", base_url="http://localhost:4000"
    )

    async with sse_client(url="http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            print("Initializing session")
            await session.initialize()
            print("Session initialized")

            # Get tools
            print("Loading tools")
            tools = await load_mcp_tools(session)
            print("Tools loaded")
            print(tools)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "Send an "})
            print(agent_response)


@pytest.mark.asyncio
async def test_mcp_routes_with_vertex_ai():
    # Create and run the agent
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key="sk-1234", base_url="http://localhost:4000")
    async with sse_client(url="http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            MCP_TOOLS = await session.list_tools()

            print("MCP TOOLS from litellm proxy: ", MCP_TOOLS)
            messages = [
                {
                    "role": "user",
                    "content": "send an email about litellm supporting MCP and send it to krrish@berri.ai",
                }
            ]
            llm_response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[
                    transform_mcp_tool_to_openai_tool(tool) for tool in MCP_TOOLS.tools
                ],
                tool_choice="required",
            )
            print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))

            # Add assertions to verify the response
            openai_tool = llm_response.choices[0].message.tool_calls[0]

            # Call the tool using MCP client
            mcp_tool_call_request = (
                _transform_openai_tool_call_to_mcp_tool_call_request(
                    openai_tool.model_dump()
                )
            )
            call_result = await session.call_tool(
                name=mcp_tool_call_request.name,
                arguments=mcp_tool_call_request.arguments,
            )
            print("CALL RESULT: ", call_result)
    pass
