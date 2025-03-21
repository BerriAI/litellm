# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI

import pytest


@pytest.mark.asyncio
async def test_mcp_agent():
    server_params = StdioServerParameters(
        command="python3",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["./mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            print("MCP TOOLS: ", tools)

            # Create and run the agent
            print(os.getenv("OPENAI_API_KEY"))
            model = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

            # Add assertions to verify the response
            assert isinstance(agent_response, dict)
            print(agent_response)
