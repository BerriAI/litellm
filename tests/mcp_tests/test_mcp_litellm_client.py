# Create server parameters for stdio connection
import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from litellm.mcp_client.tools import load_mcp_tools
import litellm
import pytest
import json


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
            tools = await load_mcp_tools(session=session, format="openai")
            print("MCP TOOLS: ", tools)

            # Create and run the agent
            print(os.getenv("OPENAI_API_KEY"))
            llm_response = await litellm.acompletion(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                messages=[{"role": "user", "content": "what's (3 + 5) x 12?"}],
                tools=tools,
            )
            print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))

            # Add assertions to verify the response
            assert llm_response["choices"][0]["message"]["tool_calls"] is not None
            assert (
                llm_response["choices"][0]["message"]["tool_calls"][0]["function"][
                    "name"
                ]
                == "add"
            )
