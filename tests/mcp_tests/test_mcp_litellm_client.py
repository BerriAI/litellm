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
from litellm import experimental_mcp_client
import litellm
import pytest
import json


@pytest.mark.asyncio
async def test_mcp_agent():
    local_server_path = "./mcp_server.py"
    ci_cd_server_path = "tests/mcp_tests/mcp_server.py"
    server_params = StdioServerParameters(
        command="python3",
        # Make sure to update to the full absolute path to your math_server.py file
        args=[ci_cd_server_path],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await experimental_mcp_client.load_mcp_tools(
                session=session, format="openai"
            )
            print("MCP TOOLS: ", tools)

            # Create and run the agent
            messages = [{"role": "user", "content": "what's (3 + 5)"}]
            llm_response = await litellm.acompletion(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                messages=messages,
                tools=tools,
                tool_choice="required",
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
            openai_tool = llm_response["choices"][0]["message"]["tool_calls"][0]

            # Call the tool using MCP client
            call_result = await experimental_mcp_client.call_openai_tool(
                session=session,
                openai_tool=openai_tool,
            )
            print("CALL RESULT: ", call_result)

            # send the tool result to the LLM
            messages.append(llm_response["choices"][0]["message"])
            messages.append(
                {
                    "role": "tool",
                    "content": str(call_result.content[0].text),
                    "tool_call_id": openai_tool["id"],
                }
            )
            print("final messages: ", messages)
            llm_response = await litellm.acompletion(
                model="gpt-4o",
                api_key=os.getenv("OPENAI_API_KEY"),
                messages=messages,
                tools=tools,
            )
            print(
                "FINAL LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str)
            )
