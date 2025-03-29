import asyncio
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from mcp import ClientSession
from mcp.client.sse import sse_client
from litellm.experimental_mcp_client.tools import (
    transform_mcp_tool_to_openai_tool,
    transform_openai_tool_call_request_to_mcp_tool_call_request,
)


async def main():
    # Initialize clients
    client = AsyncOpenAI(api_key="sk-1234", base_url="http://localhost:4000")

    # Connect to MCP
    async with sse_client("http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            print("List of MCP tools for MCP server:", mcp_tools.tools)

            # Create message
            messages = [
                ChatCompletionUserMessageParam(
                    content="Send an email about LiteLLM supporting MCP", role="user"
                )
            ]

            # Request with tools
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=[
                    transform_mcp_tool_to_openai_tool(tool) for tool in mcp_tools.tools
                ],
                tool_choice="auto",
            )

            # Handle tool call
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call:
                    # Convert format
                    mcp_call = (
                        transform_openai_tool_call_request_to_mcp_tool_call_request(
                            openai_tool=tool_call.model_dump()
                        )
                    )

                    # Execute tool
                    result = await session.call_tool(
                        name=mcp_call.name, arguments=mcp_call.arguments
                    )

                    print("Result:", result)


# Run it
asyncio.run(main())
