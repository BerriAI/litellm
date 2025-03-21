# Create server parameters for stdio connection
import asyncio
import os

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    model = ChatOpenAI(model="gpt-4o", api_key="sk-12")

    async with sse_client(url="http://localhost:8000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            print(tools)

            # # Create and run the agent
            # agent = create_react_agent(model, tools)
            # agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
