# Create server parameters for stdio connection
import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client(url="http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            print("Initializing session")
            await session.initialize()
            print("Session initialized")

            # Get tools
            print("Loading tools")
            tools = await session.list_tools()
            print("Tools loaded")
            print(tools)

            if tools.tools:
                first = tools.tools[0]
                print(f"Calling tool {first.name}")
                result = await session.call_tool(first.name, {})
                print(result)


# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
