import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import ElicitResult


async def main():
    print("Connecting to LiteLLM Proxy via SSE...")

    async def my_elicitation_callback(context, params):
        print(f"\n[CLIENT] Received elicitation request from upstream!")

        # We will simulate the user filling out the form
        user_response = {
            "topic": "a time-traveling developer",
            "adjective": "suspenseful",
        }

        print(f"[CLIENT] User is filling the form with: {user_response}")

        return ElicitResult(action="accept", content=user_response)

    async with sse_client(
        "http://localhost:4000/mcp/sse", headers={"Authorization": "Bearer sk-1234"}
    ) as (read_stream, write_stream):
        print("SSE connection established.")
        async with ClientSession(
            read_stream, write_stream, elicitation_callback=my_elicitation_callback
        ) as session:
            await session.initialize()
            print("Initialized!")

            print("\n--- Testing Complex Pipeline (Elicitation + Sampling) ---")
            print("Calling 'test_server-test_complex_pipeline'...")
            try:
                result = await session.call_tool(
                    "test_server-test_complex_pipeline", arguments={}
                )
                print("\nFINAL TOOL RESULT:")
                print("==================")
                print(result.content[0].text)
            except Exception as e:
                print(f"Error calling test_complex_pipeline: {e}")


if __name__ == "__main__":
    asyncio.run(main())
