import asyncio
import os
import logging
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession
from mcp.types import ElicitResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    if os.environ.get("RUN_LIVE_MCP_TEST") != "1":
        logger.info("Skipping live integration test. Set RUN_LIVE_MCP_TEST=1 to run.")
        return
    logger.info("Connecting to LiteLLM Proxy via SSE...")

    async def my_elicitation_callback(context, params):
        logger.info("\n[CLIENT] Received elicitation request from upstream!")

        # We will simulate the user filling out the form
        user_response = {
            "topic": "a time-traveling developer",
            "adjective": "suspenseful",
        }

        logger.info(f"[CLIENT] User is filling the form with: {user_response}")

        return ElicitResult(action="accept", content=user_response)

    async with sse_client("http://localhost:4000/mcp/sse", headers={"Authorization": "Bearer sk-1234"}) as (
        read_stream,
        write_stream,
    ):
        logger.info("SSE connection established.")
        async with ClientSession(read_stream, write_stream, elicitation_callback=my_elicitation_callback) as session:
            await session.initialize()
            logger.info("Initialized!")

            logger.info("\n--- Testing Complex Pipeline (Elicitation + Sampling) ---")
            logger.info("Calling 'test_server-test_complex_pipeline'...")
            try:
                result = await session.call_tool("test_server-test_complex_pipeline", arguments={})
                logger.info("\nFINAL TOOL RESULT:")
                logger.info("==================")
                logger.info(result.content[0].text)
            except Exception as e:
                logger.info(f"Error calling test_complex_pipeline: {e}")


if __name__ == "__main__":
    asyncio.run(main())
