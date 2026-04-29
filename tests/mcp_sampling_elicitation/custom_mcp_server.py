import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    CallToolResult,
    TextContent,
    SamplingMessage,
)

import logging

logging.basicConfig(filename="custom_server.log", level=logging.DEBUG, force=True)

server = Server("custom-test-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="test_complex_pipeline",
            description="Tests the full pipeline: Asks user for a topic, then asks AI to write a story about it.",
            inputSchema={"type": "object", "properties": {}},
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    session = server.request_context.session

    if name == "test_complex_pipeline":
        # 1. Elicitation: Ask the user for inputs
        elicit_result = await session.elicit_form(
            message="Please provide details for the story",
            requestedSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What should the story be about?",
                    },
                    "adjective": {
                        "type": "string",
                        "description": "What is the tone of the story?",
                    },
                },
                "required": ["topic", "adjective"],
            },
        )

        # Parse the user's response
        topic = "a random thing"
        adjective = "weird"
        content = getattr(elicit_result, "content", None)
        if isinstance(content, dict):
            topic = content.get("topic", topic)
            adjective = content.get("adjective", adjective)
        elif hasattr(content, "topic"):
            topic = content.topic
            adjective = content.adjective

        logging.info(f"Got elicitation: topic={topic}, adjective={adjective}")

        # 2. Sampling: Ask the AI to write the story based on user input
        sample_result = await session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Write a very short, 3-sentence {adjective} story about {topic}.",
                    ),
                )
            ],
            max_tokens=150,
        )

        ai_story = sample_result.content.text

        # 3. Return final result
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Pipeline Complete!\nUser chose: a {adjective} story about '{topic}'.\n\nAI generated story:\n{ai_story}",
                )
            ]
        )

    return CallToolResult(
        content=[TextContent(type="text", text="Tool not found")], isError=True
    )


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
