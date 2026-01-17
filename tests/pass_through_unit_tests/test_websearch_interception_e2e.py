"""
Real E2E Test for WebSearch Interception

Makes actual calls to test WebSearch interception with Perplexity.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.websearch_interception import (
    WebSearchInterceptionLogger,
)
from litellm.anthropic_interface import messages
from litellm.types.utils import LlmProviders
import asyncio


async def test_websearch_interception_real_call():
    """
    Real e2e test with actual API calls.
    Tests that WebSearch tool calls are intercepted and executed.
    """
    litellm._turn_on_debug()

    print("\n" + "="*80)
    print("E2E TEST: WebSearch Interception with Perplexity")
    print("="*80)

    # Initialize real router with search_tools configuration
    import litellm.proxy.proxy_server as proxy_server
    from litellm import Router

    # Create real router with search_tools
    router = Router(
        search_tools=[
            {
                "search_tool_name": "my-perplexity-search",
                "litellm_params": {
                    "search_provider": "perplexity"
                }
            }
        ]
    )
    proxy_server.llm_router = router

    print("\n✅ Initialized router with search_tools:")
    print(f"   - search_tool_name: my-perplexity-search")
    print(f"   - search_provider: perplexity")

    # Enable WebSearch interception for bedrock
    websearch_logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK],
        search_tool_name="my-perplexity-search",  # Will look up this tool in router
    )
    litellm.callbacks = [websearch_logger]
    litellm.set_verbose = True  # Enable verbose logging

    print("\n✅ Configured WebSearch interception for Bedrock")
    print("✅ Will use search tool from router")

    try:
        # Make request with WebSearch tool
        print("\n📞 Making litellm.messages.acreate() call...")
        print(f"   Model: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        print(f"   Query: 'What is LiteLLM?'")
        print(f"   Tools: WebSearch")

        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "What is LiteLLM? Give me a brief overview."}],
            tools=[
                {
                    "name": "WebSearch",
                    "description": "Search the web for information",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ],
            max_tokens=1024,
        )

        print("\n✅ Received response!")

        # Handle both dict and object responses
        if isinstance(response, dict):
            response_id = response.get("id")
            response_model = response.get("model")
            response_stop_reason = response.get("stop_reason")
            response_content = response.get("content", [])
        else:
            response_id = response.id
            response_model = response.model
            response_stop_reason = response.stop_reason
            response_content = response.content

        print(f"\n📄 Response ID: {response_id}")
        print(f"📄 Model: {response_model}")
        print(f"📄 Stop Reason: {response_stop_reason}")
        print(f"📄 Content blocks: {len(response_content)}")

        # Debug: Print all content block types
        for i, block in enumerate(response_content):
            block_type = block.get("type") if isinstance(block, dict) else block.type
            print(f"   Block {i}: type={block_type}")
            if block_type == "tool_use":
                block_name = block.get("name") if isinstance(block, dict) else block.name
                print(f"            name={block_name}")

        # Validate response
        assert response is not None, "Response should not be None"
        assert response_content is not None, "Response should have content"
        assert len(response_content) > 0, "Response should have at least one content block"

        # Check if response contains tool_use (means interception didn't work)
        has_tool_use = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "tool_use"
            for block in response_content
        )

        # Check if we got a text response
        has_text = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "text"
            for block in response_content
        )

        if has_tool_use:
            # Got tool_use response - this means interception didn't work
            print("\n❌ INTERCEPTION DID NOT WORK")
            print(f"❌ Stop reason: {response_stop_reason}")
            print("❌ Response contains tool_use blocks")
            print("❌ Expected: Final answer with search results")
            print("❌ Got: Tool use request to client")

            if has_text:
                text_block = next(
                    block for block in response_content
                    if (block.get("type") if isinstance(block, dict) else block.type) == "text"
                )
                text_content = text_block.get("text") if isinstance(text_block, dict) else text_block.text
                print(f"\n📝 Text from response: {text_content}")

            return False

        elif has_text and response_stop_reason != "tool_use":
            # Got final text response without tool_use - interception worked!
            text_block = next(
                block for block in response_content
                if (block.get("type") if isinstance(block, dict) else block.type) == "text"
            )
            text_content = text_block.get("text") if isinstance(text_block, dict) else text_block.text

            print(f"\n📝 Response Text:")
            print(f"   {text_content}")

            # Check if response mentions LiteLLM
            if "litellm" in text_content.lower():
                print("\n" + "="*80)
                print("✅ TEST PASSED!")
                print("="*80)
                print("✅ User made ONE litellm.messages.acreate() call")
                print("✅ Got back final answer (not tool_use)")
                print("✅ Agentic loop executed transparently")
                print("✅ WebSearch interception working!")
                print("="*80)
                return True
            else:
                print("\n⚠️  Got text response but doesn't mention LiteLLM")
                return False
        else:
            print("\n❌ Unexpected response format")
            print(f"   has_tool_use: {has_tool_use}")
            print(f"   has_text: {has_text}")
            print(f"   stop_reason: {response_stop_reason}")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_websearch_interception_real_call())
    sys.exit(0 if result else 1)
