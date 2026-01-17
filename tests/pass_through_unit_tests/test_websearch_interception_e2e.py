"""
Real E2E Tests for WebSearch Interception

Makes actual calls to test WebSearch interception with Perplexity.
Tests both streaming and non-streaming requests.
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


async def test_websearch_interception_non_streaming():
    """
    Test WebSearch interception with non-streaming request.
    Validates that agentic loop executes transparently.
    """
    litellm._turn_on_debug()

    print("\n" + "="*80)
    print("E2E TEST 1: WebSearch Interception (Non-Streaming)")
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
        search_tool_name="my-perplexity-search",
    )
    litellm.callbacks = [websearch_logger]
    litellm.set_verbose = True

    print("\n✅ Configured WebSearch interception for Bedrock")
    print("✅ Will use search tool from router")

    try:
        # Make request with WebSearch tool (non-streaming)
        print("\n📞 Making litellm.messages.acreate() call...")
        print(f"   Model: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        print(f"   Query: 'What is LiteLLM?'")
        print(f"   Tools: WebSearch")
        print(f"   Stream: False")

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
            stream=False,
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
            print("\n❌ TEST 1 FAILED: Interception did not work")
            print(f"❌ Stop reason: {response_stop_reason}")
            print("❌ Response contains tool_use blocks")
            return False

        elif has_text and response_stop_reason != "tool_use":
            text_block = next(
                block for block in response_content
                if (block.get("type") if isinstance(block, dict) else block.type) == "text"
            )
            text_content = text_block.get("text") if isinstance(text_block, dict) else text_block.text

            print(f"\n📝 Response Text:")
            print(f"   {text_content[:200]}...")

            if "litellm" in text_content.lower():
                print("\n" + "="*80)
                print("✅ TEST 1 PASSED!")
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
            return False

    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_websearch_interception_streaming():
    """
    Test WebSearch interception with streaming request.
    Validates that stream=True is converted to stream=False transparently.
    """
    print("\n" + "="*80)
    print("E2E TEST 2: WebSearch Interception (Streaming)")
    print("="*80)

    # Router already initialized from test 1
    print("\n✅ Using existing router configuration")
    print("✅ WebSearch interception already enabled for Bedrock")
    print("✅ Streaming will be converted to non-streaming for WebSearch interception")

    try:
        # Make request with WebSearch tool AND stream=True
        print("\n📞 Making litellm.messages.acreate() call with stream=True...")
        print(f"   Model: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        print(f"   Query: 'What is LiteLLM?'")
        print(f"   Tools: WebSearch")
        print(f"   Stream: True (will be converted to False)")

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
            stream=True,  # REQUEST STREAMING
        )

        print("\n✅ Received response!")

        # Check if response is actually a stream (async generator)
        import inspect
        is_stream = inspect.isasyncgen(response)

        if is_stream:
            print("\n⚠️  WARNING: Response is a stream (async_generator)")
            print("⚠️  This means stream conversion didn't work!")
            print("\n📦 Consuming stream chunks:")

            chunks = []
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                print(f"\n--- Chunk {chunk_count} ---")
                print(chunk)
                chunks.append(chunk)

            print(f"\n❌ TEST 2 FAILED: Got {len(chunks)} stream chunks instead of single response")
            return False

        # If not a stream, validate as normal response
        print("✅ Response is NOT a stream (conversion worked!)")

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
            print("\n❌ TEST 2 FAILED: Interception did not work")
            print("❌ Response contains tool_use blocks")
            return False

        elif has_text and response_stop_reason != "tool_use":
            text_block = next(
                block for block in response_content
                if (block.get("type") if isinstance(block, dict) else block.type) == "text"
            )
            text_content = text_block.get("text") if isinstance(text_block, dict) else text_block.text

            print(f"\n📝 Response Text:")
            print(f"   {text_content[:200]}...")

            if "litellm" in text_content.lower():
                print("\n" + "="*80)
                print("✅ TEST 2 PASSED!")
                print("="*80)
                print("✅ User made ONE litellm.messages.acreate() call with stream=True")
                print("✅ Stream was transparently converted to non-streaming")
                print("✅ Got back final answer (not tool_use)")
                print("✅ Agentic loop executed transparently")
                print("✅ WebSearch interception working with streaming!")
                print("="*80)
                return True
            else:
                print("\n⚠️  Got text response but doesn't mention LiteLLM")
                return False
        else:
            print("\n❌ Unexpected response format")
            return False

    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
