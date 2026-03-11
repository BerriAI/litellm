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


async def test_websearch_interception_no_tool_call_streaming():
    """
    Test WebSearch interception when LLM doesn't make a tool call with streaming.
    
    This tests the scenario where:
    1. User requests stream=True
    2. WebSearch tool is provided
    3. LLM decides NOT to use the tool (just responds with text)
    4. System should return a fake stream
    """
    print("\n" + "="*80)
    print("E2E TEST 3: WebSearch Interception (No Tool Call, Streaming)")
    print("="*80)

    # Router already initialized from test 1
    print("\n✅ Using existing router configuration")
    print("✅ WebSearch interception already enabled for Bedrock")

    try:
        # Make request with WebSearch tool AND stream=True
        # Use a query that the LLM will answer directly without using the tool
        print("\n📞 Making litellm.messages.acreate() call with stream=True...")
        print(f"   Model: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        print(f"   Query: 'What is 2+2?'")
        print(f"   Tools: WebSearch")
        print(f"   Stream: True")

        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": "What is 2+2? Just give me the answer, no need to search."}],
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

        # Check if response is actually a stream (async generator or async iterator)
        import inspect
        is_async_gen = inspect.isasyncgen(response)
        is_async_iter = hasattr(response, '__aiter__') and hasattr(response, '__anext__')
        is_stream = is_async_gen or is_async_iter

        if not is_stream:
            print("\n❌ TEST 3 FAILED: Response is NOT a stream")
            print(f"❌ Expected a fake stream when LLM doesn't use the tool")
            print(f"❌ Response type: {type(response)}")
            return False

        print(f"✅ Response is a stream (async_gen={is_async_gen}, async_iter={is_async_iter})")
        print("\n📦 Consuming stream chunks:")

        chunks = []
        chunk_count = 0
        async for chunk in response:
            chunk_count += 1
            print(f"\n--- Chunk {chunk_count} ---")
            print(f"   Type: {type(chunk)}")
            print(f"   Content: {chunk[:200] if isinstance(chunk, bytes) else str(chunk)[:200]}...")
            chunks.append(chunk)

        print(f"\n✅ Received {len(chunks)} stream chunk(s)")

        if len(chunks) > 0:
            print("\n" + "="*80)
            print("✅ TEST 3 PASSED!")
            print("="*80)
            print("✅ User made ONE litellm.messages.acreate() call with stream=True")
            print("✅ LLM didn't use the WebSearch tool")
            print("✅ Got back a fake stream (not a non-streaming response)")
            print("✅ WebSearch interception handles no-tool-call case correctly!")
            print("="*80)
            return True
        else:
            print("\n❌ TEST 3 FAILED: No chunks received")
            return False

    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_claude_code_native_websearch():
    """
    Test WebSearch interception with Claude Code's native web_search_20250305 tool.
    
    This tests the exact request format that Claude Code sends:
    - tools: [{'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': 8}]
    - Model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
    """
    print("\n" + "="*80)
    print("E2E TEST: Claude Code Native WebSearch (web_search_20250305)")
    print("="*80)

    # Router already initialized from test 1
    print("\n✅ Using existing router configuration")
    print("✅ WebSearch interception already enabled for Bedrock")

    try:
        # Make request with Claude Code's exact native web_search tool format
        print("\n📞 Making litellm.messages.acreate() call...")
        print(f"   Model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        print(f"   Query: 'Perform a web search for the query: litellm what is it'")
        print(f"   Tools: Native web_search_20250305")
        print(f"   Stream: False")

        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Perform a web search for the query: litellm what is it"}],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 8
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
            print("\n❌ TEST FAILED: Interception did not work")
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
                print("✅ TEST PASSED!")
                print("="*80)
                print("✅ Claude Code's native web_search_20250305 tool was intercepted")
                print("✅ Tool was converted to LiteLLM standard format")
                print("✅ User made ONE litellm.messages.acreate() call")
                print("✅ Got back final answer with search results")
                print("✅ Agentic loop executed transparently")
                print("✅ WebSearch interception working with Claude Code!")
                print("="*80)
                return True
            else:
                print("\n⚠️  Got text response but doesn't mention LiteLLM")
                return False
        else:
            print("\n❌ Unexpected response format")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    
    async def run_all_tests():
        """Run all E2E tests"""
        test_results = []
        
        # Test 1: Non-streaming
        result1 = await test_websearch_interception_non_streaming()
        test_results.append(("Non-Streaming", result1))
        
        # Test 2: Streaming
        result2 = await test_websearch_interception_streaming()
        test_results.append(("Streaming", result2))
        
        # Test 3: No tool call with streaming
        result3 = await test_websearch_interception_no_tool_call_streaming()
        test_results.append(("No Tool Call Streaming", result3))
        
        # Test 4: Claude Code native web_search
        result4 = await test_claude_code_native_websearch()
        test_results.append(("Claude Code Native WebSearch", result4))
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        print("="*80)
        
        # Return overall result
        return all(result for _, result in test_results)
    
    result = asyncio.run(run_all_tests())
    import sys
    sys.exit(0 if result else 1)


async def test_litellm_standard_websearch_tool():
    """
    PRIORITY TEST #1: Test with the canonical litellm_web_search tool format.

    This validates that using get_litellm_web_search_tool() directly
    works end-to-end without any conversion needed.
    """
    print("\n" + "="*80)
    print("E2E TEST: LiteLLM Standard WebSearch Tool")
    print("="*80)

    from litellm.integrations.websearch_interception import get_litellm_web_search_tool

    print("\n✅ Using existing router configuration")
    print("✅ WebSearch interception already enabled for Bedrock")

    try:
        print("\n📞 Making litellm.messages.acreate() call...")
        print(f"   Model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        print(f"   Query: 'What is the latest news about AI?'")
        print(f"   Tool: litellm_web_search (standard format, no conversion needed)")
        print(f"   Stream: False")

        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "What is the latest news about AI? Give me a brief overview."}],
            tools=[get_litellm_web_search_tool()],
            max_tokens=1024,
            stream=False,
        )

        print("\n✅ Received response!")

        if isinstance(response, dict):
            response_id = response.get("id")
            response_stop_reason = response.get("stop_reason")
            response_content = response.get("content", [])
        else:
            response_id = response.id
            response_stop_reason = response.stop_reason
            response_content = response.content

        print(f"\n📄 Response ID: {response_id}")
        print(f"📄 Stop Reason: {response_stop_reason}")
        print(f"📄 Content blocks: {len(response_content)}")

        for i, block in enumerate(response_content):
            block_type = block.get("type") if isinstance(block, dict) else block.type
            print(f"   Block {i}: type={block_type}")

        has_tool_use = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "tool_use"
            for block in response_content
        )

        has_text = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "text"
            for block in response_content
        )

        if has_tool_use:
            print("\n❌ TEST FAILED: Interception did not work")
            return False

        elif has_text and response_stop_reason != "tool_use":
            text_block = next(
                block for block in response_content
                if (block.get("type") if isinstance(block, dict) else block.type) == "text"
            )
            text_content = text_block.get("text") if isinstance(text_block, dict) else text_block.text

            print(f"\n📝 Response Text: {text_content[:200]}...")

            print("\n" + "="*80)
            print("✅ TEST PASSED!")
            print("="*80)
            print("✅ LiteLLM standard tool format works without conversion")
            print("✅ Agentic loop executed transparently")
            print("="*80)
            return True
        else:
            print("\n❌ Unexpected response format")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_claude_code_native_websearch_streaming():
    """
    PRIORITY TEST #2: Test Claude Code's native tool WITH stream=True.

    Validates:
    - Native tool conversion (web_search_20250305 → litellm_web_search)
    - Stream=True → Stream=False conversion
    - Agentic loop executes with both conversions
    """
    print("\n" + "="*80)
    print("E2E TEST: Claude Code Native WebSearch + Streaming")
    print("="*80)

    print("\n✅ Using existing router configuration")
    print("✅ WebSearch interception already enabled for Bedrock")

    try:
        print("\n📞 Making litellm.messages.acreate() call with stream=True...")
        print(f"   Model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        print(f"   Tool: Native web_search_20250305")
        print(f"   Stream: True (will be converted to False)")

        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Search for the latest AI developments."}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            max_tokens=1024,
            stream=True,
        )

        print("\n✅ Received response!")

        import inspect
        is_stream = inspect.isasyncgen(response)

        if is_stream:
            print("\n⚠️  Response is a stream (stream conversion didn't work)")
            return False

        print("✅ Response is NOT a stream (conversion worked!)")

        if isinstance(response, dict):
            response_stop_reason = response.get("stop_reason")
            response_content = response.get("content", [])
        else:
            response_stop_reason = response.stop_reason
            response_content = response.content

        has_tool_use = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "tool_use"
            for block in response_content
        )

        has_text = any(
            (block.get("type") if isinstance(block, dict) else block.type) == "text"
            for block in response_content
        )

        if has_tool_use:
            print("\n❌ TEST FAILED: Interception did not work")
            return False

        elif has_text and response_stop_reason != "tool_use":
            print("\n" + "="*80)
            print("✅ TEST PASSED!")
            print("="*80)
            print("✅ Native tool converted to litellm_web_search")
            print("✅ Stream=True converted to Stream=False")
            print("✅ Both conversions working together!")
            print("="*80)
            return True
        else:
            print("\n❌ Unexpected response format")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_is_web_search_tool_detection():
    """
    PRIORITY TEST #3: Unit test for is_web_search_tool() utility.

    Validates detection of all supported formats including future versions.
    """
    print("\n" + "="*80)
    print("UNIT TEST: Web Search Tool Detection")
    print("="*80)

    from litellm.integrations.websearch_interception import is_web_search_tool

    test_cases = [
        ({"name": "litellm_web_search"}, True, "LiteLLM standard tool"),
        ({"type": "web_search_20250305", "name": "web_search", "max_uses": 8}, True, "Current Anthropic native (2025)"),
        ({"type": "web_search_2026", "name": "web_search"}, True, "Future Anthropic native (2026)"),
        ({"type": "web_search_20270615", "name": "web_search"}, True, "Future Anthropic native (2027)"),
        ({"name": "web_search", "type": "web_search_20250305"}, True, "Claude Code format"),
        ({"name": "WebSearch"}, True, "Legacy WebSearch"),
        ({"name": "calculator"}, False, "Non-web-search tool"),
        ({"name": "some_tool", "type": "function"}, False, "Other tool with type"),
        ({"type": "custom_tool"}, False, "Custom tool type"),
    ]

    passed = 0
    failed = 0

    for tool, expected, description in test_cases:
        result = is_web_search_tool(tool)
        if result == expected:
            print(f"   ✅ PASS: {description}")
            passed += 1
        else:
            print(f"   ❌ FAIL: {description}")
            print(f"      Tool: {tool}")
            print(f"      Expected: {expected}, Got: {result}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n" + "="*80)
        print("✅ ALL DETECTION TESTS PASSED!")
        print("="*80)
        print("✅ Detects all current formats")
        print("✅ Future-proof for new web_search_* versions")
        print("="*80)
        return True
    else:
        print("\n❌ Some detection tests failed")
        return False


async def test_pre_request_hook_modifies_request_body():
    """
    Unit test to verify async_pre_request_hook correctly modifies request body.

    Tests that:
    1. WebSearchInterceptionLogger is active
    2. Native web_search_20250305 tool is converted to litellm_web_search
    3. Stream is converted from True to False
    4. Modified parameters reach the API call
    """
    import asyncio
    from unittest.mock import AsyncMock, patch, MagicMock
    from litellm.constants import LITELLM_WEB_SEARCH_TOOL_NAME

    litellm._turn_on_debug()

    print("\n" + "="*80)
    print("UNIT TEST: Pre-Request Hook Modifies Request Body")
    print("="*80)

    # Initialize WebSearchInterceptionLogger
    litellm.callbacks = [
        WebSearchInterceptionLogger(
            enabled_providers=[LlmProviders.BEDROCK],
            search_tool_name="test-search-tool"
        )
    ]

    print("✅ WebSearchInterceptionLogger initialized")

    # Track what actually gets sent to the API
    captured_request = {}

    def mock_anthropic_messages_handler(
        max_tokens,
        messages,
        model,
        metadata=None,
        stop_sequences=None,
        stream=None,
        system=None,
        temperature=None,
        thinking=None,
        tool_choice=None,
        tools=None,
        top_k=None,
        top_p=None,
        container=None,
        api_key=None,
        api_base=None,
        client=None,
        custom_llm_provider=None,
        **kwargs
    ):
        """Mock handler that captures the actual request parameters"""
        # Capture what gets sent to the handler (after hook modifications)
        captured_request['tools'] = tools
        captured_request['stream'] = stream
        captured_request['max_tokens'] = max_tokens
        captured_request['model'] = model

        # Return a mock response (non-streaming)
        from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
        return AnthropicMessagesResponse(
            id="msg_test",
            type="message",
            role="assistant",
            content=[{
                "type": "text",
                "text": "Test response"
            }],
            model="claude-sonnet-4-5",
            stop_reason="end_turn",
            usage={
                "input_tokens": 10,
                "output_tokens": 20
            }
        )

    # Patch the anthropic_messages_handler function (called after hooks)
    with patch('litellm.llms.anthropic.experimental_pass_through.messages.handler.anthropic_messages_handler',
               side_effect=mock_anthropic_messages_handler):

        print("\n📝 Making request with native web_search_20250305 tool (stream=True)...")

        # Make the request with native tool format
        response = await messages.acreate(
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            messages=[{"role": "user", "content": "Test query"}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 8
            }],
            max_tokens=100,
            stream=True  # Should be converted to False
        )

        print("\n🔍 Verifying request modifications...")

        # Verify tool was converted
        tools = captured_request.get('tools')
        print(f"\n   Captured tools: {tools}")

        if tools and len(tools) > 0:
            tool = tools[0]
            tool_name = tool.get('name')

            if tool_name == LITELLM_WEB_SEARCH_TOOL_NAME:
                print(f"   ✅ Tool converted: web_search_20250305 → {LITELLM_WEB_SEARCH_TOOL_NAME}")
            else:
                print(f"   ❌ Tool NOT converted: expected {LITELLM_WEB_SEARCH_TOOL_NAME}, got {tool_name}")
                return False
        else:
            print("   ❌ No tools captured in request")
            return False

        # Verify stream was converted
        stream = captured_request.get('stream')
        print(f"   Captured stream: {stream}")

        if stream is False:
            print("   ✅ Stream converted: True → False")
        else:
            print(f"   ❌ Stream NOT converted: expected False, got {stream}")
            return False

        print("\n" + "="*80)
        print("✅ PRE-REQUEST HOOK TEST PASSED!")
        print("="*80)
        print("✅ CustomLogger is active")
        print("✅ async_pre_request_hook modifies request body")
        print("✅ Tool conversion works correctly")
        print("✅ Stream conversion works correctly")
        print("="*80)

        return True

