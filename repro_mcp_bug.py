"""
Reproduction script for the MCP + Responses API back-to-back calls bug.

This script demonstrates the issue where sending 2 back-to-back MCP Responses API 
calls with streaming fails with:
    "Error creating initial response iterator: argument of type 'NoneType' is not iterable"

To reproduce the bug (before fix):
1. Revert the defensive None checks in the MCP handler
2. Run this script against a LiteLLM proxy with MCP servers configured

Usage:
    # Against a local LiteLLM proxy with MCP configured
    export LITELLM_PROXY_URL="http://localhost:4000"
    export LITELLM_API_KEY="your-api-key"
    python repro_mcp_bug.py

    # Or test with mocks (no real API needed)
    python repro_mcp_bug.py --mock
"""

import asyncio
import os
import sys
import argparse
from typing import Any, List

# Ensure the local litellm package is used
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm
from litellm._logging import verbose_logger
import logging

# Enable verbose logging to see the error
verbose_logger.setLevel(logging.DEBUG)


async def make_streaming_mcp_call(call_number: int, model: str = "gpt-4o-mini"):
    """
    Make a single streaming MCP Responses API call.
    
    Args:
        call_number: Which call this is (for logging)
        model: The model to use
    """
    print(f"\n{'='*60}")
    print(f"📞 Making MCP Responses API call #{call_number}")
    print(f"{'='*60}")
    
    # MCP tool configuration - this tells LiteLLM to use the MCP gateway
    mcp_tool_config = {
        "type": "mcp",
        "server_url": "litellm_proxy",  # Use litellm_proxy to trigger MCP handling
        "require_approval": "never",     # Auto-execute tools
    }
    
    try:
        # Make the streaming call
        response = await litellm.aresponses(
            model=model,
            tools=[mcp_tool_config],
            tool_choice="auto",
            input=[
                {
                    "role": "user",
                    "type": "message",
                    "content": f"Test message #{call_number} - what is 2+2?",
                }
            ],
            stream=True,
        )
        
        print(f"✅ Call #{call_number}: Got response iterator: {type(response)}")
        
        # Consume the stream
        chunks = []
        chunk_count = 0
        async for chunk in response:
            chunk_count += 1
            chunk_type = getattr(chunk, 'type', 'unknown')
            chunks.append(chunk)
            if chunk_count <= 5 or chunk_count % 10 == 0:
                print(f"   📦 Chunk {chunk_count}: {chunk_type}")
        
        print(f"✅ Call #{call_number}: Received {len(chunks)} chunks successfully")
        return True
        
    except Exception as e:
        print(f"❌ Call #{call_number} FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_back_to_back_calls_with_mocks():
    """
    Test back-to-back MCP calls using mocks (no real API needed).
    This demonstrates where the bug occurs in the code path.
    """
    from unittest.mock import AsyncMock, patch, MagicMock
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    print("\n" + "="*70)
    print("🧪 Testing back-to-back MCP calls WITH MOCKS")
    print("="*70)
    
    # Create mock MCP tools
    mock_mcp_tools = [
        MagicMock(
            name='search_tool',
            description='Search for information',
            inputSchema={'type': 'object', 'properties': {'query': {'type': 'string'}}}
        ),
    ]
    # Set the name attribute properly (MagicMock's name kwarg doesn't work as expected)
    mock_mcp_tools[0].name = 'search_tool'
    
    # Create a mock streaming response
    class MockStreamingResponse:
        def __init__(self):
            self.chunks = [
                MagicMock(type='response.created'),
                MagicMock(type='response.output_item.added'),
                MagicMock(type='response.completed'),
            ]
            self._index = 0
            
        def __aiter__(self):
            return self
            
        async def __anext__(self):
            if self._index >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self._index]
            self._index += 1
            return chunk
    
    # Track call count to verify both calls are made
    call_count = 0
    
    async def mock_get_tools(*args, **kwargs):
        """Mock that returns tools but could return None in edge cases"""
        nonlocal call_count
        call_count += 1
        print(f"   🔧 _get_mcp_tools_from_manager called (call #{call_count})")
        return (mock_mcp_tools, ["test_server"])
    
    # Apply mocks
    with patch.object(
        LiteLLM_Proxy_MCP_Handler,
        '_get_mcp_tools_from_manager',
        new=mock_get_tools
    ), patch.object(
        LiteLLM_Proxy_MCP_Handler,
        '_execute_tool_calls',
        new=AsyncMock(return_value=[])
    ), patch(
        'litellm.aresponses',
        return_value=MockStreamingResponse()
    ):
        
        results = []
        
        # Make back-to-back calls
        for i in range(1, 3):
            print(f"\n--- Call #{i} ---")
            try:
                result = await make_streaming_mcp_call(i)
                results.append(result)
            except Exception as e:
                print(f"❌ Call #{i} raised exception: {e}")
                results.append(False)
        
        print("\n" + "="*70)
        print("📊 Results Summary:")
        print("="*70)
        for i, result in enumerate(results, 1):
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   Call #{i}: {status}")
        
        if all(results):
            print("\n🎉 All back-to-back calls succeeded!")
        else:
            print("\n⚠️  Some calls failed - this indicates the bug!")
            
        return all(results)


async def test_back_to_back_calls_real_api():
    """
    Test back-to-back MCP calls against a real LiteLLM proxy.
    Requires LITELLM_PROXY_URL and LITELLM_API_KEY environment variables.
    """
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not api_key:
        print("⚠️  LITELLM_API_KEY not set, skipping real API test")
        return None
    
    print("\n" + "="*70)
    print("🧪 Testing back-to-back MCP calls against REAL API")
    print(f"   Proxy URL: {proxy_url}")
    print("="*70)
    
    # Configure litellm to use the proxy
    litellm.api_base = proxy_url
    litellm.api_key = api_key
    
    results = []
    
    # Make back-to-back calls
    for i in range(1, 3):
        result = await make_streaming_mcp_call(i)
        results.append(result)
        # Small delay between calls to simulate real usage
        await asyncio.sleep(0.5)
    
    print("\n" + "="*70)
    print("📊 Results Summary:")
    print("="*70)
    for i, result in enumerate(results, 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   Call #{i}: {status}")
    
    if all(results):
        print("\n🎉 All back-to-back calls succeeded!")
    else:
        print("\n⚠️  Some calls failed - this indicates the bug!")
        
    return all(results)


async def test_none_handling_directly():
    """
    Directly test the functions that caused the NoneType error.
    This shows exactly where the bug occurs.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    print("\n" + "="*70)
    print("🧪 Testing None handling in MCP handler functions")
    print("="*70)
    
    test_cases = [
        {
            "name": "_deduplicate_mcp_tools with None mcp_tools",
            "func": lambda: LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(None, None),
            "expected": ([], {}),
        },
        {
            "name": "_filter_mcp_tools_by_allowed_tools with None inputs",
            "func": lambda: LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(None, None),
            "expected": [],
        },
    ]
    
    all_passed = True
    for tc in test_cases:
        try:
            result = tc["func"]()
            if result == tc["expected"]:
                print(f"   ✅ {tc['name']}: PASSED")
            else:
                print(f"   ⚠️  {tc['name']}: Got {result}, expected {tc['expected']}")
                all_passed = False
        except TypeError as e:
            if "NoneType" in str(e):
                print(f"   ❌ {tc['name']}: NoneType ERROR - {e}")
                all_passed = False
            else:
                raise
    
    # Test async function
    print("\n   Testing async function...")
    try:
        result = await LiteLLM_Proxy_MCP_Handler._process_mcp_tools_without_openai_transform(
            user_api_key_auth=None,
            mcp_tools_with_litellm_proxy=None
        )
        if result == ([], {}):
            print(f"   ✅ _process_mcp_tools_without_openai_transform with None: PASSED")
        else:
            print(f"   ⚠️  Got {result}, expected ([], {{}})")
            all_passed = False
    except TypeError as e:
        if "NoneType" in str(e):
            print(f"   ❌ _process_mcp_tools_without_openai_transform: NoneType ERROR - {e}")
            all_passed = False
        else:
            raise
    
    return all_passed


async def main():
    parser = argparse.ArgumentParser(description="Reproduce MCP + Responses API bug")
    parser.add_argument("--mock", action="store_true", help="Use mocks instead of real API")
    parser.add_argument("--direct", action="store_true", help="Test None handling directly")
    args = parser.parse_args()
    
    print("="*70)
    print("🐛 MCP + Responses API Bug Reproduction Script")
    print("="*70)
    print("""
This script reproduces the bug where back-to-back MCP Responses API 
streaming calls fail with:

    "Error creating initial response iterator: argument of type 'NoneType' is not iterable"

The bug occurs in the MCPEnhancedStreamingIterator when:
1. First streaming call works fine
2. Second streaming call fails because certain parameters are None

Root cause: Missing defensive None checks in:
- _get_allowed_mcp_servers_from_mcp_server_names()
- _deduplicate_mcp_tools()
- _filter_mcp_tools_by_allowed_tools()
- _extract_mcp_headers_from_params()
""")
    
    results = {}
    
    # Test 1: Direct None handling test
    if args.direct or not args.mock:
        results["direct"] = await test_none_handling_directly()
    
    # Test 2: Mock test (always run)
    if args.mock or not os.getenv("LITELLM_API_KEY"):
        results["mock"] = await test_back_to_back_calls_with_mocks()
    
    # Test 3: Real API test (if credentials available)
    if not args.mock and os.getenv("LITELLM_API_KEY"):
        results["real_api"] = await test_back_to_back_calls_real_api()
    
    # Summary
    print("\n" + "="*70)
    print("📋 FINAL SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    # Return exit code based on results
    if all(r for r in results.values() if r is not None):
        print("\n✅ All tests passed - the bug appears to be fixed!")
        return 0
    else:
        print("\n❌ Some tests failed - the bug may still exist!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
