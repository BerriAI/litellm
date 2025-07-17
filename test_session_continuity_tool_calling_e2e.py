#!/usr/bin/env python3
"""
E2E test for tool calling with session continuity issue.

This test reproduces the issue where tool/function calling fails with session continuity
(previous_response_id) on Gemini models but works with OpenAI models.

Bug: https://github.com/BerriAI/litellm/issues/XXXX
"""

import asyncio
import json
import os
import sys
import httpx
import pytest
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath("."))

import litellm


async def test_tool_calling_with_session_continuity():
    """
    Test that demonstrates the tool calling + session continuity issue.
    
    Steps:
    1. Make initial request with tool to both Gemini and OpenAI
    2. Extract response_id from the response
    3. Make follow-up request with function result using previous_response_id
    4. Verify that OpenAI works but Gemini fails
    """
    
    # Test configuration
    gemini_model = "gemini/gemini-2.5-flash"
    openai_model = "gpt-4o"
    
    # Tool definition matching the issue description
    tool_def = {
        "name": "get_date",
        "description": "Get current date in 'YYYY-MM-DD' format.",
        "parameters": {"type": "object", "properties": {}},
        "type": "function"
    }
    
    tools = [tool_def]
    initial_messages = [{"role": "user", "content": "What date is it?"}]
    
    print("🔬 Testing tool calling with session continuity...")
    print(f"Gemini model: {gemini_model}")
    print(f"OpenAI model: {openai_model}")
    print()
    
    # Test results storage
    results = {
        "openai": {"initial": None, "followup": None, "error": None},
        "gemini": {"initial": None, "followup": None, "error": None}
    }
    
    # Test OpenAI first (should work)
    print("🤖 Testing OpenAI (expected to work)...")
    try:
        # Step 1: Initial request with tool
        print("  📤 Making initial request with tool...")
        openai_response_1 = await litellm.aresponses(
            model=openai_model,
            input=initial_messages,
            tools=tools,
            temperature=0
        )
        
        results["openai"]["initial"] = openai_response_1
        print(f"  ✅ Initial response ID: {openai_response_1.id}")
        
        # Extract tool call information
        tool_calls = []
        for output in openai_response_1.output:
            if hasattr(output, 'type') and output.type == 'function_call':
                tool_calls.append({
                    "call_id": output.call_id,
                    "name": output.name,
                    "arguments": output.arguments
                })
        
        if not tool_calls:
            print("  ⚠️  No tool calls found in OpenAI response")
            return
            
        print(f"  🔧 Found {len(tool_calls)} tool call(s)")
        
        # Step 2: Follow-up with function result
        print("  📤 Making follow-up request with function result...")
        function_result = [{"type": "function_call_output", "call_id": tool_calls[0]["call_id"], "output": "\"2025-07-17\""}]
        
        openai_response_2 = await litellm.aresponses(
            model=openai_model,
            input=function_result,
            previous_response_id=openai_response_1.id,
            tools=tools,
            temperature=0
        )
        
        results["openai"]["followup"] = openai_response_2
        print(f"  ✅ Follow-up response ID: {openai_response_2.id}")
        print("  ✅ OpenAI: Tool calling with session continuity WORKS")
        
    except Exception as e:
        results["openai"]["error"] = str(e)
        print(f"  ❌ OpenAI error: {e}")
    
    print()
    
    # Test Gemini (expected to fail)
    print("🔮 Testing Gemini (expected to fail)...")
    try:
        # Step 1: Initial request with tool
        print("  📤 Making initial request with tool...")
        gemini_response_1 = await litellm.aresponses(
            model=gemini_model,
            input=initial_messages,
            tools=tools,
            temperature=0
        )
        
        results["gemini"]["initial"] = gemini_response_1
        print(f"  ✅ Initial response ID: {gemini_response_1.id}")
        
        # Extract tool call information
        tool_calls = []
        for output in gemini_response_1.output:
            if hasattr(output, 'type') and output.type == 'function_call':
                tool_calls.append({
                    "call_id": output.call_id,
                    "name": output.name,
                    "arguments": output.arguments
                })
        
        if not tool_calls:
            print("  ⚠️  No tool calls found in Gemini response")
            return
            
        print(f"  🔧 Found {len(tool_calls)} tool call(s)")
        
        # Step 2: Follow-up with function result (this should fail)
        print("  📤 Making follow-up request with function result...")
        function_result = [{"type": "function_call_output", "call_id": tool_calls[0]["call_id"], "output": "\"2025-07-17\""}]
        
        gemini_response_2 = await litellm.aresponses(
            model=gemini_model,
            input=function_result,
            previous_response_id=gemini_response_1.id,
            tools=tools,
            temperature=0
        )
        
        results["gemini"]["followup"] = gemini_response_2
        print(f"  ✅ Follow-up response ID: {gemini_response_2.id}")
        print("  🎉 Gemini: Tool calling with session continuity WORKS (unexpected!)")
        
    except Exception as e:
        results["gemini"]["error"] = str(e)
        print(f"  ❌ Gemini error: {e}")
        if "function response parts is equal to the number of function call parts" in str(e):
            print("  🎯 This is the expected error - bug confirmed!")
        
    print()
    
    # Print summary
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for provider in ["openai", "gemini"]:
        print(f"\n{provider.upper()}:")
        result = results[provider]
        
        if result["error"]:
            print(f"  ❌ FAILED: {result['error']}")
        elif result["initial"] and result["followup"]:
            print(f"  ✅ SUCCESS: Both requests completed")
        elif result["initial"]:
            print(f"  ⚠️  PARTIAL: Initial request worked, follow-up failed")
        else:
            print(f"  ❌ FAILED: Initial request failed")
    
    # Check if we reproduced the expected behavior
    openai_works = results["openai"]["followup"] is not None and results["openai"]["error"] is None
    gemini_fails = results["gemini"]["error"] is not None and "function response parts is equal to the number of function call parts" in results["gemini"]["error"]
    
    print(f"\n🔍 Issue Status:")
    if openai_works and gemini_fails:
        print("  ✅ Bug reproduced successfully!")
        print("  - OpenAI: Tool calling with session continuity works")
        print("  - Gemini: Tool calling with session continuity fails with expected error")
    elif openai_works and not gemini_fails:
        print("  ⚠️  Unexpected: Both providers work - bug may be fixed!")
    elif not openai_works and gemini_fails:
        print("  ⚠️  Unexpected: Both providers fail - different issue?")
    else:
        print("  ❌ Could not reproduce the issue")
    
    return results


async def test_with_proxy():
    """
    Test the same scenario through LiteLLM proxy server.
    """
    print("\n🚀 Testing through LiteLLM Proxy...")
    
    # Proxy configuration - we'll test direct API calls
    proxy_url = "http://localhost:4000"
    
    try:
        # Test if proxy is running
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{proxy_url}/health")
            if health_response.status_code != 200:
                print("❌ Proxy is not running. Please start it first.")
                return
    except Exception as e:
        print(f"❌ Cannot connect to proxy: {e}")
        print("Please start the proxy with: litellm --config proxy_config.yaml")
        return
    
    print("✅ Proxy is running")
    
    # Use the same test but through proxy endpoints
    # This would require making HTTP requests to the proxy
    # For now, we'll skip this part as it requires the proxy to be set up
    print("⏭️  Proxy testing requires proxy setup - skipping for now")


if __name__ == "__main__":
    print("🧪 LiteLLM Tool Calling + Session Continuity E2E Test")
    print("=" * 60)
    
    # Check environment variables
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        print("❌ Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        sys.exit(1)
        
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run the test
    try:
        results = asyncio.run(test_tool_calling_with_session_continuity())
        print("\n🏁 Test completed!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()