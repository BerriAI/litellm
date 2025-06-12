#!/usr/bin/env python3
"""
Test script for dual authentication in MCP servers.

This script tests the dual authentication implementation where:
- x-litellm-key is used for LiteLLM system authentication
- Authorization header is used for user authorization passed through to MCP servers
"""

import asyncio
import aiohttp
import json


async def test_dual_auth():
    """Test dual authentication for MCP tool calls."""
    
    # Test configuration
    base_url = "http://localhost:4000"
    litellm_key = "sk-1234"  # Replace with your actual LiteLLM key
    user_auth_token = "Bearer user-token-5678"  # Replace with actual user token
    
    headers = {
        "x-litellm-key": litellm_key,
        "Authorization": user_auth_token,
        "Content-Type": "application/json"
    }
    
    # Test payload - replace with actual tool name and arguments
    payload = {
        "name": "echo_tool",  # Replace with actual tool name
        "arguments": {
            "message": "Hello from dual auth test"
        }
    }
    
    async with aiohttp.ClientSession() as session:
        print("Testing dual authentication for MCP tool calls...")
        print(f"LiteLLM Key: {litellm_key}")
        print(f"User Auth Token: {user_auth_token[:20]}...")
        print(f"Tool: {payload['name']}")
        print()
        
        try:
            # Test 1: Call with both headers
            print("Test 1: Calling with both x-litellm-key and Authorization headers")
            async with session.post(
                f"{base_url}/mcp/tools/call",
                headers=headers,
                json=payload
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print(f"Result: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
            print()
            
            # Test 2: Call with only x-litellm-key (should work but no user auth)
            print("Test 2: Calling with only x-litellm-key header")
            headers_no_auth = {
                "x-litellm-key": litellm_key,
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{base_url}/mcp/tools/call",
                headers=headers_no_auth,
                json=payload
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    print(f"Result: {json.dumps(result, indent=2)}")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
            print()
            
            # Test 3: Call with only Authorization header (should fail LiteLLM auth)
            print("Test 3: Calling with only Authorization header (should fail)")
            headers_no_litellm = {
                "Authorization": user_auth_token,
                "Content-Type": "application/json"
            }
            async with session.post(
                f"{base_url}/mcp/tools/call",
                headers=headers_no_litellm,
                json=payload
            ) as response:
                print(f"Status: {response.status}")
                error_text = await response.text()
                print(f"Expected error: {error_text}")
            print()
            
        except Exception as e:
            print(f"Test failed with exception: {e}")


async def test_mcp_tools_list():
    """Test listing MCP tools with dual authentication."""
    
    base_url = "http://localhost:4000"
    litellm_key = "sk-1234"  # Replace with your actual LiteLLM key
    
    headers = {
        "x-litellm-key": litellm_key,
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        print("Testing MCP tools list endpoint...")
        
        try:
            async with session.get(
                f"{base_url}/mcp/tools/list",
                headers=headers
            ) as response:
                print(f"Status: {response.status}")
                if response.status == 200:
                    tools = await response.json()
                    print(f"Available tools: {len(tools)}")
                    for tool in tools[:3]:  # Show first 3 tools
                        print(f"  - {tool.get('name', 'N/A')}: {tool.get('description', 'N/A')[:50]}...")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    
        except Exception as e:
            print(f"Test failed with exception: {e}")


if __name__ == "__main__":
    print("=== MCP Dual Authentication Test ===")
    print()
    
    # Test tools list first
    asyncio.run(test_mcp_tools_list())
    print()
    
    # Test dual auth tool calls
    asyncio.run(test_dual_auth())
    
    print("=== Test Complete ===")
    print()
    print("Note: Check the LiteLLM logs for authorization header detection messages.")
    print("You should see warnings about authorization headers being provided but not")
    print("fully supported due to MCP SDK limitations.")