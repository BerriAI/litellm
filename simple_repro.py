#!/usr/bin/env python3
"""
Simple reproduction of the MCP NoneType bug.

This script demonstrates the exact bug that causes:
  "argument of type 'NoneType' is not iterable"

Run with fix reverted to see the bug:
  git checkout HEAD~2 -- litellm/responses/mcp/litellm_proxy_mcp_handler.py
  python simple_repro.py

Run with fix applied to see it working:
  git checkout HEAD -- litellm/responses/mcp/litellm_proxy_mcp_handler.py
  python simple_repro.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

print("="*60)
print("🧪 MCP NoneType Bug Reproduction")
print("="*60)

# This is the exact scenario that causes the bug:
# When allowed_mcp_servers or mcp_tools is None (which can happen
# in certain edge cases with back-to-back calls), the code crashes.

print("\nTest 1: _deduplicate_mcp_tools(None, None)")
print("-" * 40)
try:
    result = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(None, None)
    print(f"✅ SUCCESS: Returned {result}")
except TypeError as e:
    print(f"🐛 BUG REPRODUCED: {e}")

print("\nTest 2: _filter_mcp_tools_by_allowed_tools(None, None)")
print("-" * 40)
try:
    result = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(None, None)
    print(f"✅ SUCCESS: Returned {result}")
except TypeError as e:
    print(f"🐛 BUG REPRODUCED: {e}")

print("\n" + "="*60)
print("How this bug manifests in production:")
print("="*60)
print("""
When making back-to-back MCP Responses API calls, if one of these
conditions occurs:

1. get_allowed_mcp_servers() returns None (network/timing issue)
2. get_mcp_servers_from_ids() returns None 
3. The MCP server has no tools configured

Then the code paths above get called with None, causing:
  "Error creating initial response iterator: argument of type 'NoneType' is not iterable"

The fix adds defensive None checks:
  if mcp_tools is None:
      mcp_tools = []
""")
