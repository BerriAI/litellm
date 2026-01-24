"""
Force reproduction of the MCP + Responses API bug.

This script patches the internal functions to return None, 
simulating the conditions that cause the bug.

Usage:
    python force_repro_mcp_bug.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import patch, AsyncMock, MagicMock
import litellm
from litellm._logging import verbose_logger
import logging

# Enable verbose logging to see the error
verbose_logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def test_with_none_allowed_servers():
    """
    Force the bug by making get_mcp_servers_from_ids return None.
    This simulates the condition where allowed_mcp_servers becomes None.
    """
    print("\n" + "="*70)
    print("🐛 TEST 1: Force None from get_mcp_servers_from_ids")
    print("="*70)
    
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
    
    # Save original method
    original_method = global_mcp_server_manager.get_mcp_servers_from_ids
    
    # Patch to return None
    def return_none(*args, **kwargs):
        print("   🔧 get_mcp_servers_from_ids called - returning None!")
        return None  # This should cause the bug!
    
    global_mcp_server_manager.get_mcp_servers_from_ids = return_none
    
    try:
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never",
        }
        
        response = await litellm.aresponses(
            model="gpt-4o-mini",
            tools=[mcp_tool_config],
            input=[{"role": "user", "type": "message", "content": "test"}],
            stream=True,
        )
        
        async for chunk in response:
            print(f"   Chunk: {getattr(chunk, 'type', 'unknown')}")
            
        print("   ✅ No error occurred (bug might be fixed)")
        
    except TypeError as e:
        if "NoneType" in str(e) and "not iterable" in str(e):
            print(f"   🐛 BUG REPRODUCED! Error: {e}")
            import traceback
            traceback.print_exc()
            return True
        else:
            print(f"   ❌ Different error: {e}")
            raise
    except Exception as e:
        print(f"   ❌ Other error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original method
        global_mcp_server_manager.get_mcp_servers_from_ids = original_method
    
    return False


async def test_with_none_allowed_mcp_servers():
    """
    Force the bug by making get_allowed_mcp_servers return None.
    """
    print("\n" + "="*70)
    print("🐛 TEST 2: Force None from get_allowed_mcp_servers")
    print("="*70)
    
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
    
    # Save original method
    original_method = global_mcp_server_manager.get_allowed_mcp_servers
    
    # Patch to return None
    async def return_none(*args, **kwargs):
        print("   🔧 get_allowed_mcp_servers called - returning None!")
        return None  # This should cause the bug!
    
    global_mcp_server_manager.get_allowed_mcp_servers = return_none
    
    try:
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never",
        }
        
        response = await litellm.aresponses(
            model="gpt-4o-mini",
            tools=[mcp_tool_config],
            input=[{"role": "user", "type": "message", "content": "test"}],
            stream=True,
        )
        
        async for chunk in response:
            print(f"   Chunk: {getattr(chunk, 'type', 'unknown')}")
            
        print("   ✅ No error occurred (bug might be fixed)")
        
    except TypeError as e:
        if "NoneType" in str(e) and "not iterable" in str(e):
            print(f"   🐛 BUG REPRODUCED! Error: {e}")
            import traceback
            traceback.print_exc()
            return True
        else:
            print(f"   ❌ Different error: {e}")
            raise
    except Exception as e:
        print(f"   ❌ Other error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original method
        global_mcp_server_manager.get_allowed_mcp_servers = original_method
    
    return False


async def test_with_none_tools_from_server():
    """
    Force the bug by making _get_tools_from_mcp_servers return None.
    """
    print("\n" + "="*70)
    print("🐛 TEST 3: Force None from _get_tools_from_mcp_servers")
    print("="*70)
    
    from litellm.proxy._experimental.mcp_server import server as mcp_server_module
    
    # Save original function
    original_func = mcp_server_module._get_tools_from_mcp_servers
    
    # Patch to return None
    async def return_none(*args, **kwargs):
        print("   🔧 _get_tools_from_mcp_servers called - returning None!")
        return None  # This should cause the bug!
    
    mcp_server_module._get_tools_from_mcp_servers = return_none
    
    try:
        mcp_tool_config = {
            "type": "mcp",
            "server_url": "litellm_proxy",
            "require_approval": "never",
        }
        
        response = await litellm.aresponses(
            model="gpt-4o-mini",
            tools=[mcp_tool_config],
            input=[{"role": "user", "type": "message", "content": "test"}],
            stream=True,
        )
        
        async for chunk in response:
            print(f"   Chunk: {getattr(chunk, 'type', 'unknown')}")
            
        print("   ✅ No error occurred (bug might be fixed)")
        
    except TypeError as e:
        if "NoneType" in str(e) and "not iterable" in str(e):
            print(f"   🐛 BUG REPRODUCED! Error: {e}")
            import traceback
            traceback.print_exc()
            return True
        else:
            print(f"   ❌ Different error: {e}")
            raise
    except Exception as e:
        print(f"   ❌ Other error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original function
        mcp_server_module._get_tools_from_mcp_servers = original_func
    
    return False


async def test_handler_functions_directly():
    """
    Test the handler functions directly with None inputs to reproduce the bug.
    """
    print("\n" + "="*70)
    print("🐛 TEST 4: Direct function calls with None")
    print("="*70)
    
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
    
    bugs_found = []
    
    # Test _deduplicate_mcp_tools
    print("\n   Testing _deduplicate_mcp_tools(None, None)...")
    try:
        result = LiteLLM_Proxy_MCP_Handler._deduplicate_mcp_tools(None, None)
        print(f"   ✅ Returned: {result}")
    except TypeError as e:
        if "NoneType" in str(e):
            print(f"   🐛 BUG! {e}")
            bugs_found.append("_deduplicate_mcp_tools")
    
    # Test _filter_mcp_tools_by_allowed_tools
    print("\n   Testing _filter_mcp_tools_by_allowed_tools(None, None)...")
    try:
        result = LiteLLM_Proxy_MCP_Handler._filter_mcp_tools_by_allowed_tools(None, None)
        print(f"   ✅ Returned: {result}")
    except TypeError as e:
        if "NoneType" in str(e):
            print(f"   🐛 BUG! {e}")
            bugs_found.append("_filter_mcp_tools_by_allowed_tools")
    
    # Test _get_allowed_mcp_servers_from_mcp_server_names
    print("\n   Testing _get_allowed_mcp_servers_from_mcp_server_names([], None)...")
    try:
        from litellm.proxy._experimental.mcp_server.server import _get_allowed_mcp_servers_from_mcp_server_names
        result = await _get_allowed_mcp_servers_from_mcp_server_names([], None)
        print(f"   ✅ Returned: {result}")
    except ImportError as e:
        print(f"   ⏭️  Skipped (missing dependency: {e})")
    except TypeError as e:
        if "NoneType" in str(e):
            print(f"   🐛 BUG! {e}")
            bugs_found.append("_get_allowed_mcp_servers_from_mcp_server_names")
    
    return bugs_found


async def main():
    print("="*70)
    print("🔬 FORCE REPRODUCTION OF MCP BUG")
    print("="*70)
    print("""
This script forces the bug to occur by patching internal functions
to return None, simulating the conditions that cause:

  "argument of type 'NoneType' is not iterable"
""")
    
    all_bugs = []
    
    # Test 4 first - direct function calls (doesn't need real API)
    bugs = await test_handler_functions_directly()
    all_bugs.extend(bugs)
    
    # The other tests need more setup, skip for now
    # await test_with_none_allowed_servers()
    # await test_with_none_allowed_mcp_servers() 
    # await test_with_none_tools_from_server()
    
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if all_bugs:
        print(f"\n🐛 Found bugs in {len(all_bugs)} function(s):")
        for bug in all_bugs:
            print(f"   - {bug}")
        print("\nThese functions don't handle None inputs properly!")
    else:
        print("\n✅ No bugs found - all functions handle None properly")
        print("   (The defensive checks are working)")


if __name__ == "__main__":
    asyncio.run(main())
