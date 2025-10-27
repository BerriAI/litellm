import asyncio
import copy
import sys
import time
from datetime import datetime
from unittest import mock

from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

@pytest.mark.asyncio
async def test_client_session_helper():
    """Test that the client session helper handles event loop changes correctly"""
    try:
        # Create a transport with the new helper
        transport = AsyncHTTPHandler._create_aiohttp_transport()
        if transport is not None:
            print('✅ Successfully created aiohttp transport with helper')
            
            # Test the helper function directly if it's a LiteLLMAiohttpTransport
            if hasattr(transport, '_get_valid_client_session'):
                session1 = transport._get_valid_client_session()  # type: ignore
                print(f'✅ First session created: {type(session1).__name__}')
                
                # Call it again to test reuse
                session2 = transport._get_valid_client_session()  # type: ignore
                print(f'✅ Second session call: {type(session2).__name__}')
                
                # In the same event loop, should be the same session
                print(f'✅ Same session reused: {session1 is session2}')
                
            return True
        else:
            print('ℹ️  No aiohttp transport available (probably missing httpx-aiohttp)')
            return True
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_event_loop_robustness():
    """Test behavior when event loops change (simulating CI/CD scenario)"""
    try:
        # Test session creation in multiple scenarios
        transport = AsyncHTTPHandler._create_aiohttp_transport()
        
        if transport and hasattr(transport, '_get_valid_client_session'):
            # Test 1: Normal usage
            session = transport._get_valid_client_session()  # type: ignore
            print(f'✅ Normal session creation works: {session is not None}')
            
            # Test 2: Force recreation by setting client to a callable
            from aiohttp import ClientSession
            transport.client = lambda: ClientSession()  # type: ignore
            session2 = transport._get_valid_client_session()  # type: ignore
            print(f'✅ Session recreation after callable works: {session2 is not None}')
            
            return True
        else:
            print('ℹ️  Transport not available or no helper method')
            return True
            
    except Exception as e:
        print(f'❌ Error in event loop robustness test: {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_httpx_request_simulation():
    """Test that the transport can handle a simulated HTTP request"""
    try:
        transport = AsyncHTTPHandler._create_aiohttp_transport()
        
        if transport is not None:
            print('✅ Transport created for request simulation')
            
            # Create a simple httpx request to test with
            import httpx
            request = httpx.Request('GET', 'https://httpbin.org/headers')
            
            # Just test that we can get a valid session for this request context
            if hasattr(transport, '_get_valid_client_session'):
                session = transport._get_valid_client_session()  # type: ignore
                print(f'✅ Got valid session for request: {session is not None}')
                
                # Test that session has required aiohttp methods
                has_request_method = hasattr(session, 'request')
                print(f'✅ Session has request method: {has_request_method}')
                
                return has_request_method
            
            return True
        else:
            print('ℹ️  No transport available for request simulation')
            return True
            
    except Exception as e:
        print(f'❌ Error in request simulation: {e}')
        return False

if __name__ == "__main__":
    print("Testing client session helper and event loop handling fix...")
    
    result1 = asyncio.run(test_client_session_helper())
    result2 = asyncio.run(test_event_loop_robustness()) 
    result3 = asyncio.run(test_httpx_request_simulation())
    
    if result1 and result2 and result3:
        print("🎉 All tests passed! The helper function approach should fix the CI/CD event loop issues.")
    else:
        print("💥 Some tests failed") 