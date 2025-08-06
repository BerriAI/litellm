#!/usr/bin/env python3
"""
Test script to verify Redis session patch is working for issue #12364
Requires Redis to be running on localhost:6379 and a valid Gemini API key
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import litellm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import litellm
from litellm.caching.caching import Cache
from litellm.responses.main import aresponses


async def test_redis_session_patch():
    """Test that conversation context is maintained with Redis patch"""
    
    # Check if Redis cache is configured
    if not hasattr(litellm, 'cache') or litellm.cache is None:
        print("❌ Redis cache not configured. Please set up Redis cache in your config.")
        return False
    
    # Test configuration
    API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
    if API_KEY == "your-api-key-here":
        print("❌ Please set GEMINI_API_KEY environment variable")
        return False
    
    try:
        print("Testing Redis session patch for conversation context...")
        
        # First request - establish context
        print("Making first request...")
        response1 = await aresponses(
            model="gemini-1.5-flash",
            input="Hello, my name is John. Please remember my name.",
            api_key=API_KEY,
            custom_llm_provider="gemini"
        )
        
        print(f"First response: {response1.output[0].content[0].text}")
        
        # Second request - test context continuity
        print("Making second request with context...")
        response2 = await aresponses(
            model="gemini-1.5-flash",
            input="What is my name?",
            previous_response_id=response1.id,
            api_key=API_KEY,
            custom_llm_provider="gemini"
        )
        
        print(f"Second response: {response2.output[0].content[0].text}")
        
        # Check if context was maintained
        response_text = response2.output[0].content[0].text.lower()
        if "john" in response_text:
            print("✅ SUCCESS: Context maintained - AI remembered the name!")
            return True
        else:
            print("❌ FAILED: Context not maintained - AI forgot the name")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    # Set up Redis cache if not already configured
    if not hasattr(litellm, 'cache') or litellm.cache is None:
        print("Setting up Redis cache...")
        litellm.cache = Cache(
            type="redis",
            host="localhost",
            port=6379,
            supported_call_types=["completion", "acompletion", "aresponses", "responses"]
        )
    
    # Run test
    success = asyncio.run(test_redis_session_patch())
    
    if success:
        print("\n✅ Redis session patch is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Redis session patch test failed!")
        sys.exit(1)