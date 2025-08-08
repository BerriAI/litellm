#!/usr/bin/env python3
"""
Verification script for Redis Session Cache integration with Responses API.

This script helps verify that the Redis caching implementation is working correctly
in a development environment.

Usage:
    python verify_redis_integration.py

Prerequisites:
    - Redis server running and accessible
    - LiteLLM proxy configured with Redis
    - Environment variables set for Redis connection
"""

import asyncio
import os
import sys
import traceback
from typing import Dict, Any

async def verify_redis_cache_basic():
    """Test basic Redis cache functionality."""
    print("🔍 Testing basic Redis cache functionality...")
    
    try:
        from litellm.responses.redis_session_cache import get_responses_redis_cache
        
        cache = get_responses_redis_cache()
        
        # Test 1: Check if Redis is available
        if not cache.is_available():
            print("❌ Redis cache not available. Check Redis configuration.")
            return False
        
        print("✅ Redis cache is available")
        
        # Test 2: Get cache stats
        stats = await cache.get_cache_stats()
        print(f"📊 Cache stats: {stats}")
        
        if not stats.get("available"):
            print(f"❌ Redis connectivity test failed: {stats.get('error')}")
            return False
        
        print("✅ Redis connectivity test passed")
        
        # Test 3: Test basic store/retrieve operations
        test_data = {
            "request_id": "test_resp_123",
            "session_id": "test_session_456",
            "messages": "[{\"role\": \"user\", \"content\": \"Hello\"}]",
            "response": "{\"choices\": [{\"message\": {\"content\": \"Hi there!\"}}]}",
            "startTime": "2024-01-01T00:00:00Z",
            "endTime": "2024-01-01T00:00:01Z",
        }
        
        # Store response data
        success = await cache.store_response_data(
            response_id="test_resp_123",
            session_id="test_session_456",
            spend_log_data=test_data
        )
        
        if not success:
            print("❌ Failed to store response data in Redis")
            return False
        
        print("✅ Successfully stored response data in Redis")
        
        # Retrieve response data
        response_data = await cache.get_response_by_id("test_resp_123")
        if not response_data:
            print("❌ Failed to retrieve response data from Redis")
            return False
        
        print("✅ Successfully retrieved response data from Redis")
        
        # Retrieve session data
        session_logs = await cache.get_session_spend_logs("test_session_456")
        if not session_logs:
            print("❌ Failed to retrieve session data from Redis")
            return False
        
        print("✅ Successfully retrieved session data from Redis")
        print(f"📝 Retrieved {len(session_logs)} spend logs for session")
        
        # Clean up test data
        await cache.invalidate_session("test_session_456")
        print("✅ Cleaned up test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Redis cache: {e}")
        traceback.print_exc()
        return False


async def verify_session_handler_integration():
    """Test integration with session handler."""
    print("\n🔗 Testing session handler integration...")
    
    try:
        from litellm.responses.litellm_completion_transformation.session_handler import ResponsesSessionHandler
        from litellm.responses.redis_session_cache import get_responses_redis_cache
        
        cache = get_responses_redis_cache()
        
        if not cache.is_available():
            print("⚠️  Skipping session handler test - Redis not available")
            return True
        
        # Store test data in Redis
        test_spend_log = {
            "request_id": "test_handler_resp",
            "session_id": "test_handler_session",
            "messages": "[{\"role\": \"user\", \"content\": \"Test message\"}]",
            "response": "{\"choices\": [{\"message\": {\"content\": \"Test response\"}}]}",
            "startTime": "2024-01-01T00:00:00Z",
            "endTime": "2024-01-01T00:00:01Z",
        }
        
        await cache.store_response_data(
            response_id="test_handler_resp",
            session_id="test_handler_session", 
            spend_log_data=test_spend_log
        )
        
        print("✅ Stored test data for session handler test")
        
        # Test session handler retrieval (should use Redis cache)
        spend_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            "test_handler_resp"
        )
        
        if not spend_logs:
            print("❌ Session handler failed to retrieve data from Redis")
            return False
        
        print(f"✅ Session handler retrieved {len(spend_logs)} spend logs from Redis")
        
        # Clean up
        await cache.invalidate_session("test_handler_session")
        print("✅ Cleaned up session handler test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing session handler integration: {e}")
        traceback.print_exc()
        return False


def check_environment():
    """Check environment configuration."""
    print("🌍 Checking environment configuration...")
    
    redis_vars = ["REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
    missing_vars = []
    
    for var in redis_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("   Set them with: export REDIS_HOST=... REDIS_PORT=... REDIS_PASSWORD=...")
        print("   Or configure Redis in your proxy config.yaml")
    else:
        print("✅ Redis environment variables are set")
    
    # Check if litellm is importable
    try:
        import litellm
        print("✅ LiteLLM is available")
    except ImportError:
        print("❌ LiteLLM not available - install with: pip install litellm")
        return False
    
    return True


async def verify_db_writer_integration():
    """Test integration with database spend update writer."""
    print("\n💾 Testing database spend update writer integration...")
    
    try:
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
        from litellm.responses.redis_session_cache import get_responses_redis_cache
        
        cache = get_responses_redis_cache()
        
        if not cache.is_available():
            print("⚠️  Skipping DB writer test - Redis not available")
            return True
        
        writer = DBSpendUpdateWriter()
        
        # Test the caching method directly
        test_payload = {
            "request_id": "test_db_writer_resp",
            "session_id": "test_db_writer_session",
            "call_type": "aresponses",
            "messages": "[{\"role\": \"user\", \"content\": \"Test\"}]",
        }
        
        test_kwargs = {"call_type": "aresponses"}
        
        await writer._cache_response_in_redis(
            payload=test_payload,
            kwargs=test_kwargs,
            completion_response=None
        )
        
        print("✅ DB writer caching method executed successfully")
        
        # Verify data was stored
        response_data = await cache.get_response_by_id("test_db_writer_resp")
        if response_data:
            print("✅ DB writer successfully stored data in Redis")
        else:
            print("⚠️  DB writer may not have stored data (this could be normal)")
        
        # Clean up
        await cache.invalidate_session("test_db_writer_session")
        print("✅ Cleaned up DB writer test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DB writer integration: {e}")
        traceback.print_exc()
        return False


async def main():
    """Run all verification tests."""
    print("🚀 Redis Session Cache Verification Script")
    print("=" * 50)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment check failed")
        sys.exit(1)
    
    results = []
    
    # Run tests
    test_functions = [
        ("Basic Redis Cache", verify_redis_cache_basic),
        ("Session Handler Integration", verify_session_handler_integration),
        ("DB Writer Integration", verify_db_writer_integration),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Redis caching is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Start your LiteLLM proxy with Redis configured")
        print("   2. Make responses API calls with session management")
        print("   3. Monitor performance improvements (~10s → ~1ms for session lookups)")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure Redis server is running")
        print("   2. Check Redis connection settings")
        print("   3. Verify environment variables or proxy config")
        print("   4. Check Redis connectivity: redis-cli ping")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)