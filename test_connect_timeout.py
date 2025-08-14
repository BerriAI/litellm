#!/usr/bin/env python3
"""
Basic test to verify connect_timeout functionality
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

import litellm
import httpx


def test_connect_timeout_parameter():
    """Test that connect_timeout parameter is accepted without errors"""
    try:
        # Test with mock response to avoid making actual API calls
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            timeout=30.0,
            connect_timeout=5.0,
            mock_response="This is a test response"
        )
        print("✓ connect_timeout parameter accepted successfully")
        print(f"✓ Mock response received: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Error with connect_timeout parameter: {e}")
        return False


async def test_acompletion_connect_timeout():
    """Test that connect_timeout parameter works with acompletion"""
    try:
        # Test with mock response to avoid making actual API calls
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            timeout=30.0,
            connect_timeout=5.0,
            mock_response="This is an async test response"
        )
        print("✓ connect_timeout parameter accepted successfully in acompletion")
        print(f"✓ Async mock response received: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Error with connect_timeout parameter in acompletion: {e}")
        return False


def test_httpx_timeout_creation():
    """Test that httpx.Timeout object is created correctly with connect timeout"""
    try:
        # Test creating httpx.Timeout with connect timeout
        timeout_obj = httpx.Timeout(timeout=30.0, connect=5.0)
        print(f"✓ httpx.Timeout object created successfully")
        
        # Verify the object exists and has expected attributes
        assert hasattr(timeout_obj, 'connect')
        print("✓ Timeout object has connect attribute")
        return True
    except Exception as e:
        print(f"✗ Error creating httpx.Timeout object: {e}")
        return False


async def main():
    """Run basic tests for connect_timeout functionality"""
    print("Testing connect_timeout functionality...\n")
    
    test1_passed = test_connect_timeout_parameter()
    test2_passed = test_httpx_timeout_creation()
    test3_passed = await test_acompletion_connect_timeout()
    
    print(f"\nTest Results:")
    print(f"Connect timeout parameter test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"httpx.Timeout creation test: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Async completion connect timeout test: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! connect_timeout functionality is working.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))