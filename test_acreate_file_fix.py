#!/usr/bin/env python3
"""
Test script to verify the acreate_file RuntimeWarning fix.
This reproduces the issue described in GitHub issue #20798.
"""
import asyncio
import warnings
import os
import tempfile

# Capture warnings
warnings.simplefilter("always")

# Add the project to path
import sys
sys.path.insert(0, '/tmp/oss-litellm')

import litellm
from litellm.utils import _is_async_request


async def test_acreate_file_async_detection():
    """Test that _is_async_request properly detects acreate_file calls."""
    print("Testing _is_async_request function...")
    
    # Test kwargs similar to what acreate_file sets
    test_kwargs = {"acreate_file": True}
    result = _is_async_request(test_kwargs)
    
    print(f"_is_async_request with acreate_file=True returned: {result}")
    assert result is True, "Expected _is_async_request to return True for acreate_file=True"
    print("✅ _is_async_request correctly detects acreate_file!")


async def test_acreate_file_mock():
    """Test acreate_file to ensure no runtime warnings are generated."""
    print("\nTesting acreate_file function...")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Hello, World! This is a test file.")
        temp_file_path = f.name
    
    try:
        # Note: This will fail because we don't have valid API keys
        # but it should NOT generate a RuntimeWarning about coroutines
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                result = await litellm.acreate_file(
                    file=open(temp_file_path, "rb"),
                    purpose="assistants",
                    api_key="test-key",  # Mock key
                )
        except Exception as e:
            print(f"Expected exception (no valid API key): {type(e).__name__}")
            
            # Check if any RuntimeWarnings about coroutines were captured
            runtime_warnings = [warn for warn in w if issubclass(warn.category, RuntimeWarning)]
            coroutine_warnings = [warn for warn in runtime_warnings if "coroutine" in str(warn.message)]
            
            if coroutine_warnings:
                print("❌ Found RuntimeWarnings about coroutines:")
                for warn in coroutine_warnings:
                    print(f"  {warn.message}")
                assert False, "RuntimeWarning about coroutines found!"
            else:
                print("✅ No RuntimeWarning about coroutines detected!")
                
    finally:
        # Clean up temp file
        os.unlink(temp_file_path)


async def main():
    print("Testing GitHub Issue #20798 fix: acreate_file RuntimeWarning")
    print("=" * 60)
    
    await test_acreate_file_async_detection()
    await test_acreate_file_mock()
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The fix appears to be working correctly.")


if __name__ == "__main__":
    asyncio.run(main())