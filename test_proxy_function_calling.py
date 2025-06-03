#!/usr/bin/env python3
"""
Test script to demonstrate the issue with supports_function_calling for litellm_proxy models.
"""

import litellm
import sys


def test_proxy_function_calling_support():
    """Test that function calling support works correctly for proxied models."""

    # Test direct model - should work
    print("Testing direct model...")
    try:
        direct_result = litellm.supports_function_calling("gpt-3.5-turbo")
        print(f"Direct gpt-3.5-turbo supports function calling: {direct_result}")
    except Exception as e:
        print(f"Error with direct model: {e}")

    # Test proxied model - this is where the issue occurs
    print("\nTesting proxied model...")
    try:
        proxy_result = litellm.supports_function_calling("litellm_proxy/gpt-3.5-turbo")
        print(f"Proxied gpt-3.5-turbo supports function calling: {proxy_result}")
    except Exception as e:
        print(f"Error with proxied model: {e}")

    # Test another model that supports function calling
    print("\nTesting another model...")
    try:
        claude_result = litellm.supports_function_calling("anthropic/claude-3-sonnet-20240229")
        print(f"Direct anthropic/claude-3-sonnet-20240229 supports function calling: {claude_result}")
    except Exception as e:
        print(f"Error with anthropic model: {e}")

    # Test proxied Claude
    print("\nTesting proxied Claude...")
    try:
        proxy_claude_result = litellm.supports_function_calling("litellm_proxy/anthropic/claude-3-sonnet-20240229")
        print(f"Proxied anthropic/claude-3-sonnet-20240229 supports function calling: {proxy_claude_result}")
    except Exception as e:
        print(f"Error with proxied Claude: {e}")


if __name__ == "__main__":
    test_proxy_function_calling_support()
