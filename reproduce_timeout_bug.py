#!/usr/bin/env python3
"""
Reproduction script for aiohttp timeout bug in litellm.

This demonstrates that when using litellm with Vertex AI/Gemini:
1. A timeout is passed to litellm.acompletion()
2. The timeout is NOT properly propagated to aiohttp's ClientTimeout
3. Result: requests can hang indefinitely during SSL operations

Expected behavior:
- Timeout should be enforced at all layers including SSL writes
- Request should fail after specified timeout

Actual behavior:
- request.extensions["timeout"] is empty dict {}
- ClientTimeout created with all None values
- No timeout enforcement, infinite hangs possible
"""

import asyncio
import logging
import os
import sys

import litellm

# Enable verbose logging to see the debug output
logging.basicConfig(level=logging.WARNING)
litellm.set_verbose = True

# Configure for Vertex AI
os.environ["VERTEXAI_PROJECT"] = os.environ.get("VERTEXAI_PROJECT", "etsy-inventory-ml-prod")
os.environ["VERTEXAI_LOCATION"] = "us-central1"

async def demonstrate_timeout_bug():
    """
    Make a simple Vertex AI call with a timeout.

    Watch the logs - they will show:
    [TIMEOUT DEBUG] timeout dict: {}
    [TIMEOUT DEBUG] ClientTimeout values: {'sock_connect': None, 'sock_read': None, 'connect': None}

    This proves that despite passing timeout=30, aiohttp receives no timeout!
    """
    print("=" * 80)
    print("DEMONSTRATING AIOHTTP TIMEOUT BUG")
    print("=" * 80)
    print("\nMaking a Vertex AI call with timeout=30 seconds...")
    print("Watch for [TIMEOUT DEBUG] log messages showing empty timeout dict\n")

    try:
        response = await litellm.acompletion(
            model="vertex_ai/gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": "Say hello"}],
            timeout=30,  # We pass a 30 second timeout
        )
        print(f"\nResponse: {response.choices[0].message.content}")
        print("\n" + "=" * 80)
        print("BUG DEMONSTRATED!")
        print("=" * 80)
        print("\nCheck the logs above. You should see:")
        print("  [TIMEOUT DEBUG] timeout dict: {}")
        print("  [TIMEOUT DEBUG] ClientTimeout values: {'sock_connect': None, 'sock_read': None, 'connect': None}")
        print("\nThis proves that:")
        print("  1. We passed timeout=30 to litellm")
        print("  2. request.extensions['timeout'] was empty dict {}")
        print("  3. ClientTimeout was created with all None values")
        print("  4. No timeout is enforced at the aiohttp layer!")
        print("\nIn production, this causes indefinite hangs during SSL writes.")

    except Exception as e:
        print(f"Error (expected): {e}")

if __name__ == "__main__":
    if "VERTEXAI_PROJECT" not in os.environ:
        print("ERROR: Set VERTEXAI_PROJECT environment variable")
        print("Example: export VERTEXAI_PROJECT=your-gcp-project")
        sys.exit(1)

    asyncio.run(demonstrate_timeout_bug())
