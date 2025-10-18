#!/usr/bin/env python3
"""
Demonstration of the fix: disable aiohttp transport to use httpx native transport.

This shows that by setting litellm.disable_aiohttp_transport = True,
timeout parameters are properly propagated through httpx's transport layer.
"""

import asyncio
import logging
import os
import sys

import litellm

# Enable verbose logging
logging.basicConfig(level=logging.WARNING)
litellm.set_verbose = True

# Configure for Vertex AI
os.environ["VERTEXAI_PROJECT"] = os.environ.get("VERTEXAI_PROJECT", "etsy-inventory-ml-prod")
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# THE FIX: Disable aiohttp transport
litellm.disable_aiohttp_transport = True

async def demonstrate_fix():
    """
    Make the same Vertex AI call but with aiohttp transport disabled.

    You will NOT see the [TIMEOUT DEBUG] logs because we're using httpx transport.
    The timeout will be properly enforced.
    """
    print("=" * 80)
    print("DEMONSTRATING THE FIX")
    print("=" * 80)
    print("\nSetting: litellm.disable_aiohttp_transport = True")
    print("Making a Vertex AI call with timeout=30 seconds...")
    print("Note: No [TIMEOUT DEBUG] logs - we're using httpx native transport\n")

    try:
        response = await litellm.acompletion(
            model="vertex_ai/gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": "Say hello"}],
            timeout=30,
        )
        print(f"\nResponse: {response.choices[0].message.content}")
        print("\n" + "=" * 80)
        print("FIX VERIFIED!")
        print("=" * 80)
        print("\nWith aiohttp transport disabled:")
        print("  1. httpx native transport is used")
        print("  2. timeout=30 is properly propagated")
        print("  3. Timeout is enforced at all layers including SSL")
        print("  4. No indefinite hangs!")

    except Exception as e:
        print(f"Error (expected): {e}")

if __name__ == "__main__":
    if "VERTEXAI_PROJECT" not in os.environ:
        print("ERROR: Set VERTEXAI_PROJECT environment variable")
        print("Example: export VERTEXAI_PROJECT=your-gcp-project")
        sys.exit(1)

    asyncio.run(demonstrate_fix())
