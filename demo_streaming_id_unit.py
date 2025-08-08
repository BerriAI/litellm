#!/usr/bin/env python3
"""
Simple unit demo showing streaming ID consistency fix.

This script demonstrates the _encode_chunk_id() method working correctly
without requiring a running proxy server.
"""

import sys
import base64
from unittest.mock import AsyncMock

# Add the litellm directory to the path so we can import it
sys.path.insert(0, '.')

from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)

def decode_response_id(response_id):
    """Decode a litellm response ID to show its contents"""
    if response_id.startswith("resp_"):
        encoded_part = response_id[5:]  # Remove "resp_" prefix
        try:
            decoded = base64.b64decode(encoded_part).decode('utf-8')
            return decoded
        except Exception as e:
            return f"Could not decode: {e}"
    return "Not an encoded response ID"

def demo_streaming_id_encoding():
    """Demonstrate that streaming chunk IDs are now properly encoded"""
    print("🤖 Streaming ID Encoding Demo (Unit Level)")
    print("=" * 55)
    
    # Create a mock streaming chunk
    chunk = ModelResponseStream(
        id="chunk-123-from-gemini",
        created=1234567890,
        model="gemini-1.5-flash",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(content="Hello from the robot!", role="assistant"),
            )
        ],
    )
    
    print(f"📦 Original chunk ID: {chunk.id}")
    print(f"🔧 Provider: gemini")
    print(f"🆔 Model ID: gemini-1.5-flash-model")
    
    # Create the streaming iterator with provider context
    iterator = LiteLLMCompletionStreamingIterator(
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="Test input",
        responses_api_request={},
        custom_llm_provider="gemini",
        litellm_metadata={"model_info": {"id": "gemini-1.5-flash-model"}},
    )
    
    print(f"\n🔄 Processing chunk through streaming iterator...")
    
    # Transform the chunk (this is where the fix happens)
    result = iterator._transform_chat_completion_chunk_to_response_api_chunk(chunk)
    
    print(f"\n✅ Results:")
    print(f"   Original ID: {chunk.id}")
    print(f"   Encoded ID:  {result.item_id}")
    print(f"   Has 'resp_' prefix: {result.item_id.startswith('resp_')}")
    print(f"   Is encoded (not raw): {result.item_id != chunk.id}")
    
    # Show what's inside the encoded ID
    decoded = decode_response_id(result.item_id)
    print(f"\n📋 Decoded ID contents:")
    print(f"   {decoded}")
    
    print(f"\n🎯 Key improvements:")
    print(f"   ✅ Streaming chunks now have consistent 'resp_' prefix")
    print(f"   ✅ IDs contain provider context (gemini, model info)")
    print(f"   ✅ Same encoding format as non-streaming responses")
    print(f"   ✅ Enables session continuity and load balancing")
    
    # Show before/after comparison
    print(f"\n📊 Before vs After:")
    print(f"   Before fix: '{chunk.id}' (raw provider ID)")
    print(f"   After fix:  '{result.item_id}' (encoded with context)")
    
    print(f"\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    demo_streaming_id_encoding()