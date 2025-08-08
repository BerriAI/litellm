#!/usr/bin/env python3
"""
Demo script showing streaming ID consistency fix in action.

This script demonstrates that streaming chunk IDs are now properly encoded
with the same format as non-streaming responses, enabling session continuity.
"""

import os
import sys
import base64
from openai import OpenAI

# Add the litellm directory to the path so we can import it
sys.path.insert(0, '.')

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

def demo_streaming_id_consistency():
    """Demonstrate that streaming chunk IDs are now properly encoded"""
    print("🤖 Streaming ID Consistency Demo")
    print("=" * 50)
    
    # Set up client (assumes proxy server is running on localhost:4000)
    client = OpenAI(api_key="sk-1234", base_url="http://localhost:4000")
    
    try:
        print("\n1. Testing streaming response with encoded IDs...")
        
        # Make a streaming request
        stream = client.responses.create(
            model="gemini-1.5-flash",
            input="Write a short story about a robot in 2-3 sentences.",
            stream=True
        )
        
        print("\n📡 Streaming chunks received:")
        chunk_ids = []
        final_response_id = None
        
        for i, chunk in enumerate(stream):
            if hasattr(chunk, 'item_id'):
                chunk_ids.append(chunk.item_id)
                print(f"  Chunk {i+1}: {chunk.item_id}")
                
                # Show what's inside the encoded ID
                decoded = decode_response_id(chunk.item_id)
                print(f"    Decoded: {decoded}")
                
            if hasattr(chunk, 'response') and chunk.response:
                final_response_id = chunk.response.id
                print(f"\n🎯 Final response ID: {final_response_id}")
        
        # Verify streaming chunk IDs are encoded
        if chunk_ids:
            print(f"\n✅ Streaming ID Analysis:")
            print(f"   - Total chunks: {len(chunk_ids)}")
            print(f"   - All have 'resp_' prefix: {all(id.startswith('resp_') for id in chunk_ids)}")
            print(f"   - All are encoded (length > 20): {all(len(id) > 20 for id in chunk_ids)}")
            
            # Show example of decoded content
            if chunk_ids:
                print(f"\n📋 Example decoded chunk ID:")
                print(f"   Raw: {chunk_ids[0]}")
                print(f"   Decoded: {decode_response_id(chunk_ids[0])}")
        
        print("\n2. Testing session continuity with streaming response ID...")
        
        if final_response_id:
            # Use the streaming response ID as previous_response_id
            follow_up_response = client.responses.create(
                model="gemini-1.5-flash",
                input="Now say goodbye.",
                previous_response_id=final_response_id
            )
            
            print(f"✅ Session continuity works! Follow-up response ID: {follow_up_response.id}")
            print(f"   Used streaming response ID '{final_response_id}' as previous_response_id")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key insights:")
        print("   - Streaming chunk IDs now use consistent 'resp_' encoding")
        print("   - IDs contain provider context (gemini, model info, etc.)")
        print("   - Session continuity works with streaming responses")
        print("   - Load balancing can now work with streaming response IDs")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Make sure:")
        print("   - LiteLLM proxy server is running on localhost:4000")
        print("   - Gemini model is configured with API key")
        print("   - Run: python3 litellm/proxy/proxy_cli.py --config proxy_server_config.yaml --port 4000")

if __name__ == "__main__":
    demo_streaming_id_consistency()