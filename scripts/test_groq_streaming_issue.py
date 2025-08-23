"""
Test script to reproduce the Groq streaming ASCII encoding issue.

This reproduces the issue described in #12660 where streaming responses
containing non-ASCII characters like µ cause encoding errors.
"""
import asyncio
import os
import traceback
from litellm import acompletion

async def test_groq_streaming_with_special_chars():
    """Test that reproduces the ASCII encoding issue with Groq streaming."""
    try:
        print("Testing acompletion + streaming with Groq...")
        
        # Test message that should trigger the µ character or similar non-ASCII content
        test_messages = [
            {"content": "What is the symbol for micro? Please include the µ symbol in your response.", "role": "user"}
        ]
        
        # This should trigger the ASCII encoding error described in the issue
        response = await acompletion(
            model="groq/llama-3.3-70b-versatile",
            messages=test_messages, 
            stream=True
        )
        
        print(f"Response type: {type(response)}")
        
        # Try to iterate through the stream
        async for chunk in response:
            print(f"Chunk: {chunk}")
            
        print("✅ Test completed successfully - no encoding errors!")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"Error type: {type(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return False
        
    return True

if __name__ == "__main__":
    # Note: This requires GROQ_API_KEY to be set
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  GROQ_API_KEY not set. Skipping test.")
    else:
        success = asyncio.run(test_groq_streaming_with_special_chars())
        if success:
            print("🎉 All tests passed!")
        else:
            print("💥 Test failed!")