"""
Test script for WebSearch interception with chat completions API.

This script demonstrates how to use the websearch_interception callback
with litellm.acompletion() for transparent server-side web search execution.
"""
import asyncio
import litellm

# Enable verbose logging to see what's happening
litellm.set_verbose = True


async def test_websearch_chat_completion():
    """Test websearch interception with chat completions API."""
    
    # Configure WebSearch interception
    litellm.callbacks = ["websearch_interception"]
    
    print("\n" + "="*80)
    print("Testing WebSearch Interception with Chat Completions API")
    print("="*80 + "\n")
    
    # User makes a simple completion call with tools
    print("Making request to GPT-4o with litellm_web_search tool...")
    print("Question: What's the weather in San Francisco today?")
    print("\nExpected behavior:")
    print("1. Model calls litellm_web_search tool")
    print("2. Server executes web search automatically")
    print("3. Server makes follow-up request with search results")
    print("4. User gets final answer\n")
    
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's the weather in San Francisco today?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "litellm_web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    )
    
    print("\n" + "-"*80)
    print("FINAL RESPONSE:")
    print("-"*80)
    print(f"\nContent: {response.choices[0].message.content}")
    print(f"\nFinish reason: {response.choices[0].finish_reason}")
    
    # Check if we got tool_calls (should NOT if agentic loop worked)
    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
        print("\n⚠️  WARNING: Got tool_calls in response!")
        print("This means the agentic loop did NOT execute automatically.")
        print(f"Tool calls: {response.choices[0].message.tool_calls}")
    else:
        print("\n✅ SUCCESS: No tool_calls in response!")
        print("The agentic loop executed automatically and returned the final answer.")
    
    print("\n" + "="*80 + "\n")


async def test_streaming_websearch():
    """Test websearch interception with streaming."""
    
    # Configure WebSearch interception
    litellm.callbacks = ["websearch_interception"]
    
    print("\n" + "="*80)
    print("Testing WebSearch Interception with STREAMING")
    print("="*80 + "\n")
    
    print("Making STREAMING request to GPT-4o with litellm_web_search tool...")
    print("Question: What are the latest AI news?")
    
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What are the latest AI news from today?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "litellm_web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            }
        ],
        stream=True
    )
    
    print("\n" + "-"*80)
    print("STREAMING RESPONSE:")
    print("-"*80 + "\n")
    
    full_content = ""
    async for chunk in response:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content
    
    print("\n\n✅ Streaming completed successfully!")
    print(f"Total content length: {len(full_content)} chars")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\nWebSearch Interception Test Suite")
    print("==================================\n")
    print("This test demonstrates transparent server-side web search execution.")
    print("The agentic loop happens automatically - user just gets the final answer.\n")
    
    # Run tests
    asyncio.run(test_websearch_chat_completion())
    
    # Uncomment to test streaming
    # asyncio.run(test_streaming_websearch())
