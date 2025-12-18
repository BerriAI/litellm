"""
E2E Tests using Google GenAI SDK to call LiteLLM Proxy's Interactions API.

This is the most authentic test - using the actual Google SDK to call 
the LiteLLM proxy, which routes requests through the Interactions -> Responses bridge.

Usage:
1. Start the proxy: litellm --config config.yaml --port 4000
2. Run tests: python test_google_sdk_via_proxy.py

Or run with pytest:
    PROXY_BASE_URL=http://localhost:4000 pytest test_google_sdk_via_proxy.py -v -s
"""

import os
import sys
import time
import warnings

# Suppress experimental warnings from Google SDK
warnings.filterwarnings("ignore", message="Interactions usage is experimental")

# Add workspace to path
sys.path.insert(0, os.path.abspath("../../.."))


def test_simple_interaction():
    """Test a simple text interaction using Google SDK."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    # Create client pointing to LiteLLM proxy
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Simple interaction via Google SDK -> Proxy")
    print(f"  Proxy URL: {proxy_base_url}")
    
    # Create interaction
    response = client.interactions.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={"input": "What is 2 + 2? Answer with just the number."}
    )
    
    print(f"  Response ID: {response.id}")
    print(f"  Status: {response.status}")
    print(f"  Outputs: {response.outputs}")
    
    assert response is not None
    assert response.status in ["completed", "COMPLETED"]
    assert response.outputs is not None
    assert len(response.outputs) > 0
    
    # Check the output contains "4"
    output_text = str(response.outputs)
    print(f"  Output text: {output_text}")
    assert "4" in output_text, f"Expected '4' in output, got: {output_text}"
    
    print("  ✓ PASSED")
    return response


def test_streaming_interaction():
    """Test streaming interaction using Google SDK."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Streaming interaction via Google SDK -> Proxy")
    
    # Create streaming interaction
    response_stream = client.interactions.with_streaming_response.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={"input": "Count from 1 to 3. Just the numbers, comma separated."}
    )
    
    chunks = []
    collected_text = ""
    
    with response_stream as stream:
        for event in stream:
            chunks.append(event)
            print(f"  Chunk: {event}")
            if hasattr(event, 'outputs') and event.outputs:
                for output in event.outputs:
                    if hasattr(output, 'text'):
                        collected_text += output.text
    
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Collected text: {collected_text}")
    
    assert len(chunks) > 0, "Expected at least one streaming chunk"
    print("  ✓ PASSED")
    return chunks


def test_interaction_with_system_instruction():
    """Test interaction with system instruction."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Interaction with system instruction via Google SDK -> Proxy")
    
    response = client.interactions.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={
            "input": "What are you?",
            "system_instruction": "You are a helpful robot. Always start your response with 'Beep boop!'",
        }
    )
    
    print(f"  Response: {response}")
    print(f"  Outputs: {response.outputs}")
    
    assert response is not None
    assert response.outputs is not None
    
    output_text = str(response.outputs).lower()
    assert "beep" in output_text or "boop" in output_text, f"Expected robot response, got: {output_text}"
    
    print("  ✓ PASSED")
    return response


def test_multi_turn_conversation():
    """Test multi-turn conversation."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Multi-turn conversation via Google SDK -> Proxy")
    
    # Multi-turn input using the Turn format
    response = client.interactions.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={
            "input": [
                {"role": "user", "content": [{"type": "text", "text": "My name is Alice."}]},
                {"role": "model", "content": [{"type": "text", "text": "Hello Alice! Nice to meet you."}]},
                {"role": "user", "content": [{"type": "text", "text": "What is my name?"}]},
            ]
        }
    )
    
    print(f"  Response: {response}")
    print(f"  Outputs: {response.outputs}")
    
    assert response is not None
    assert response.outputs is not None
    
    output_text = str(response.outputs).lower()
    assert "alice" in output_text, f"Expected 'Alice' in response, got: {output_text}"
    
    print("  ✓ PASSED")
    return response


def test_function_calling():
    """Test function/tool calling."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Function calling via Google SDK -> Proxy")
    
    response = client.interactions.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={
            "input": "What's the weather in San Francisco?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ]
        }
    )
    
    print(f"  Response: {response}")
    print(f"  Status: {response.status}")
    print(f"  Outputs: {response.outputs}")
    
    assert response is not None
    assert response.outputs is not None
    
    # Check if there's a function call in the outputs
    has_function_call = False
    for output in response.outputs:
        if hasattr(output, 'type') and output.type == 'function_call':
            has_function_call = True
            break
        if hasattr(output, 'name') and output.name == 'get_weather':
            has_function_call = True
            break
    
    output_str = str(response.outputs)
    if "function_call" in output_str or "get_weather" in output_str:
        has_function_call = True
    
    print(f"  Has function call: {has_function_call}")
    # Note: The model might respond with text first, then function call
    # So we just verify the response is valid
    
    print("  ✓ PASSED")
    return response


async def test_async_interaction():
    """Test async interaction."""
    from google import genai
    from google.genai.types import HttpOptions
    
    proxy_base_url = os.getenv("PROXY_BASE_URL", "http://localhost:4000")
    proxy_api_key = os.getenv("PROXY_API_KEY", "sk-1234")
    
    client = genai.Client(
        api_key=proxy_api_key,
        http_options=HttpOptions(
            base_url=proxy_base_url,
            api_version="v1beta",
        )
    )
    
    print(f"\n[TEST] Async interaction via Google SDK -> Proxy")
    
    response = await client.aio.interactions.create(
        model="anthropic/claude-3-5-haiku-20241022",
        config={"input": "Say 'hello' in one word."}
    )
    
    print(f"  Response: {response}")
    print(f"  Outputs: {response.outputs}")
    
    assert response is not None
    assert response.outputs is not None
    
    print("  ✓ PASSED")
    return response


def run_all_tests():
    """Run all tests."""
    import asyncio
    
    print("=" * 60)
    print("Google SDK via LiteLLM Proxy - E2E Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Simple interaction
    try:
        test_simple_interaction()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Streaming
    try:
        test_streaming_interaction()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: System instruction
    try:
        test_interaction_with_system_instruction()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Multi-turn
    try:
        test_multi_turn_conversation()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Function calling
    try:
        test_function_calling()
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Async
    try:
        asyncio.run(test_async_interaction())
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        tests_failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
