#!/usr/bin/env python3
"""
E2E Tests using Google SDK format with litellm.interactions API.

This tests the Interactions -> Responses bridge by using Google SDK-style
request formats directly with the litellm.interactions API.

The Google GenAI SDK sends requests in a specific format to the /interactions endpoint.
This test validates that our bridge correctly handles these requests when routed
through the litellm.interactions.create() function.

Run with: 
    ANTHROPIC_API_KEY=<key> python test_google_sdk_format.py
or:
    ANTHROPIC_API_KEY=<key> pytest test_google_sdk_format.py -v -s
"""

import asyncio
import os
import sys
import warnings

# Suppress experimental warnings
warnings.filterwarnings("ignore")

# Add workspace to path BEFORE any litellm imports
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, workspace_path)

# Now import litellm from workspace
import litellm
interactions = litellm.interactions


def get_api_key():
    """Get API key from environment."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return key


class TestGoogleSDKFormatViaInteractionsBridge:
    """
    Tests that validate the Interactions -> Responses bridge works correctly
    when receiving Google SDK-style requests.
    """
    
    def test_simple_string_input(self):
        """Test simple string input - most basic Google SDK usage."""
        api_key = get_api_key()
        
        print("\n[TEST] Simple string input (Google SDK style)")
        
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What is 2 + 2? Answer with just the number.",
            api_key=api_key,
        )
        
        print(f"  Response ID: {response.id}")
        print(f"  Status: {response.status}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.status == "completed"
        assert response.outputs is not None
        assert len(response.outputs) > 0
        
        # Check output contains "4"
        output_text = str(response.outputs)
        assert "4" in output_text, f"Expected '4' in output, got: {output_text}"
        
        print("  ✓ PASSED")
        return response
    
    def test_turn_format_input(self):
        """Test Turn[] format input - Google SDK multi-turn style."""
        api_key = get_api_key()
        
        print("\n[TEST] Turn[] format input (Google SDK style)")
        
        # Google SDK sends multi-turn as list of Turn objects
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "My name is Charlie."}]
                },
                {
                    "role": "model",  # Google uses "model" not "assistant"
                    "content": [{"type": "text", "text": "Hello Charlie! Nice to meet you."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is my name?"}]
                }
            ],
            api_key=api_key,
        )
        
        print(f"  Response: {response}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.outputs is not None
        
        output_text = str(response.outputs).lower()
        assert "charlie" in output_text, f"Expected 'Charlie' in response, got: {output_text}"
        
        print("  ✓ PASSED")
        return response
    
    def test_with_system_instruction(self):
        """Test with system_instruction - Google SDK style."""
        api_key = get_api_key()
        
        print("\n[TEST] System instruction (Google SDK style)")
        
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What are you?",
            system_instruction="You are a helpful robot. Always start responses with 'Beep boop!'",
            api_key=api_key,
        )
        
        print(f"  Response: {response}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.outputs is not None
        
        output_text = str(response.outputs).lower()
        assert "beep" in output_text or "boop" in output_text, f"Expected robot response, got: {output_text}"
        
        print("  ✓ PASSED")
        return response
    
    def test_with_generation_config(self):
        """Test with generation_config - Google SDK style parameters."""
        api_key = get_api_key()
        
        print("\n[TEST] Generation config (Google SDK style)")
        
        # Google SDK uses generation_config for parameters
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Say hello in exactly 3 words.",
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 50,
            },
            api_key=api_key,
        )
        
        print(f"  Response: {response}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.outputs is not None
        
        print("  ✓ PASSED")
        return response
    
    def test_streaming(self):
        """Test streaming - Google SDK style."""
        api_key = get_api_key()
        
        print("\n[TEST] Streaming (Google SDK style)")
        
        response_stream = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Count from 1 to 3. Just numbers, comma separated.",
            stream=True,
            api_key=api_key,
        )
        
        chunks = []
        event_types = []
        
        for chunk in response_stream:
            chunks.append(chunk)
            if hasattr(chunk, 'event_type'):
                event_types.append(chunk.event_type)
            print(f"  Chunk: {chunk}")
        
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Event types: {event_types}")
        
        assert len(chunks) > 0, "Expected streaming chunks"
        
        # Verify we got the expected event types
        assert "interaction.start" in event_types, "Expected interaction.start event"
        assert "interaction.complete" in event_types, "Expected interaction.complete event"
        
        print("  ✓ PASSED")
        return chunks
    
    def test_function_calling(self):
        """Test function/tool calling - Google SDK style."""
        api_key = get_api_key()
        
        print("\n[TEST] Function calling (Google SDK style)")
        
        # Google SDK uses 'tools' with 'function' type
        response = interactions.create(
            model="anthropic/claude-3-5-haiku-20241022",
            input="What's the weather in Tokyo?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            ],
            api_key=api_key,
        )
        
        print(f"  Response: {response}")
        print(f"  Status: {response.status}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.outputs is not None
        
        # Check for function call in outputs
        output_str = str(response.outputs)
        has_function_call = "function_call" in output_str or "get_weather" in output_str
        print(f"  Has function call: {has_function_call}")
        
        print("  ✓ PASSED")
        return response
    
    async def test_async_interaction(self):
        """Test async interaction."""
        api_key = get_api_key()
        
        print("\n[TEST] Async interaction (Google SDK style)")
        
        response = await interactions.acreate(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Say 'hello' in one word.",
            api_key=api_key,
        )
        
        print(f"  Response: {response}")
        print(f"  Outputs: {response.outputs}")
        
        assert response is not None
        assert response.outputs is not None
        
        print("  ✓ PASSED")
        return response
    
    async def test_async_streaming(self):
        """Test async streaming."""
        api_key = get_api_key()
        
        print("\n[TEST] Async streaming (Google SDK style)")
        
        response_stream = await interactions.acreate(
            model="anthropic/claude-3-5-haiku-20241022",
            input="Count: 1, 2, 3",
            stream=True,
            api_key=api_key,
        )
        
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
            print(f"  Async chunk: {chunk}")
        
        print(f"  Total async chunks: {len(chunks)}")
        
        assert len(chunks) > 0, "Expected async streaming chunks"
        
        print("  ✓ PASSED")
        return chunks


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Google SDK Format via Interactions Bridge - E2E Tests")
    print("=" * 70)
    
    tests = TestGoogleSDKFormatViaInteractionsBridge()
    
    tests_passed = 0
    tests_failed = 0
    
    # Sync tests
    sync_tests = [
        ("Simple string input", tests.test_simple_string_input),
        ("Turn[] format input", tests.test_turn_format_input),
        ("System instruction", tests.test_with_system_instruction),
        ("Generation config", tests.test_with_generation_config),
        ("Streaming", tests.test_streaming),
        ("Function calling", tests.test_function_calling),
    ]
    
    for name, test_fn in sync_tests:
        try:
            test_fn()
            tests_passed += 1
        except Exception as e:
            print(f"  ✗ FAILED ({name}): {e}")
            import traceback
            traceback.print_exc()
            tests_failed += 1
    
    # Async tests
    async def run_async_tests():
        nonlocal tests_passed, tests_failed
        
        async_tests = [
            ("Async interaction", tests.test_async_interaction),
            ("Async streaming", tests.test_async_streaming),
        ]
        
        for name, test_fn in async_tests:
            try:
                await test_fn()
                tests_passed += 1
            except Exception as e:
                print(f"  ✗ FAILED ({name}): {e}")
                import traceback
                traceback.print_exc()
                tests_failed += 1
    
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 70)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 70)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
