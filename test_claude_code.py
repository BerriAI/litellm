"""
Test script for Claude Code provider in LiteLLM
"""

import litellm

# Enable verbose logging for debugging

def test_basic_completion():
    """Test basic chat completion"""
    print("\n" + "=" * 50)
    print("TEST 1: Basic Completion")
    print("=" * 50)

    try:
        response = litellm.completion(
            model="claude_code/claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Say hello in exactly 3 words"}]
        )
        print(f"✓ Response: {response.choices[0].message.content}")
        print(f"✓ Model: {response.model}")
        print(f"✓ Usage: {response.usage}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_streaming():
    """Test streaming completion"""
    print("\n" + "=" * 50)
    print("TEST 2: Streaming Completion")
    print("=" * 50)

    try:
        response = litellm.completion(
            model="claude_code/claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Count from 1 to 5, one number per line"}],
            stream=True
        )

        print("✓ Streaming response: ", end="")
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n✓ Stream completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multi_turn():
    """Test multi-turn conversation"""
    print("\n" + "=" * 50)
    print("TEST 3: Multi-turn Conversation")
    print("=" * 50)

    try:
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What's my name?"}
        ]

        response = litellm.completion(
            model="claude_code/claude-sonnet-4-5-20250929",
            messages=messages
        )
        print(f"✓ Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_system_prompt():
    """Test with system prompt"""
    print("\n" + "=" * 50)
    print("TEST 4: System Prompt")
    print("=" * 50)

    try:
        response = litellm.completion(
            model="claude_code/claude-sonnet-4-5-20250929",
            messages=[
                {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print(f"✓ Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_different_model():
    """Test with a different model"""
    print("\n" + "=" * 50)
    print("TEST 5: Different Model (Haiku)")
    print("=" * 50)

    try:
        response = litellm.completion(
            model="claude_code/claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        )
        print(f"✓ Response: {response.choices[0].message.content}")
        print(f"✓ Model: {response.model}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Claude Code Provider Test Suite")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("Basic Completion", test_basic_completion()))
    results.append(("Streaming", test_streaming()))
    results.append(("Multi-turn", test_multi_turn()))
    results.append(("System Prompt", test_system_prompt()))
    results.append(("Different Model", test_different_model()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
