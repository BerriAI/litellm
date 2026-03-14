"""
Live test: Perplexity Responses API via LiteLLM.
Tests: non-streaming, streaming, preset models, models fallback param,
chat completions (regression check), and cost dict→float parsing.

DO NOT COMMIT this file.
"""

import os
import traceback

from dotenv import load_dotenv

load_dotenv()

import litellm

# litellm.set_verbose = True


def test_non_streaming_preset():
    """Test non-streaming with preset model."""
    print("=" * 60)
    print("TEST 1: Non-streaming preset/pro-search")
    print("=" * 60)

    response = litellm.responses(
        model="perplexity/preset/pro-search",
        input="What is 2 + 2? Answer in one word.",
    )

    print(f"  Response ID: {response.id}")
    print(f"  Model: {response.model}")
    print(f"  Status: {response.status}")

    assert response.status == "completed", f"FAIL: status={response.status}"
    assert response.output, "FAIL: no output"
    print("  PASS: non-streaming preset works")

    if response.usage and response.usage.cost is not None:
        assert isinstance(response.usage.cost, (int, float)), (
            f"FAIL: cost is {type(response.usage.cost)}: {response.usage.cost}"
        )
        print(f"  PASS: cost={response.usage.cost} (float, not dict)")
    print()


def test_streaming_preset():
    """Test streaming with preset model."""
    print("=" * 60)
    print("TEST 2: Streaming preset/pro-search")
    print("=" * 60)

    response = litellm.responses(
        model="perplexity/preset/pro-search",
        input="What is the capital of France? One word.",
        stream=True,
    )

    chunks = 0
    completed = False
    for chunk in response:
        chunks += 1
        event_type = getattr(chunk, "type", "unknown")
        if event_type == "response.output_text.delta":
            print(f"  delta: {chunk.delta}", end="", flush=True)
        elif event_type == "response.completed":
            completed = True
            print(f"\n  [completed] model={chunk.response.model}")
            if chunk.response.usage and chunk.response.usage.cost is not None:
                cost = chunk.response.usage.cost
                assert isinstance(cost, (int, float)), (
                    f"FAIL: streaming cost is {type(cost)}: {cost}"
                )
                print(f"  PASS: streaming cost={cost} (float)")

    assert chunks > 0, "FAIL: no chunks received"
    assert completed, "FAIL: never got response.completed event"
    print(f"  Total chunks: {chunks}")
    print("  PASS: streaming preset works")
    print()


def test_models_fallback_param():
    """Test that 'models' param (Perplexity fallback chain) is forwarded."""
    print("=" * 60)
    print("TEST 3: models param (fallback chain)")
    print("=" * 60)

    response = litellm.responses(
        model="perplexity/openai/gpt-5.1",
        input="Say 'hello' and nothing else.",
        models=["openai/gpt-5-mini", "openai/gpt-5.1"],
    )

    print(f"  Response ID: {response.id}")
    print(f"  Model used: {response.model}")
    print(f"  Status: {response.status}")

    assert response.status == "completed", f"FAIL: status={response.status}"
    print("  PASS: models fallback param works")
    print()


def test_chat_completions_not_broken():
    """Regression: Perplexity chat completions must still use PerplexityChatConfig."""
    print("=" * 60)
    print("TEST 4: Chat completions regression check")
    print("=" * 60)

    response = litellm.completion(
        model="perplexity/sonar",
        messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
        max_tokens=10,
    )

    print(f"  Model: {response.model}")
    print(f"  Content: {response.choices[0].message.content[:50]}")

    assert response.choices, "FAIL: no choices"
    assert response.choices[0].message.content, "FAIL: empty content"
    print("  PASS: chat completions still work (no regression)")
    print()


def test_with_instructions():
    """Test instructions param."""
    print("=" * 60)
    print("TEST 5: instructions param")
    print("=" * 60)

    response = litellm.responses(
        model="perplexity/preset/pro-search",
        input="What is Python?",
        instructions="Answer in exactly 5 words.",
    )

    print(f"  Status: {response.status}")
    # Extract text from output
    for item in response.output:
        if hasattr(item, "content"):
            for c in item.content:
                if hasattr(c, "text"):
                    print(f"  Answer: {c.text}")
                    break

    assert response.status == "completed", f"FAIL: status={response.status}"
    print("  PASS: instructions param works")
    print()


if __name__ == "__main__":
    api_key = os.environ.get("PERPLEXITYAI_API_KEY", "NOT SET")
    print(f"Using PERPLEXITYAI_API_KEY: {api_key[:10]}...")
    print()

    tests = [
        test_non_streaming_preset,
        test_streaming_preset,
        test_models_fallback_param,
        test_chat_completions_not_broken,
        test_with_instructions,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
