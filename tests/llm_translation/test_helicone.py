import os
import sys
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


def test_completion_helicone():
    """Test basic completion through Helicone gateway"""
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="helicone/gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello from Helicone' and nothing else"}],
        max_tokens=10,
    )
    print(resp)
    assert resp.choices[0].message.content is not None
    assert len(resp.choices[0].message.content) > 0

def test_completion_helicone_specific_provider():
    """Test basic completion through Helicone gateway"""
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="helicone/claude-4.5-haiku/anthropic",
        messages=[{"role": "user", "content": "Say 'Hello from Helicone' and nothing else"}],
        max_tokens=10,
    )
    print(resp)
    assert resp.choices[0].message.content is not None
    assert len(resp.choices[0].message.content) > 0


def test_completion_helicone_streaming():
    """Test streaming completion through Helicone gateway"""
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="helicone/gpt-4o-mini",
        messages=[{"role": "user", "content": "Count to 3"}],
        max_tokens=20,
        stream=True,
    )

    chunks = []
    for chunk in resp:
        print(chunk)
        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

    full_response = "".join(chunks)
    assert len(full_response) > 0
    print(f"Full response: {full_response}")


def test_completion_helicone_with_metadata():
    """Test Helicone with custom properties"""
    litellm._turn_on_debug()
    resp = litellm.completion(
        model="helicone/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10,
        metadata={
            "Helicone-Property-Environment": "test",
            "Helicone-Property-Session": "test-session-123"
        }
    )
    print(resp)
    assert resp.choices[0].message.content is not None

