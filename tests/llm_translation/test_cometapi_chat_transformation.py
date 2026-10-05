"""
Unit tests for CometAPI Chat Configuration

Tests the CometAPIChatConfig class methods using mocks
"""

import os

import pytest




# Integration test example (requires real API key)
@pytest.mark.skip(reason="Skipping integration test")
def test_cometapi_integration():
    """
    Integration test - requires real API key
    Run with: pytest -k test_cometapi_integration -s
    """
    from litellm import completion

    # Try to get API key from multiple environment variables
    api_key = (
        os.getenv("COMETAPI_API_KEY")
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )

    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping integration test")

    response = completion(
        model="cometapi/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_key=api_key,
        max_tokens=10,
        temperature=0.7,
    )

    # Verify response structure
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content.strip()) > 0
    assert response.model
    assert response.usage
    assert response.usage.total_tokens > 0


def test_cometapi_streaming_integration():
    """
    Integration test for streaming - requires real API key
    Run with: pytest -k test_cometapi_streaming_integration -s
    """
    from litellm import completion

    # Try to get API key from multiple environment variables
    api_key = (
        os.getenv("COMETAPI_API_KEY")
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )

    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping streaming integration test")

    try:
        print(
            f"🔍 Testing streaming with API key: {api_key[:6]}...{api_key[-4:]} (length: {len(api_key)})"
        )
        print(f"🔍 API base URL: {os.getenv('COMETAPI_API_BASE', 'default')}")

        # test streaming API call
        response = completion(
            model="cometapi/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            api_key=api_key,
            max_tokens=50,
            stream=True,
        )

        # collect streaming response
        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Verify we received at least one chunk and content
        assert len(chunks) > 0, "Should receive at least one chunk"
        assert len(content_parts) > 0, "Should receive content in chunks"

        full_content = "".join(content_parts)
        assert len(full_content.strip()) > 0, "Should have non-empty content"

        print(f"✅ Received {len(chunks)} chunks")
        print(f"✅ Full content: {full_content}")

    except Exception as e:
        print(f"❌ Streaming integration test error details:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        if hasattr(e, "status_code"):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, "response"):
            print(f"   Response: {e.response}")

        # Re-raise with more context for pytest
        pytest.fail(f"Streaming integration test failed: {type(e).__name__}: {str(e)}")


def test_cometapi_with_custom_base_url():
    """
    Test CometAPI with custom base URL
    """
    from litellm import completion

    api_key = (
        os.getenv("COMETAPI_API_KEY")
        or os.getenv("COMETAPI_KEY")
        or os.getenv("COMET_API_KEY")
    )

    custom_base_url = os.getenv("COMETAPI_API_BASE", "https://api.cometapi.com/v1")

    if not api_key:
        pytest.skip("COMETAPI_API_KEY not set - skipping custom base URL test")

    try:
        response = completion(
            model="cometapi/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base=custom_base_url,
            max_tokens=5,
        )

        assert response.choices[0].message.content
        print(f"✅ Custom base URL test passed: {response.choices[0].message.content}")

    except Exception as e:
        pytest.fail(f"Custom base URL test failed: {str(e)}")


if __name__ == "__main__":
    # Quick test runner
    pytest.main([__file__, "-v"])
