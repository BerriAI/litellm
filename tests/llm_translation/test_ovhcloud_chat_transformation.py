"""
Unit tests for OVHCloud AI Endpoints chat integration.
"""

import os

import pytest




model = "ovhcloud/Mistral-7B-Instruct-v0.3"


def test_ovhcloud_integration():
    from litellm import completion

    api_key = os.getenv("OVHCLOUD_API_KEY")

    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")

    response = completion(
        model,
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_key=api_key,
        max_tokens=10,
        temperature=0.7,
    )

    assert response.choices[0].message.content
    assert len(response.choices[0].message.content.strip()) > 0
    assert response.model
    assert response.usage
    assert response.usage.total_tokens > 0


def test_OVHCloud_streaming_integration():
    """
    Integration test for streaming - requires real API key
    Run with: pytest -k test_OVHCloud_streaming_integration -s
    """
    from litellm import completion

    api_key = os.getenv("OVHCLOUD_API_KEY")

    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")

    try:
        print(
            f"🔍 Testing streaming with API key: {api_key[:6]}...{api_key[-4:]} (length: {len(api_key)})"
        )
        print(f"🔍 API base URL: {os.getenv('OVHCLOUD_API_BASE')}")

        response = completion(
            model,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            api_key=api_key,
            max_tokens=50,
            stream=True,
        )

        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

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

        pytest.fail(f"Streaming integration test failed: {type(e).__name__}: {str(e)}")


def test_ovhcloud_with_custom_base_url():
    """
    Test OVHCloud with custom base URL
    """
    from litellm import completion

    api_key = os.getenv("OVHCLOUD_API_KEY")

    if not api_key:
        pytest.skip("OVHCLOUD_API_KEY not set, skipping test")

    custom_base_url = os.getenv(
        "OVHCLOUD_API_BASE", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    )

    try:
        response = completion(
            model,
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
    pytest.main([__file__, "-v"])
