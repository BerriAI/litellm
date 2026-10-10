"""
Tests for JSON-based provider configuration system.
"""

import os

import pytest

import litellm


class TestPublicAIIntegration:
    """Integration tests for PublicAI provider"""

    def test_publicai_completion_basic(self):
        """Test basic completion call to PublicAI"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": "Say 'test successful' and nothing else",
                    }
                ],
                max_tokens=10,
            )

            # Verify response structure
            assert response is not None
            assert hasattr(response, "choices")
            assert len(response.choices) > 0
            assert hasattr(response.choices[0], "message")
            assert hasattr(response.choices[0].message, "content")
            assert response.choices[0].message.content is not None

            # Check that we got a response
            content = response.choices[0].message.content.lower()
            assert len(content) > 0

            print(
                f"✓ PublicAI completion successful: {response.choices[0].message.content}"
            )

        except Exception as e:
            if pytest:
                pytest.fail(f"PublicAI completion failed: {str(e)}")
            else:
                raise

    def test_publicai_completion_with_streaming(self):
        """Test streaming completion with PublicAI"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[{"role": "user", "content": "Count to 3"}],
                max_tokens=20,
                stream=True,
            )

            # Collect chunks
            chunks = []
            for chunk in response:
                assert chunk is not None
                if hasattr(chunk.choices[0], "delta") and hasattr(
                    chunk.choices[0].delta, "content"
                ):
                    if chunk.choices[0].delta.content:
                        chunks.append(chunk.choices[0].delta.content)

            # Verify we got chunks
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0

            print(f"✓ PublicAI streaming successful: {full_response}")

        except Exception as e:
            if pytest:
                pytest.fail(f"PublicAI streaming failed: {str(e)}")
            else:
                raise

    def test_publicai_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            # Use max_completion_tokens (OpenAI's newer parameter)
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=5,  # This should be mapped to max_tokens
            )

            assert response is not None
            assert len(response.choices) > 0

            print("✓ Parameter mapping successful")

        except Exception as e:
            if pytest:
                pytest.fail(f"Parameter mapping test failed: {str(e)}")
            else:
                raise

    def test_publicai_content_list_conversion(self):
        """Test that content list format is converted to string"""
        # Skip test if API key not set in environment
        if not os.environ.get("PUBLICAI_API_KEY"):
            if pytest:
                pytest.skip("PUBLICAI_API_KEY not set")
            return

        try:
            # Send message with content as list (should be converted to string)
            response = litellm.completion(
                model="publicai/swiss-ai/apertus-8b-instruct",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Say hello"}]}
                ],
                max_tokens=10,
            )

            assert response is not None
            assert len(response.choices) > 0

            print("✓ Content list conversion successful")

        except Exception as e:
            if pytest:
                pytest.fail(f"Content list conversion test failed: {str(e)}")
            else:
                raise
