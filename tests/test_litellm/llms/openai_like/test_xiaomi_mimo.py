"""
Tests for Xiaomi MiMo provider configuration and integration.
Related to issue #18794
"""

import os

import pytest

import litellm


class TestXiaomiMiMoIntegration:
    """Integration tests for Xiaomi MiMo provider"""

    def test_xiaomi_mimo_completion_basic(self):
        """Test basic completion call to Xiaomi MiMo"""
        # Skip test if API key not set in environment
        if not os.environ.get("XIAOMI_MIMO_API_KEY"):
            if pytest:
                pytest.skip("XIAOMI_MIMO_API_KEY not set")
            return

        try:
            response = litellm.completion(
                model="xiaomi_mimo/mimo-v2-flash",
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
                f"✓ Xiaomi MiMo completion successful: {response.choices[0].message.content}"
            )

        except Exception as e:
            if pytest:
                pytest.fail(f"Xiaomi MiMo completion failed: {str(e)}")
            else:
                raise
