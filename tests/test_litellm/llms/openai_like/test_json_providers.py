"""
Tests for JSON-based provider configuration system.
"""

import os
import sys

try:
    import pytest
except ImportError:
    # pytest not available, will run as standalone script
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

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


if __name__ == "__main__":
    # Run basic tests
    print("Testing JSON Provider System...")

    test_loader = TestJSONProviderLoader()
    print("\n1. Testing JSON provider loading...")
    test_loader.test_load_json_providers()
    print("   ✓ JSON providers loaded")

    print("\n2. Testing dynamic config generation...")
    test_loader.test_dynamic_config_generation()
    print("   ✓ Dynamic config works")

    print("\n3. Testing parameter mapping...")
    test_loader.test_parameter_mapping()
    print("   ✓ Parameter mapping works")

    print("\n4. Testing excluded params...")
    test_loader.test_excluded_params()
    print("   ✓ Excluded params work")

    print("\n5. Testing provider resolution...")
    test_loader.test_provider_resolution()
    print("   ✓ Provider resolution works")

    print("\n6. Testing provider config manager...")
    test_loader.test_provider_config_manager()
    print("   ✓ Config manager works")

    print("\n" + "=" * 50)
    print("PublicAI Integration Tests...")
    print("=" * 50)

    test_integration = TestPublicAIIntegration()

    print("\n7. Testing basic completion...")
    test_integration.test_publicai_completion_basic()

    print("\n8. Testing streaming...")
    test_integration.test_publicai_completion_with_streaming()

    print("\n9. Testing parameter mapping...")
    test_integration.test_publicai_parameter_mapping()

    print("\n10. Testing content list conversion...")
    test_integration.test_publicai_content_list_conversion()

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
