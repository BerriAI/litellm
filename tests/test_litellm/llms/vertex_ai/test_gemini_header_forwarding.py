"""
Test to verify that custom headers are correctly forwarded to Gemini/Vertex AI API calls.

This test verifies the fix for the issue where headers configured via
forward_client_headers_to_llm_api were not being passed to Gemini/Vertex AI providers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import litellm
from litellm import completion


class TestGeminiHeaderForwarding:
    """Test cases for verifying header forwarding to Gemini/Vertex AI."""

    @pytest.mark.parametrize(
        "custom_llm_provider,model",
        [
            ("gemini", "gemini/gemini-1.5-pro"),
            ("vertex_ai_beta", "gemini-1.5-pro"),
            ("vertex_ai", "gemini-1.5-pro"),
        ],
    )
    def test_headers_forwarded_to_gemini(self, custom_llm_provider, model):
        """
        Test that headers from kwargs are correctly merged and passed to Gemini completion.
        
        This test verifies that when headers are passed via kwargs (as the proxy does when
        forward_client_headers_to_llm_api is configured), they are correctly merged with
        extra_headers and passed to the Vertex AI completion handler.
        """
        messages = [{"role": "user", "content": "Hello"}]
        
        # Headers that would be set by the proxy when forwarding client headers
        custom_headers = {
            "X-Custom-Header": "CustomValue",
            "X-BYOK-Token": "secret-token",
        }
        
        # Mock the vertex completion handler
        with patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM.completion"
        ) as mock_vertex_completion:
            # Configure the mock to return a proper response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Hello back!"
            mock_vertex_completion.return_value = mock_response
            
            try:
                # Call completion with custom headers via kwargs
                # This simulates what the proxy does when forward_client_headers_to_llm_api is set
                completion(
                    model=model,
                    messages=messages,
                    headers=custom_headers,  # This is how proxy passes forwarded headers
                    custom_llm_provider=custom_llm_provider,
                    api_key="dummy-key",
                )
                
                # Verify that the completion handler was called
                assert mock_vertex_completion.called, "Vertex completion handler should be called"
                
                # Get the actual call arguments
                call_kwargs = mock_vertex_completion.call_args.kwargs
                
                # Verify that extra_headers parameter contains our custom headers
                assert "extra_headers" in call_kwargs, "extra_headers should be passed to completion"
                
                passed_headers = call_kwargs["extra_headers"]
                assert passed_headers is not None, "extra_headers should not be None"
                
                # Verify our custom headers are present in the passed headers
                for header_key, header_value in custom_headers.items():
                    assert (
                        header_key in passed_headers
                        or header_key.lower() in passed_headers
                    ), f"Header {header_key} should be in extra_headers"
                    
                print(f"✓ Test passed for {custom_llm_provider}/{model}")
                print(f"  Headers correctly forwarded: {passed_headers}")
                
            except Exception as e:
                pytest.fail(
                    f"Failed to forward headers to {custom_llm_provider}/{model}: {str(e)}"
                )

    def test_extra_headers_and_headers_merge(self):
        """
        Test that both extra_headers and headers parameters are correctly merged.
        
        This ensures that headers from kwargs (forwarded by proxy) and extra_headers
        (passed explicitly) are both included in the final headers sent to the provider.
        """
        messages = [{"role": "user", "content": "Hello"}]
        
        # Headers from proxy (via kwargs["headers"])
        proxy_headers = {"X-Forwarded-Header": "ProxyValue"}
        
        # Explicit extra_headers
        explicit_headers = {"X-Explicit-Header": "ExplicitValue"}
        
        with patch(
            "litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini.VertexLLM.completion"
        ) as mock_vertex_completion:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Response"
            mock_vertex_completion.return_value = mock_response
            
            try:
                completion(
                    model="gemini/gemini-1.5-pro",
                    messages=messages,
                    headers=proxy_headers,  # From proxy forwarding
                    extra_headers=explicit_headers,  # Explicitly passed
                    custom_llm_provider="gemini",
                    api_key="dummy-key",
                )
                
                call_kwargs = mock_vertex_completion.call_args.kwargs
                passed_headers = call_kwargs.get("extra_headers", {})
                
                # Both sets of headers should be present
                assert (
                    "X-Forwarded-Header" in passed_headers
                    or "x-forwarded-header" in passed_headers
                ), "Proxy forwarded header should be present"
                
                assert (
                    "X-Explicit-Header" in passed_headers
                    or "x-explicit-header" in passed_headers
                ), "Explicitly passed header should be present"
                
                print("✓ Both header sources correctly merged and forwarded")
                print(f"  Final headers: {passed_headers}")
                
            except Exception as e:
                pytest.fail(f"Failed to merge and forward headers: {str(e)}")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestGeminiHeaderForwarding()
    
    print("\n" + "="*80)
    print("Testing Gemini/Vertex AI Header Forwarding")
    print("="*80 + "\n")
    
    # Test each provider
    for provider, model in [
        ("gemini", "gemini/gemini-1.5-pro"),
        ("vertex_ai_beta", "gemini-1.5-pro"),
        ("vertex_ai", "gemini-1.5-pro"),
    ]:
        print(f"\nTesting {provider}/{model}...")
        try:
            test_instance.test_headers_forwarded_to_gemini(provider, model)
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print("\n\nTesting header merging...")
    try:
        test_instance.test_extra_headers_and_headers_merge()
    except Exception as e:
        print(f"✗ Test failed: {e}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80 + "\n")

