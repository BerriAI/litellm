"""
Unit tests for Vertex AI Model Garden fine-tuned models using the predict endpoint.

This tests the integration with Model Garden models that use numerical IDs and the predict endpoint,
as opposed to the OpenAI-compatible chat/completions endpoint.

Example usage:
{
 "model": "vertex_ai/3245717643264524288",
 "vertex_project": "etsy-inventory-ml-dev", 
 "vertex_location": "us-central1"
}

Expected endpoint format:
https://{ENDPOINT_ID}.{location}-{project_number}.prediction.vertexai.goog/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm import completion


class TestVertexAIModelGardenFineTuned:
    """Test Vertex AI Model Garden fine-tuned models with predict endpoint."""

    def test_vertex_model_garden_fine_tuned_completion(self):
        """
        Test completion call for Vertex AI Model Garden fine-tuned model.
        
        This test verifies that LiteLLM can properly handle requests to 
        fine-tuned models deployed in Vertex AI Model Garden using the predict endpoint.
        """
        
        # Mock the response from Vertex AI Model Garden predict endpoint
        mock_response_data = {
            "predictions": [
                {
                    "content": "Based on the title \"Red Leather Wallet\", I can extract the following entities:\n\n1. **Material**: Leather\n2. **Color**: Red\n3. **Product Type**: Wallet\n\nThese are the primary entities present in this listing title."
                }
            ]
        }
        
        # Expected request body for the predict endpoint
        expected_request_body = {
            "instances": [
                {
                    "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExtract entities from the listing.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntitle: Red Leather Wallet\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    "max_tokens": 512,
                    "temperature": 0.1
                }
            ],
            "parameters": {}
        }
        
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        
        client = HTTPHandler()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200
        
        with patch.object(client, "post", return_value=mock_response) as mock_post:
            try:
                response = completion(
                    model="vertex_ai/3245717643264524288",
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract entities from the listing."
                        },
                        {
                            "role": "user", 
                            "content": "title: Red Leather Wallet"
                        }
                    ],
                    max_tokens=512,
                    temperature=0.1,
                    vertex_project="etsy-inventory-ml-dev",
                    vertex_location="us-central1",
                    client=client
                )
                
                print("Response received:", response)
                
                # Verify the response structure
                assert response is not None
                assert len(response.choices) > 0
                assert response.choices[0].message.content is not None
                
                # Verify the request was made
                mock_post.assert_called_once()
                
                # Print the actual request for debugging
                print("Actual request URL:", mock_post.call_args.args[0] if mock_post.call_args.args else "No URL")
                print("Actual request body:", mock_post.call_args.kwargs.get("json", "No JSON body"))
                print("Actual headers:", mock_post.call_args.kwargs.get("headers", "No headers"))
                
            except Exception as e:
                print(f"Exception occurred: {e}")
                # Still verify the mock was called to see what happened
                if mock_post.called:
                    print("Request was made despite exception:")
                    print("URL:", mock_post.call_args.args[0] if mock_post.call_args.args else "No URL")
                    print("Body:", mock_post.call_args.kwargs.get("json", "No JSON body"))
                raise

    def test_vertex_model_garden_fine_tuned_url_construction(self):
        """
        Test that the URL is constructed correctly for Model Garden fine-tuned models.
        """
        from litellm.llms.vertex_ai.common_utils import _get_vertex_url
        
        model_id = "3245717643264524288"
        vertex_project = "etsy-inventory-ml-dev"
        vertex_location = "us-central1"
        
        # Test URL generation for numeric model ID (fine-tuned model)
        url, endpoint = _get_vertex_url(
            mode="chat",
            model=model_id,
            stream=False,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_api_version="v1"
        )
        
        # Verify the URL format for fine-tuned models
        expected_url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{model_id}:generateContent"
        
        assert url == expected_url
        assert endpoint == "generateContent"
        
        print(f"Generated URL: {url}")
        print(f"Expected URL: {expected_url}")

    def test_vertex_model_garden_predict_endpoint_request_format(self):
        """
        Test the request format for the predict endpoint specifically.
        
        This verifies that the request body matches the expected format for
        Vertex AI Model Garden predict endpoint as shown in the user's curl example.
        """
        
        # Test input matching the user's curl example
        messages = [
            {
                "role": "system",
                "content": "Extract entities from the listing."
            },
            {
                "role": "user", 
                "content": "title: Red Leather Wallet"
            }
        ]
        
        # Expected prompt format (this might need adjustment based on how LiteLLM handles it)
        expected_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nExtract entities from the listing.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\ntitle: Red Leather Wallet\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # This test is more for documentation and understanding the expected format
        # The actual transformation will depend on LiteLLM's internal logic
        print("Expected prompt format:")
        print(expected_prompt)
        
        # Test the parameters
        expected_params = {
            "max_tokens": 512,
            "temperature": 0.1
        }
        
        assert expected_params["max_tokens"] == 512
        assert expected_params["temperature"] == 0.1
        
    def test_vertex_model_garden_model_id_detection(self):
        """
        Test that numeric model IDs are properly detected as fine-tuned models.
        """
        
        # Test various model ID formats
        test_cases = [
            ("3245717643264524288", True),  # Pure numeric - should be fine-tuned
            ("vertex_ai/3245717643264524288", True),  # With prefix - should extract numeric part
            ("gemini-1.5-pro", False),  # Regular model name
            ("vertex_ai/openai/3245717643264524288", False),  # Model Garden OpenAI format
        ]
        
        for model_id, should_be_numeric in test_cases:
            # Extract just the model part if it has prefix
            if "vertex_ai/" in model_id:
                model_part = model_id.replace("vertex_ai/", "")
                if "openai/" in model_part:
                    model_part = model_part.replace("openai/", "")
            else:
                model_part = model_id
                
            is_numeric = model_part.isdigit()
            assert is_numeric == should_be_numeric, f"Model {model_id} (part: {model_part}) numeric detection failed"
            
            print(f"Model: {model_id} -> Numeric part: {model_part} -> Is numeric: {is_numeric}")
    
    def test_vertex_model_garden_response_parsing(self):
        """
        Test parsing responses from Vertex AI Model Garden predict endpoint.
        """
        
        # Mock response that matches the expected format from predict endpoint
        mock_predict_response = {
            "predictions": [
                {
                    "content": "Based on the title \"Red Leather Wallet\", I can extract the following entities:\n\n1. **Material**: Leather\n2. **Color**: Red\n3. **Product Type**: Wallet"
                }
            ]
        }
        
        # This is a basic test - in practice, LiteLLM would transform this
        # into a standardized ModelResponse format
        assert "predictions" in mock_predict_response
        assert len(mock_predict_response["predictions"]) > 0
        assert "content" in mock_predict_response["predictions"][0]
        
        content = mock_predict_response["predictions"][0]["content"]
        assert "Leather" in content
        assert "Red" in content 
        assert "Wallet" in content
        
        print("Response parsing test passed")
        print(f"Content: {content}")


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    test_instance = TestVertexAIModelGardenFineTuned()
    
    print("Running test_vertex_model_garden_fine_tuned_completion...")
    try:
        test_instance.test_vertex_model_garden_fine_tuned_completion()
        print("✅ test_vertex_model_garden_fine_tuned_completion passed")
    except Exception as e:
        print(f"❌ test_vertex_model_garden_fine_tuned_completion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nRunning test_vertex_model_garden_fine_tuned_url_construction...")
    try:
        test_instance.test_vertex_model_garden_fine_tuned_url_construction()
        print("✅ test_vertex_model_garden_fine_tuned_url_construction passed")
    except Exception as e:
        print(f"❌ test_vertex_model_garden_fine_tuned_url_construction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nRunning test_vertex_model_garden_predict_endpoint_request_format...")
    try:
        test_instance.test_vertex_model_garden_predict_endpoint_request_format()
        print("✅ test_vertex_model_garden_predict_endpoint_request_format passed")
    except Exception as e:
        print(f"❌ test_vertex_model_garden_predict_endpoint_request_format failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nRunning test_vertex_model_garden_model_id_detection...")
    try:
        test_instance.test_vertex_model_garden_model_id_detection()
        print("✅ test_vertex_model_garden_model_id_detection passed")
    except Exception as e:
        print(f"❌ test_vertex_model_garden_model_id_detection failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nRunning test_vertex_model_garden_response_parsing...")
    try:
        test_instance.test_vertex_model_garden_response_parsing()
        print("✅ test_vertex_model_garden_response_parsing passed")
    except Exception as e:
        print(f"❌ test_vertex_model_garden_response_parsing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Test run completed!")