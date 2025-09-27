import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
import pytest
import base64
import httpx

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

titan_embedding_response = {
        "embedding": [0.1, 0.2, 0.3],
        "inputTextTokenCount": 10
}

cohere_embedding_response = {
    "embeddings": [[0.1, 0.2, 0.3]],
    "inputTextTokenCount": 10
}

img_base_64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABB0lEQVRYhe2SzWrEIBCAh2A0jxEs4j6GLDS9hqWmV5Flt0cJS+lRwv742DXpEjY1kOZW6HwHFZnPmVEBEARBEARB/jd0KYA/bcUYbPrRLh6amXHJ/K+ypMoyUaGthILzw0l+xI0jsO7ZcmCcm4ILd+QuVYgpHOmDmz6jBeJImdcUCmeBqQpuqRIbVmQsLCrAalrGpfoEqEogqbLTWuXCPCo+Ki1XGqgQ+jVVuhB8bOaHkvmYuzm/b0KYLWwoK58oFqi6XfxQ4Uz7d6WeKpna6ytUs5e8betMcqAv5YPC5EZB2Lm9FIn0/VP6R58+/GEY1X1egVoZ/3bt/EqF6malgSAIgiDIH+QL41409QMY0LMAAAAASUVORK5CYII="

@pytest.mark.parametrize(
    "model,input_type,embed_response",
    [
        ("bedrock/amazon.titan-embed-text-v1", "text", titan_embedding_response),  # V1 text model
        ("bedrock/amazon.titan-embed-text-v2:0", "text", titan_embedding_response),  # V2 text model
        ("bedrock/amazon.titan-embed-image-v1", "image", titan_embedding_response),  # Image model
        ("bedrock/cohere.embed-english-v3", "text", cohere_embedding_response),  # Cohere English
        ("bedrock/cohere.embed-multilingual-v3", "text", cohere_embedding_response),  # Cohere Multilingual
    ],
)
def test_bedrock_embedding_models(model, input_type, embed_response):
    """Test embedding functionality for all Bedrock models with different input types"""
    litellm.set_verbose = True
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(embed_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Prepare input based on type
        input_data = img_base_64 if input_type == "image" else "Hello world from litellm"

        try:
            response = litellm.embedding(
                model=model,
                input=input_data,
                client=client,
                aws_region_name="us-west-2",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-west-2.amazonaws.com",
            )

            # Verify response structure
            assert isinstance(response, litellm.EmbeddingResponse)
            print(response.data)
            assert isinstance(response.data[0]['embedding'], list)
            assert len(response.data[0]['embedding']) == 3  # Based on mock response

            # Fetch request body
            request_data = json.loads(mock_post.call_args.kwargs["data"])

            # Verify AWS params are not in request body
            aws_params = ["aws_region_name", "aws_bedrock_runtime_endpoint"]
            for param in aws_params:
                assert param not in request_data, f"AWS param {param} should not be in request body"

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


def test_e2e_bedrock_embedding():
    """
    Test text embedding with TwelveLabs Marengo.
    Validates that the transformation properly extracts embedding data from TwelveLabs response format.
    """
    print("Testing text embedding...")
    original_region_name = os.environ.get("AWS_REGION_NAME")
    
    os.environ["AWS_REGION_NAME"] = "us-east-1"

    litellm._turn_on_debug()
    response = litellm.embedding(
        model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0",
        input=["Hello world from LiteLLM with TwelveLabs Marengo!"],
    )
    
    # Validate response structure
    assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
    assert hasattr(response, 'data'), "Response should have 'data' attribute"
    assert len(response.data) > 0, "Response data should not be empty"
    
    # Validate first embedding
    embedding_obj = response.data[0]
    assert hasattr(embedding_obj, 'embedding'), "Embedding object should have 'embedding' attribute"
    assert isinstance(embedding_obj.embedding, list), "Embedding should be a list of floats"
    assert len(embedding_obj.embedding) > 0, "Embedding vector should not be empty"
    assert all(isinstance(x, (int, float)) for x in embedding_obj.embedding), "All embedding values should be numeric"
    
    # Validate embedding properties
    assert embedding_obj.index == 0, "First embedding should have index 0"
    assert embedding_obj.object == "embedding", "Embedding object type should be 'embedding'"
    
    # Validate usage information
    assert hasattr(response, 'usage'), "Response should have usage information"
    assert response.usage is not None, "Usage should not be None"
    assert response.usage.total_tokens >= 0, "Total tokens should be non-negative"
    
    print(f"Text embedding successful! Vector size: {len(embedding_obj.embedding)}, Response: {response}")

    # Restore original region name
    if original_region_name:
        os.environ["AWS_REGION_NAME"] = original_region_name

def test_e2e_bedrock_embedding_image_twelvelabs_marengo():
    """
    Test image embedding with TwelveLabs Marengo.
    Validates that the transformation properly extracts embedding data from TwelveLabs response format for images.
    """
    print("Testing image embedding...")
    original_region_name = os.environ.get("AWS_REGION_NAME")
    os.environ["AWS_REGION_NAME"] = "us-east-1"
    litellm._turn_on_debug()
    
    # Load duck.png and convert to base64
    duck_img_path = os.path.join(os.path.dirname(__file__), "duck.png")
    with open(duck_img_path, "rb") as img_file:
        duck_img_data = base64.b64encode(img_file.read()).decode('utf-8')
        duck_img_base64 = f"data:image/png;base64,{duck_img_data}"
    
        response = litellm.embedding(
            model="bedrock/us.twelvelabs.marengo-embed-2-7-v1:0",
            input=[duck_img_base64],
            aws_region_name="us-east-1"
        )
        
        # Validate response structure
        assert isinstance(response, litellm.EmbeddingResponse), "Response should be EmbeddingResponse type"
        assert hasattr(response, 'data'), "Response should have 'data' attribute"
        assert len(response.data) > 0, "Response data should not be empty"
        
        # Validate first embedding
        embedding_obj = response.data[0]
        assert hasattr(embedding_obj, 'embedding'), "Embedding object should have 'embedding' attribute"
        assert isinstance(embedding_obj.embedding, list), "Embedding should be a list of floats"
        assert len(embedding_obj.embedding) > 0, "Embedding vector should not be empty"
        assert all(isinstance(x, (int, float)) for x in embedding_obj.embedding), "All embedding values should be numeric"
        
        # Validate embedding properties
        assert embedding_obj.index == 0, "First embedding should have index 0"
        assert embedding_obj.object == "embedding", "Embedding object type should be 'embedding'"
        
        # Validate usage information
        assert hasattr(response, 'usage'), "Response should have usage information"
        assert response.usage is not None, "Usage should not be None"
        assert response.usage.total_tokens >= 0, "Total tokens should be non-negative"
        
        # TwelveLabs Marengo should return 1024-dimensional embeddings
        expected_dimension = 1024
        assert len(embedding_obj.embedding) == expected_dimension, f"TwelveLabs Marengo should return {expected_dimension}-dimensional embeddings, got {len(embedding_obj.embedding)}"
        
        print(f"Image embedding successful! Vector size: {len(embedding_obj.embedding)}, Response: {response}")

    # Restore original region name
    if original_region_name:
        os.environ["AWS_REGION_NAME"] = original_region_name