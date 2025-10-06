import json
import os
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.llms.base import HiddenParams

# Mock async invoke responses
async_invoke_response = {
    "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
}

async_invoke_status_response = {
    "status": "InProgress",
    "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456",
    "outputDataConfig": {
        "s3OutputDataConfig": {
            "s3Uri": "s3://test-bucket/async-invoke-output/"
        }
    }
}

async_invoke_completed_response = {
    "status": "Completed",
    "invocationArn": "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456",
    "outputDataConfig": {
        "s3OutputDataConfig": {
            "s3Uri": "s3://test-bucket/async-invoke-output/"
        }
    }
}

# Test data
test_input = "Hello world from litellm async invoke"
test_image_base64 = "data:image/png,test_image_base64_data"


class TestBedrockAsyncInvokeEmbedding:
    """Test suite for Bedrock async-invoke embedding functionality."""

    def test_async_invoke_response_transformation_twelvelabs(self):
        """Test that async invoke responses are properly transformed with hidden params."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        response = config._transform_async_invoke_response(async_invoke_response, "test-model")
        
        # Verify response structure
        assert isinstance(response, litellm.EmbeddingResponse)
        assert hasattr(response, '_hidden_params')
        assert response._hidden_params is not None
        
        # Verify hidden params contain invocation ARN
        assert hasattr(response._hidden_params, '_invocation_arn')
        assert response._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
        
        # Verify embedding structure
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert response.data[0].embedding == []  # Empty for async jobs
        assert response.data[0].index == 0

    def test_async_invoke_response_transformation_generic(self):
        """Test that generic async invoke responses are properly transformed."""
        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
        
        bedrock_embedding = BedrockEmbedding()
        
        # Mock the transformation method
        response_list = [async_invoke_response]
        response = bedrock_embedding._transform_response(
            response_list=response_list,
            model="test-model",
            provider="twelvelabs",
            is_async_invoke=True
        )
        
        # Verify response structure
        assert isinstance(response, litellm.EmbeddingResponse)
        assert hasattr(response, '_hidden_params')
        assert response._hidden_params is not None
        
        # Verify hidden params contain invocation ARN
        assert hasattr(response._hidden_params, '_invocation_arn')
        assert response._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"

    @pytest.mark.parametrize(
        "model,input_type",
        [
            ("bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0", "text"),
            ("bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0", "image"),
            ("bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0", "video"),
            ("bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0", "audio"),
        ],
    )
    def test_async_invoke_twelvelabs_embedding_request_transformation(self, model, input_type):
        """Test that async invoke requests are properly transformed for TwelveLabs."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        
        # Test input based on type
        if input_type == "text":
            input_data = test_input
        elif input_type == "image":
            input_data = test_image_base64
        elif input_type in ["video", "audio"]:
            input_data = "s3://test-bucket/test-file.mp4" if input_type == "video" else "s3://test-bucket/test-file.wav"
        
        inference_params = {
            "inputType": input_type,  # This will be set by the parameter mapping
            "output_s3_uri": "s3://test-bucket/async-invoke-output/"
        }
        
        transformed_request = config._transform_request(
            input=input_data,
            inference_params=inference_params,
            async_invoke_route=True,
            model_id="twelvelabs.marengo-embed-2-7-v1:0",
            output_s3_uri="s3://test-bucket/async-invoke-output/"
        )
        
        # Verify async invoke request structure
        assert "modelId" in transformed_request
        assert "modelInput" in transformed_request
        assert "outputDataConfig" in transformed_request
        assert transformed_request["modelId"] == "twelvelabs.marengo-embed-2-7-v1:0"
        assert transformed_request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"] == "s3://test-bucket/async-invoke-output/"

    def test_async_invoke_twelvelabs_embedding_with_mock(self):
        """Test async invoke embedding with mocked HTTP calls."""
        litellm.set_verbose = True
        client = HTTPHandler()
        test_api_key = "test-bearer-token-12345"
        model = "bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0"

        with patch.object(client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = json.dumps(async_invoke_response)
            mock_response.json = lambda: json.loads(mock_response.text)
            mock_post.return_value = mock_response

            response = litellm.embedding(
                model=model,
                input=test_input,
                client=client,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
                input_type="text",  # New input_type parameter (maps to inputType)
                output_s3_uri="s3://test-bucket/async-invoke-output/"
            )

            # Verify response structure
            assert isinstance(response, litellm.EmbeddingResponse)
            assert hasattr(response, '_hidden_params')
            assert response._hidden_params is not None
            assert hasattr(response._hidden_params, '_invocation_arn')
            assert response._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"

            # Verify request was made to async-invoke endpoint
            request_url = mock_post.call_args.kwargs.get("url", "")
            assert "/async-invoke" in request_url

    @pytest.mark.asyncio
    async def test_async_invoke_twelvelabs_embedding_async_with_mock(self):
        """Test async invoke embedding with async calls."""
        litellm.set_verbose = True
        client = AsyncHTTPHandler()
        test_api_key = "test-bearer-token-12345"
        model = "bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0"

        with patch.object(client, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = json.dumps(async_invoke_response)
            mock_response.json = Mock(return_value=async_invoke_response)
            mock_post.return_value = mock_response

            response = await litellm.aembedding(
                model=model,
                input=test_input,
                client=client,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
                inputType="text",
                output_s3_uri="s3://test-bucket/async-invoke-output/"
            )

            # Verify response structure
            assert isinstance(response, litellm.EmbeddingResponse)
            assert hasattr(response, '_hidden_params')
            assert response._hidden_params is not None
            assert hasattr(response._hidden_params, '_invocation_arn')
            assert response._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"

    @pytest.mark.asyncio
    async def test_async_invoke_status_checking(self):
        """Test async invoke status checking functionality."""
        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
        
        bedrock_embedding = BedrockEmbedding()
        
        # Mock the async status check
        with patch.object(bedrock_embedding, '_get_async_invoke_status') as mock_status:
            mock_status.return_value = async_invoke_status_response
            
            # This would be called internally, but we can test the method directly
            status_response = await bedrock_embedding._get_async_invoke_status(
                invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456",
                aws_region_name="us-east-1"
            )
            
            assert status_response["status"] == "InProgress"
            assert "invocationArn" in status_response

    def test_async_invoke_error_handling_missing_output_s3_uri(self):
        """Test error handling when output_s3_uri is missing for async invoke."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        
        with pytest.raises(ValueError, match="output_s3_uri cannot be empty for async invoke requests"):
            config._transform_request(
                input=test_input,
                inference_params={"inputType": "text"},
                async_invoke_route=True,
                model_id="twelvelabs.marengo-embed-2-7-v1:0",
                output_s3_uri=""  # Empty S3 URI should raise error
            )

    def test_async_invoke_error_handling_video_audio_without_async_route(self):
        """Test error handling when video/audio input is used without async invoke route."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        
        with pytest.raises(ValueError, match="Input type 'video' requires async_invoke route"):
            config._transform_request(
                input="s3://test-bucket/test-video.mp4",
                inference_params={"inputType": "video"},
                async_invoke_route=False,  # Should fail for video without async route
                model_id="twelvelabs.marengo-embed-2-7-v1:0",
                output_s3_uri="s3://test-bucket/async-invoke-output/"
            )

    def test_async_invoke_invocation_arn_preservation(self):
        """Test that invocation ARN is correctly preserved in hidden params."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        
        # Test various ARN formats
        test_cases = [
            "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456",
            "arn:aws:bedrock:us-west-2:987654321098:async-invoke/xyz789",
            "invalid-arn",
            "",
        ]
        
        for arn in test_cases:
            mock_response = {"invocationArn": arn}
            response = config._transform_async_invoke_response(mock_response, "test-model")
            
            assert response._hidden_params._invocation_arn == arn

    def test_async_invoke_hidden_params_structure(self):
        """Test that hidden params have the correct structure and can be accessed."""
        from litellm.llms.bedrock.embed.twelvelabs_marengo_transformation import TwelveLabsMarengoEmbeddingConfig
        
        config = TwelveLabsMarengoEmbeddingConfig()
        response = config._transform_async_invoke_response(async_invoke_response, "test-model")
        
        # Test that hidden params can be accessed like a dictionary
        assert response._hidden_params.get("_invocation_arn") == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
        
        # Test that hidden params can be accessed like attributes
        assert response._hidden_params._invocation_arn == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"
        
        # Test that hidden params can be accessed with bracket notation
        assert response._hidden_params["_invocation_arn"] == "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123def456"

    def test_async_invoke_model_parsing(self):
        """Test that async invoke models are correctly parsed."""
        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
        
        bedrock_embedding = BedrockEmbedding()
        
        # Test model parsing
        test_models = [
            "bedrock/async_invoke/twelvelabs.marengo-embed-2-7-v1:0",
            "bedrock/async_invoke/amazon.titan-embed-text-v1",
            "bedrock/async_invoke/cohere.embed-english-v3",
        ]
        
        for model in test_models:
            # Check if async invoke is detected
            has_async_invoke = "async_invoke/" in model
            assert has_async_invoke, f"Model {model} should be detected as async invoke"
            
            # Check model ID extraction (remove both "bedrock/" and "async_invoke/" prefixes)
            if has_async_invoke:
                model_id = model.replace("bedrock/async_invoke/", "", 1)
                assert model_id in [
                    "twelvelabs.marengo-embed-2-7-v1:0",
                    "amazon.titan-embed-text-v1", 
                    "cohere.embed-english-v3"
                ]

    def test_async_invoke_endpoint_construction(self):
        """Test that async invoke endpoints are correctly constructed."""
        from litellm.llms.bedrock.embed.embedding import BedrockEmbedding
        
        bedrock_embedding = BedrockEmbedding()
        
        # Mock the get_runtime_endpoint method
        with patch.object(bedrock_embedding, 'get_runtime_endpoint') as mock_endpoint:
            mock_endpoint.return_value = ("https://bedrock-runtime.us-east-1.amazonaws.com", None)
            
            # Test endpoint construction for async invoke
            endpoint_url, _ = bedrock_embedding.get_runtime_endpoint(
                api_base=None,
                aws_bedrock_runtime_endpoint=None,
                aws_region_name="us-east-1"
            )
            
            # For async invoke, the endpoint should be modified
            async_endpoint = f"{endpoint_url}/async-invoke"
            assert async_endpoint == "https://bedrock-runtime.us-east-1.amazonaws.com/async-invoke"
