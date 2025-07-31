"""Unit tests for Bedrock batch functionality."""

import asyncio
import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.bedrock.batches.handler import BedrockBatchesHandler
from litellm.llms.bedrock.files.handler import BedrockFilesHandler
from litellm.types.llms.openai import CreateBatchRequest, CreateFileRequest
from litellm.types.utils import LiteLLMBatch


class TestBedrockFilesHandler:
    """Test Bedrock files handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BedrockFilesHandler()
        
    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "AWS_REGION": "us-east-1"
    })
    def test_get_s3_config(self):
        """Test S3 configuration retrieval."""
        config = self.handler._get_s3_config()
        assert config["bucket_name"] == "test-bucket"
        assert config["region_name"] == "us-east-1"

    def test_get_s3_config_missing_bucket(self):
        """Test S3 config fails when bucket is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LITELLM_BEDROCK_BATCH_BUCKET"):
                self.handler._get_s3_config()

    def test_generate_s3_key(self):
        """Test S3 key generation."""
        key = self.handler._generate_s3_key("test.jsonl", "batch")
        assert key.startswith("batch/")
        assert key.endswith(".jsonl")
        
        # Test without extension
        key_no_ext = self.handler._generate_s3_key("test", "batch")
        assert key_no_ext.startswith("batch/")
        assert not key_no_ext.endswith(".jsonl")

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "AWS_REGION": "us-east-1"
    })
    @pytest.mark.asyncio
    async def test_upload_to_s3_success(self):
        """Test successful S3 upload."""
        # Mock AWS credential retrieval and HTTP client
        with patch.object(self.handler, 'get_credentials') as mock_creds, \
             patch.object(self.handler, 'async_httpx_client') as mock_client:
            
            # Create a proper credentials mock that can be used by SigV4Auth
            mock_creds_obj = MagicMock()
            mock_creds_obj.access_key = "test-access-key"
            mock_creds_obj.secret_key = "test-secret-key"
            mock_creds_obj.token = None
            mock_creds.return_value = mock_creds_obj
            
            # Mock the httpx client PUT response  
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            
            # Use AsyncMock for async methods
            mock_client.put = AsyncMock(return_value=mock_response)
            
            result = await self.handler._upload_to_s3(
                file_content=b"test content",
                s3_key="test-key",
                bucket_name="test-bucket",
                region_name="us-east-1",
                filename="test.jsonl"
            )
            
            assert result["s3_uri"] == "s3://test-bucket/test-key"
            assert result["s3_key"] == "test-key"
            assert result["bucket_name"] == "test-bucket"
            mock_client.put.assert_called_once()

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "AWS_REGION": "us-east-1"
    })
    @pytest.mark.asyncio
    async def test_async_create_file(self):
        """Test async file creation."""
        # Mock file content
        mock_file = MagicMock()
        mock_file.read.return_value = b'{"test": "content"}'
        mock_file.name = "test.jsonl"
        
        create_request = CreateFileRequest(
            file=mock_file,
            purpose="batch"
        )
        
        # Mock S3 upload
        with patch.object(self.handler, '_upload_to_s3') as mock_upload:
            mock_upload.return_value = {
                "s3_uri": "s3://test-bucket/batch/test-key",
                "s3_key": "batch/test-key",
                "bucket_name": "test-bucket",
                "region_name": "us-east-1"
            }
            
            result = await self.handler.async_create_file(create_request)
            
            assert result.purpose == "batch"
            assert result.filename == "test.jsonl"
            assert result.bytes == len(b'{"test": "content"}')
            assert result.status == "processed"
            assert hasattr(result, '_hidden_params')


class TestBedrockBatchesHandler:
    """Test Bedrock batches handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = BedrockBatchesHandler()

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
        "AWS_REGION": "us-east-1"
    })
    def test_get_batch_config(self):
        """Test batch configuration retrieval."""
        config = self.handler._get_batch_config()
        assert config["bucket_name"] == "test-bucket"
        assert config["role_arn"] == "arn:aws:iam::123456789012:role/BedrockBatchRole"
        assert config["region_name"] == "us-east-1"

    def test_get_batch_config_missing_bucket(self):
        """Test batch config fails when bucket is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="LITELLM_BEDROCK_BATCH_BUCKET"):
                self.handler._get_batch_config()

    def test_get_batch_config_missing_role(self):
        """Test batch config fails when role ARN is missing."""
        with patch.dict(os.environ, {"LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket"}, clear=True):
            with pytest.raises(ValueError, match="LITELLM_BEDROCK_BATCH_ROLE_ARN"):
                self.handler._get_batch_config()

    def test_map_openai_to_bedrock_status(self):
        """Test status mapping from Bedrock to OpenAI format."""
        mapping_tests = [
            ("Submitted", "validating"),
            ("Validating", "validating"),
            ("Scheduled", "in_progress"),
            ("InProgress", "in_progress"),
            ("Completed", "completed"),
            ("PartiallyCompleted", "completed"),
            ("Failed", "failed"),
            ("Stopped", "cancelled"),
            ("Stopping", "cancelling"),
            ("Expired", "expired"),
            ("Unknown", "failed"),  # Default case
        ]
        
        for bedrock_status, expected_openai_status in mapping_tests:
            result = self.handler._map_openai_to_bedrock_status(bedrock_status)
            assert result == expected_openai_status

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
    })
    def test_build_bedrock_batch_request(self):
        """Test building Bedrock batch request from OpenAI format."""
        create_request = CreateBatchRequest(
            input_file_id="file-123",
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_config = {
            "bucket_name": "test-bucket",
            "role_arn": "arn:aws:iam::123456789012:role/BedrockBatchRole"
        }
        
        result = self.handler._build_bedrock_batch_request(
            create_request, batch_config, "anthropic.claude-3-haiku-20240307-v1:0"
        )
        
        assert "jobName" in result
        assert result["roleArn"] == batch_config["role_arn"]
        assert result["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
        assert result["inputDataConfig"]["s3InputDataConfig"]["s3InputFormat"] == "JSONL"
        assert "s3://test-bucket/batch/" in result["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        assert "s3://test-bucket/batch-outputs/" in result["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        assert result["timeoutDurationInHours"] == 24

    def test_transform_bedrock_response_to_litellm_batch(self):
        """Test transforming Bedrock response to LiteLLM format."""
        bedrock_response = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abcd1234",
            "jobName": "test-job"
        }
        
        create_request = CreateBatchRequest(
            input_file_id="file-123",
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        result = self.handler._transform_bedrock_response_to_litellm_batch(
            bedrock_response, create_request
        )
        
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "abcd1234"
        assert result.object == "batch"
        assert result.endpoint == "/v1/chat/completions"
        assert result.input_file_id == "file-123"
        assert result.status == "validating"
        assert hasattr(result, '_hidden_params')
        assert result._hidden_params["job_arn"] == bedrock_response["jobArn"]

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
        "AWS_REGION": "us-east-1"
    })
    @pytest.mark.asyncio
    async def test_async_create_batch(self):
        """Test async batch creation."""
        create_request = CreateBatchRequest(
            input_file_id="file-123",
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        # Mock the Bedrock API response
        mock_response = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abcd1234"
        }
        
        # Mock signing and HTTP client
        with patch.object(self.handler, '_sign_request') as mock_sign, \
             patch.object(self.handler, 'async_httpx_client') as mock_client:
            
            mock_sign.return_value = ({}, b'{"test": "data"}')
            
            # Mock the httpx client POST response
            mock_http_response = MagicMock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            
            # Use AsyncMock for async methods
            mock_client.post = AsyncMock(return_value=mock_http_response)
            
            result = await self.handler.async_create_batch(
                create_request,
                model_id="anthropic.claude-3-haiku-20240307-v1:0"
            )
            
            assert isinstance(result, LiteLLMBatch)
            assert result.id == "abcd1234"
            assert result.status == "validating"
            mock_sign.assert_called_once()
            mock_client.post.assert_called_once()

    @patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
        "AWS_REGION": "us-east-1"
    })
    @pytest.mark.asyncio
    async def test_async_retrieve_batch(self):
        """Test async batch retrieval."""
        batch_id = "abcd1234"
        
        # Mock the Bedrock API response
        mock_response = {
            "status": "InProgress",
            "submitTime": "2024-01-01T00:00:00Z",
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3Uri": "s3://test-bucket/batch/file-123.jsonl"
                }
            }
        }
        
        # Mock signing and HTTP client
        with patch.object(self.handler, '_sign_request') as mock_sign, \
             patch.object(self.handler, 'async_httpx_client') as mock_client:
            
            mock_sign.return_value = ({}, None)
            
            # Mock the httpx client GET response
            mock_http_response = MagicMock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            
            # Use AsyncMock for async methods
            mock_client.get = AsyncMock(return_value=mock_http_response)
            
            result = await self.handler.async_retrieve_batch(batch_id)
            
            assert isinstance(result, LiteLLMBatch)
            assert result.id == batch_id
            assert result.status == "in_progress"
            mock_sign.assert_called_once()
            mock_client.get.assert_called_once()


# Integration test helpers
@pytest.mark.asyncio
async def test_bedrock_batch_integration():
    """Integration test for file upload + batch creation workflow."""
    # This would require actual AWS credentials and services
    # For now, we'll test the interface integration
    
    with patch.dict(os.environ, {
        "LITELLM_BEDROCK_BATCH_BUCKET": "test-bucket",
        "LITELLM_BEDROCK_BATCH_ROLE_ARN": "arn:aws:iam::123456789012:role/BedrockBatchRole",
        "AWS_REGION": "us-east-1"
    }):
        files_handler = BedrockFilesHandler()
        batches_handler = BedrockBatchesHandler()
        
        # Test that handlers can be instantiated
        assert files_handler is not None
        assert batches_handler is not None
        
        # Test config validation
        files_config = files_handler._get_s3_config()
        batch_config = batches_handler._get_batch_config()
        
        assert files_config["bucket_name"] == batch_config["bucket_name"]


if __name__ == "__main__":
    pytest.main([__file__])