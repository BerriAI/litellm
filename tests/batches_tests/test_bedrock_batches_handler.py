# What is this?
## Unit Tests for Bedrock Batches Handler
import json
import os
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from typing import Dict, Any

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.llms.bedrock.batches.handler import BedrockBatchesAPI
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.utils import LiteLLMBatch


class TestBedrockBatchesHandler:
    """Test cases for BedrockBatchesAPI handler class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.handler = BedrockBatchesAPI()
        
        self.sample_create_batch_data: CreateBatchRequest = {
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/test-file.jsonl",
            "metadata": {"job_name": "test-handler-job", "description": "Test handler job"},
            "model": "anthropic.claude-3-sonnet-20240229-v1:0"
        }

        self.sample_bedrock_job_response = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-handler-job-123",
            "jobName": "test-handler-job",
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "status": "InProgress",
            "submitTime": datetime(2024, 1, 15, 10, 0, 0),
            "lastModifiedTime": datetime(2024, 1, 15, 10, 5, 0),
            "endTime": None,
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3Uri": "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/test-file.jsonl"
                }
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/"
                }
            }
        }

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_create_batch_success(self, mock_init_client):
        """Test successful batch creation"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock create_model_invocation_job response
        mock_client.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-handler-job-123"
        }
        
        # Mock get_model_invocation_job response
        mock_client.get_model_invocation_job.return_value = self.sample_bedrock_job_response

        result = self.handler.create_batch(
            _is_async=False,
            create_batch_data=self.sample_create_batch_data,
            timeout=3600.0,
            max_retries=None,
            litellm_params={},
            api_base=None,
            logging_obj=None,
            client=None
        )

        # Verify the result
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "test-handler-job-123"
        assert result.status == "in_progress"
        assert result.object == "batch"
        assert result.endpoint == "/v1/chat/completions"
        assert result.input_file_id == "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/test-file.jsonl"
        assert result.completion_window == "24h"

        # Verify client calls
        mock_client.create_model_invocation_job.assert_called_once()
        mock_client.get_model_invocation_job.assert_called_once()

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_create_batch_fallback_when_no_job_arn(self, mock_init_client):
        """Test batch creation fallback when job ARN is not returned"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock create_model_invocation_job response without jobArn
        mock_client.create_model_invocation_job.return_value = {}

        result = self.handler.create_batch(
            _is_async=False,
            create_batch_data=self.sample_create_batch_data,
            timeout=3600.0,
            max_retries=None,
            litellm_params={},
            api_base=None,
            logging_obj=None,
            client=None
        )

        # Verify fallback result
        assert isinstance(result, LiteLLMBatch)
        assert result.status == "validating"
        assert result.object == "batch"
        assert result.endpoint == "/v1/chat/completions"
        assert result.completion_window == "24h"

    def test_create_batch_invalid_s3_uri(self):
        """Test batch creation with invalid S3 URI"""
        invalid_batch_data = self.sample_create_batch_data.copy()
        invalid_batch_data["input_file_id"] = "invalid-uri"

        with pytest.raises(ValueError, match="input_file_id must be an s3:// URI for Bedrock batch jobs"):
            self.handler.create_batch(
                _is_async=False,
                create_batch_data=invalid_batch_data,
                timeout=3600.0,
                max_retries=None,
                litellm_params={},
                api_base=None,
                logging_obj=None,
                client=None
            )

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_create_batch_extract_model_from_s3_path(self, mock_init_client):
        """Test extracting model ID from S3 path when not explicitly provided"""
        batch_data_without_model = self.sample_create_batch_data.copy()
        del batch_data_without_model["model"]
        
        # The S3 path contains the model ID: s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/test-file.jsonl
        # So the model should be extracted as: anthropic.claude-3-sonnet-20240229-v1:0
        
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        mock_client.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job"
        }
        
        mock_client.get_model_invocation_job.return_value = self.sample_bedrock_job_response

        result = self.handler.create_batch(
            _is_async=False,
            create_batch_data=batch_data_without_model,
            timeout=3600.0,
            max_retries=None,
            litellm_params={},
            api_base=None,
            logging_obj=None,
            client=None
        )

        # Verify the result
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "test-handler-job-123"

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_retrieve_batch_success(self, mock_init_client):
        """Test successful batch retrieval"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock get_model_invocation_job response
        mock_client.get_model_invocation_job.return_value = self.sample_bedrock_job_response

        result = self.handler.retrieve_batch(
            batch_id="test-handler-job-123",
            _is_async=False,
            api_base=None,
            litellm_params={}
        )

        # Verify the result
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "test-handler-job-123"
        assert result.status == "in_progress"
        assert result.object == "batch"

        # Verify client call
        mock_client.get_model_invocation_job.assert_called_once_with(jobIdentifier="test-handler-job-123")

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_cancel_batch_success(self, mock_init_client):
        """Test successful batch cancellation"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock stop_model_invocation_job response
        mock_client.stop_model_invocation_job.return_value = {}
        
        # Mock get_model_invocation_job response for updated status
        cancelled_response = self.sample_bedrock_job_response.copy()
        cancelled_response["status"] = "Stopped"
        cancelled_response["endTime"] = datetime(2024, 1, 15, 10, 15, 0)
        mock_client.get_model_invocation_job.return_value = cancelled_response

        result = self.handler.cancel_batch(
            batch_id="test-handler-job-123",
            _is_async=False,
            api_base=None,
            litellm_params={}
        )

        # Verify the result
        assert isinstance(result, LiteLLMBatch)
        assert result.id == "test-handler-job-123"
        assert result.status == "cancelled"

        # Verify client calls
        mock_client.stop_model_invocation_job.assert_called_once_with(jobIdentifier="test-handler-job-123")
        mock_client.get_model_invocation_job.assert_called_once_with(jobIdentifier="test-handler-job-123")

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_list_batches_success(self, mock_init_client):
        """Test successful batch listing"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock list_model_invocation_jobs response
        mock_client.list_model_invocation_jobs.return_value = {
            "nextToken": "next-page-token",
            "invocationJobSummaries": [
                {
                    "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-1",
                    "status": "Completed",
                    "submitTime": datetime(2024, 1, 15, 10, 0, 0),
                    "endTime": datetime(2024, 1, 15, 11, 0, 0),
                    "inputDataConfig": {
                        "s3InputDataConfig": {
                            "s3Uri": "s3://test-bucket/model/input1.jsonl"
                        }
                    },
                    "outputDataConfig": {
                        "s3OutputDataConfig": {
                            "s3Uri": "s3://test-bucket/model/output1/"
                        }
                    }
                },
                {
                    "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-2",
                    "status": "InProgress",
                    "submitTime": datetime(2024, 1, 15, 12, 0, 0),
                    "lastModifiedTime": datetime(2024, 1, 15, 12, 5, 0),
                    "inputDataConfig": {
                        "s3InputDataConfig": {
                            "s3Uri": "s3://test-bucket/model/input2.jsonl"
                        }
                    },
                    "outputDataConfig": {
                        "s3OutputDataConfig": {
                            "s3Uri": "s3://test-bucket/model/output2/"
                        }
                    }
                }
            ]
        }

        result = self.handler.list_batches(
            _is_async=False,
            api_base=None,
            litellm_params={},
            after="2024-01-15T00:00:00Z",
            limit=10
        )

        # Verify the result structure
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["has_more"] is True
        assert result["first_id"] == "job-1"
        assert result["last_id"] == "job-2"

        # Verify the transformed batch data
        first_batch = result["data"][0]
        assert first_batch["id"] == "job-1"
        assert first_batch["status"] == "completed"
        assert first_batch["object"] == "batch"

        second_batch = result["data"][1]
        assert second_batch["id"] == "job-2"
        assert second_batch["status"] == "in_progress"
        assert second_batch["object"] == "batch"

        # Verify client call
        mock_client.list_model_invocation_jobs.assert_called_once()

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_list_batches_with_limit(self, mock_init_client):
        """Test batch listing with limit parameter"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock empty response
        mock_client.list_model_invocation_jobs.return_value = {
            "invocationJobSummaries": []
        }

        result = self.handler.list_batches(
            _is_async=False,
            api_base=None,
            litellm_params={},
            after=None,
            limit=5
        )

        # Verify the result
        assert result["object"] == "list"
        assert len(result["data"]) == 0
        assert result["has_more"] is False
        assert result["first_id"] is None
        assert result["last_id"] is None

        # Verify client call with limit and default parameters
        mock_client.list_model_invocation_jobs.assert_called_once()
        call_args = mock_client.list_model_invocation_jobs.call_args[1]
        assert call_args["maxResults"] == 5
        assert "sortBy" in call_args
        assert "sortOrder" in call_args

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_list_batches_with_after_timestamp(self, mock_init_client):
        """Test batch listing with after timestamp parameter"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock response
        mock_client.list_model_invocation_jobs.return_value = {
            "invocationJobSummaries": []
        }

        result = self.handler.list_batches(
            _is_async=False,
            api_base=None,
            litellm_params={},
            after="2024-01-15T10:00:00Z",
            limit=None
        )

        # Verify client call includes timestamp conversion
        mock_client.list_model_invocation_jobs.assert_called_once()
        call_args = mock_client.list_model_invocation_jobs.call_args[1]
        assert "submitTimeAfter" in call_args

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_list_batches_invalid_timestamp(self, mock_init_client):
        """Test batch listing with invalid timestamp parameter"""
        # Mock the boto3 client
        mock_client = MagicMock()
        mock_init_client.return_value = mock_client
        
        # Mock response
        mock_client.list_model_invocation_jobs.return_value = {
            "invocationJobSummaries": []
        }

        # Should handle invalid timestamp gracefully
        result = self.handler.list_batches(
            _is_async=False,
            api_base=None,
            litellm_params={},
            after="invalid-timestamp",
            limit=None
        )

        # Verify the result still works
        assert result["object"] == "list"
        assert len(result["data"]) == 0

        # Verify client call without timestamp but with default parameters
        mock_client.list_model_invocation_jobs.assert_called_once()
        call_args = mock_client.list_model_invocation_jobs.call_args[1]
        assert call_args["maxResults"] == 20
        assert "sortBy" in call_args
        assert "sortOrder" in call_args

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_create_batch_role_arn_from_config(self, mock_init_client):
        """Test batch creation with role ARN from litellm config"""
        # Mock litellm.s3_callback_params
        with patch.object(litellm, 's3_callback_params', {'s3_aws_role_name': 'arn:aws:iam::123456789012:role/BedrockBatchRole'}):
            # Mock the boto3 client
            mock_client = MagicMock()
            mock_init_client.return_value = mock_client
            
            mock_client.create_model_invocation_job.return_value = {
                "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job"
            }
            
            mock_client.get_model_invocation_job.return_value = self.sample_bedrock_job_response

            result = self.handler.create_batch(
                _is_async=False,
                create_batch_data=self.sample_create_batch_data,
                timeout=3600.0,
                max_retries=None,
                litellm_params={},
                api_base=None,
                logging_obj=None,
                client=None
            )

            # Verify the result
            assert isinstance(result, LiteLLMBatch)
            assert result.id == "test-handler-job-123"

    @patch('litellm.llms.bedrock.common_utils.init_bedrock_service_client')
    def test_create_batch_invalid_role_arn(self, mock_init_client):
        """Test batch creation with invalid role ARN from config"""
        # Mock litellm.s3_callback_params with invalid role ARN
        with patch.object(litellm, 's3_callback_params', {'s3_aws_role_name': 'invalid-role-name'}):
            # Mock the boto3 client
            mock_client = MagicMock()
            mock_init_client.return_value = mock_client
            
            mock_client.create_model_invocation_job.return_value = {
                "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job"
            }
            
            mock_client.get_model_invocation_job.return_value = self.sample_bedrock_job_response

            result = self.handler.create_batch(
                _is_async=False,
                create_batch_data=self.sample_create_batch_data,
                timeout=3600.0,
                max_retries=None,
                litellm_params={},
                api_base=None,
                logging_obj=None,
                client=None
            )

            # Verify the result still works (role ARN should be ignored)
            assert isinstance(result, LiteLLMBatch)
            assert result.id == "test-handler-job-123"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
