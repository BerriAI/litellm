# What is this?
## Unit Tests for Bedrock Batches API and Transformation
import os
import sys
from datetime import datetime
from typing import Any, Dict

import pytest

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.bedrock.batches.transformation import BedrockBatchTransformation
from litellm.types.llms.openai import CreateBatchRequest, LiteLLMBatchCreateRequest
from litellm.types.utils import LiteLLMBatch


class TestBedrockBatchTransformation:
    """Test cases for BedrockBatchTransformation class"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.sample_create_batch_request: CreateBatchRequest = {
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "s3://test-bucket/test-model/test-file.jsonl",
            "metadata": {"job_name": "test-job", "description": "Test batch job"},
        }

        self.sample_bedrock_response: Dict[str, Any] = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-123",
            "jobName": "test-job",
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "clientRequestToken": "test-token-123",
            "roleArn": "arn:aws:iam::123456789012:role/BedrockBatchRole",
            "status": "InProgress",
            "message": None,
            "submitTime": datetime(2024, 1, 15, 10, 0, 0),
            "lastModifiedTime": datetime(2024, 1, 15, 10, 5, 0),
            "endTime": None,
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3InputFormat": "JSONL",
                    "s3Uri": "s3://test-bucket/test-model/test-file.jsonl",
                    "s3BucketOwner": "123456789012",
                }
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": "s3://test-bucket/test-model/output/",
                    "s3EncryptionKeyId": None,
                    "s3BucketOwner": "123456789012",
                }
            },
            "vpcConfig": None,
            "timeoutDurationInHours": 24,
            "jobExpirationTime": datetime(2024, 1, 16, 10, 0, 0),
        }

    def test_transform_openai_batch_request_to_bedrock_job_request(self):
        """Test transforming OpenAI batch request to Bedrock job request"""
        # Test with all parameters
        result = BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
            req=self.sample_create_batch_request,
            s3_input_uri="s3://test-bucket/test-model/test-file.jsonl",
            s3_output_uri="s3://test-bucket/test-model/output/",
            role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        )

        assert result["modelId"] == ""  # No model specified in request
        assert result["jobName"] == "test-job"
        assert result["roleArn"] == "arn:aws:iam::123456789012:role/BedrockBatchRole"
        assert (
            result["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
            == "s3://test-bucket/test-model/test-file.jsonl"
        )
        assert (
            result["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            == "s3://test-bucket/test-model/output/"
        )

    def test_transform_openai_batch_request_to_bedrock_job_request_with_model(self):
        """Test transforming OpenAI batch request with model specified"""
        request_with_model = self.sample_create_batch_request.copy()
        request_with_model["model"] = "anthropic.claude-3-sonnet-20240229-v1:0"

        result = BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
            req=request_with_model,
            s3_input_uri="s3://test-bucket/test-model/test-file.jsonl",
            s3_output_uri="s3://test-bucket/test-model/output/",
            role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        )

        assert result["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_transform_openai_batch_request_to_bedrock_job_request_with_custom_id(self):
        """Test transforming OpenAI batch request with custom_id instead of metadata job_name"""
        request_with_custom_id = self.sample_create_batch_request.copy()
        del request_with_custom_id["metadata"]
        request_with_custom_id["custom_id"] = "custom-job-123"

        result = BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
            req=request_with_custom_id,
            s3_input_uri="s3://test-bucket/test-model/test-file.jsonl",
            s3_output_uri="s3://test-bucket/test-model/output/",
            role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        )

        assert result["jobName"] == "custom-job-123"

    def test_transform_openai_batch_request_to_bedrock_job_request_fallback_name(self):
        """Test transforming OpenAI batch request with fallback job name"""
        request_minimal = {
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "s3://test-bucket/test-model/test-file.jsonl",
        }

        result = BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
            req=request_minimal,
            s3_input_uri="s3://test-bucket/test-model/test-file.jsonl",
            s3_output_uri="s3://test-bucket/test-model/output/",
            role_arn=None,
        )

        assert result["jobName"] == "litellm-batch-job"
        assert result["roleArn"] == ""

    def test_transform_bedrock_response_to_openai_batch_in_progress(self):
        """Test transforming Bedrock response with InProgress status"""
        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=self.sample_bedrock_response
        )

        assert isinstance(result, LiteLLMBatch)
        assert result.id == "test-job-123"
        assert result.status == "in_progress"
        assert result.object == "batch"
        assert result.endpoint == "/v1/chat/completions"
        assert result.input_file_id == "s3://test-bucket/test-model/test-file.jsonl"
        assert result.output_file_id == "s3://test-bucket/test-model/output/"
        assert result.created_at == int(datetime(2024, 1, 15, 10, 0, 0).timestamp())
        assert result.in_progress_at == int(datetime(2024, 1, 15, 10, 5, 0).timestamp())
        assert result.completed_at is None
        assert result.failed_at is None
        assert result.completion_window == "24h"
        assert result.errors is None
        assert result.request_counts is None

    def test_transform_bedrock_response_to_openai_batch_completed(self):
        """Test transforming Bedrock response with Completed status"""
        completed_response = self.sample_bedrock_response.copy()
        completed_response["status"] = "Completed"
        completed_response["endTime"] = datetime(2024, 1, 15, 11, 0, 0)

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=completed_response
        )

        assert result.status == "completed"
        assert result.completed_at == int(datetime(2024, 1, 15, 11, 0, 0).timestamp())
        assert result.in_progress_at is None
        assert result.failed_at is None

    def test_transform_bedrock_response_to_openai_batch_failed(self):
        """Test transforming Bedrock response with Failed status"""
        failed_response = self.sample_bedrock_response.copy()
        failed_response["status"] = "Failed"
        failed_response["endTime"] = datetime(2024, 1, 15, 10, 30, 0)
        failed_response["message"] = "Job failed due to invalid input format"

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=failed_response
        )

        assert result.status == "failed"
        assert result.failed_at == int(datetime(2024, 1, 15, 10, 30, 0).timestamp())
        assert result.errors is not None
        assert result.errors.object == "list"
        assert len(result.errors.data) == 1
        assert result.errors.data[0].object == "error"
        assert result.errors.data[0].code == "Failed"
        assert result.errors.data[0].message == "Job failed due to invalid input format"

    def test_transform_bedrock_response_to_openai_batch_cancelled(self):
        """Test transforming Bedrock response with Stopped status"""
        cancelled_response = self.sample_bedrock_response.copy()
        cancelled_response["status"] = "Stopped"
        cancelled_response["endTime"] = datetime(2024, 1, 15, 10, 15, 0)

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=cancelled_response
        )

        assert result.status == "cancelled"
        # The cancelled_at timestamp should come from lastModifiedTime, not endTime
        assert result.cancelled_at == int(datetime(2024, 1, 15, 10, 5, 0).timestamp())

    def test_transform_bedrock_response_to_openai_batch_cancelling(self):
        """Test transforming Bedrock response with Stopping status"""
        cancelling_response = self.sample_bedrock_response.copy()
        cancelling_response["status"] = "Stopping"

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=cancelling_response
        )

        assert result.status == "cancelling"
        assert result.cancelling_at == int(datetime(2024, 1, 15, 10, 5, 0).timestamp())

    def test_transform_bedrock_response_to_openai_batch_with_input_file_id_override(
        self,
    ):
        """Test transforming Bedrock response with custom input_file_id override"""
        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=self.sample_bedrock_response,
            input_file_id="custom-input-file-id",
        )

        assert result.input_file_id == "custom-input-file-id"

    def test_transform_bedrock_response_to_openai_batch_metadata(self):
        """Test transforming Bedrock response metadata extraction"""
        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=self.sample_bedrock_response
        )

        assert result.metadata is not None
        assert result.metadata["job_name"] == "test-job"
        assert result.metadata["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert (
            result.metadata["job_arn"]
            == "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-123"
        )
        assert result.metadata["failure_message"] == ""

    def test_transform_bedrock_response_to_openai_batch_expired(self):
        """Test transforming Bedrock response with Expired status"""
        expired_response = self.sample_bedrock_response.copy()
        expired_response["status"] = "Expired"

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=expired_response
        )

        assert result.status == "expired"
        assert result.expires_at == int(datetime(2024, 1, 16, 10, 0, 0).timestamp())

    def test_transform_bedrock_response_to_openai_batch_partially_completed(self):
        """Test transforming Bedrock response with PartiallyCompleted status"""
        partial_response = self.sample_bedrock_response.copy()
        partial_response["status"] = "PartiallyCompleted"
        partial_response["endTime"] = datetime(2024, 1, 15, 10, 45, 0)

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=partial_response
        )

        assert result.status == "completed"  # Maps to completed
        assert result.completed_at == int(datetime(2024, 1, 15, 10, 45, 0).timestamp())

    def test_transform_bedrock_response_to_openai_batch_unknown_status(self):
        """Test transforming Bedrock response with unknown status"""
        unknown_response = self.sample_bedrock_response.copy()
        unknown_response["status"] = "UnknownStatus"

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=unknown_response
        )

        assert result.status == "failed"  # Default fallback

    def test_transform_bedrock_response_to_openai_batch_missing_fields(self):
        """Test transforming Bedrock response with missing optional fields"""
        minimal_response = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/minimal-job",
            "status": "InProgress",
            "submitTime": datetime(2024, 1, 15, 10, 0, 0),  # Required for created_at
        }

        result = BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
            bedrock_response=minimal_response
        )

        assert result.id == "minimal-job"
        assert result.status == "in_progress"
        assert result.input_file_id == ""
        assert result.output_file_id is None
        assert result.created_at == int(datetime(2024, 1, 15, 10, 0, 0).timestamp())
        assert result.metadata["job_name"] == ""
        assert result.metadata["model_id"] == ""

    def test_get_batch_id_from_bedrock_response(self):
        """Test extracting batch ID from Bedrock ARN"""
        job_arn = (
            "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-456"
        )
        job_id = BedrockBatchTransformation._get_batch_id_from_bedrock_response(job_arn)
        assert job_id == "test-job-456"

    def test_get_batch_id_from_bedrock_response_empty_arn(self):
        """Test extracting batch ID from empty ARN"""
        job_id = BedrockBatchTransformation._get_batch_id_from_bedrock_response("")
        assert job_id == ""

    def test_get_batch_id_from_bedrock_response_no_slashes(self):
        """Test extracting batch ID from ARN with no slashes"""
        job_id = BedrockBatchTransformation._get_batch_id_from_bedrock_response(
            "simple-arn"
        )
        assert job_id == "simple-arn"

    def test_get_input_file_id_from_bedrock_response(self):
        """Test extracting input file ID from Bedrock response"""
        input_uri = BedrockBatchTransformation._get_input_file_id_from_bedrock_response(
            self.sample_bedrock_response
        )
        assert input_uri == "s3://test-bucket/test-model/test-file.jsonl"

    def test_get_output_file_id_from_bedrock_response(self):
        """Test extracting output file ID from Bedrock response"""
        output_uri = (
            BedrockBatchTransformation._get_output_file_id_from_bedrock_response(
                self.sample_bedrock_response
            )
        )
        assert output_uri == "s3://test-bucket/test-model/output/"

    def test_get_error_information_from_bedrock_response_failed(self):
        """Test extracting error information from failed Bedrock response"""
        failed_response = self.sample_bedrock_response.copy()
        failed_response["status"] = "Failed"
        failed_response["message"] = "Test error message"

        (
            error_file_id,
            errors,
        ) = BedrockBatchTransformation._get_error_information_from_bedrock_response(
            failed_response, "failed"
        )

        assert error_file_id is None
        assert errors is not None
        assert errors["data"][0]["message"] == "Test error message"

    def test_get_error_information_from_bedrock_response_success(self):
        """Test extracting error information from successful Bedrock response"""
        (
            error_file_id,
            errors,
        ) = BedrockBatchTransformation._get_error_information_from_bedrock_response(
            self.sample_bedrock_response, "in_progress"
        )

        assert error_file_id is None
        assert errors is None

    def test_get_metadata_from_bedrock_response(self):
        """Test extracting metadata from Bedrock response"""
        metadata = BedrockBatchTransformation._get_metadata_from_bedrock_response(
            self.sample_bedrock_response,
            "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-123",
        )

        assert metadata["job_name"] == "test-job"
        assert metadata["model_id"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert (
            metadata["job_arn"]
            == "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-123"
        )
        assert metadata["failure_message"] == ""


class TestBedrockBatchTransformationIntegration:
    """Integration tests for Bedrock batch transformation with real data structures"""

    def test_full_transformation_cycle(self):
        """Test complete transformation cycle from OpenAI request to Bedrock response"""
        # Create OpenAI batch request
        openai_request: LiteLLMBatchCreateRequest = {
            "completion_window": "24h",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/input.jsonl",
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "metadata": {
                "job_name": "integration-test",
                "description": "Full cycle test",
            },
        }

        # Transform to Bedrock request
        bedrock_request = BedrockBatchTransformation.transform_openai_batch_request_to_bedrock_job_request(
            req=openai_request,
            s3_input_uri="s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/input.jsonl",
            s3_output_uri="s3://test-bucket/claude-model/output/",
            role_arn="arn:aws:iam::123456789012:role/BedrockBatchRole",
        )

        # Verify Bedrock request structure
        assert bedrock_request["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert bedrock_request["jobName"] == "integration-test"
        assert (
            bedrock_request["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
            == "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/input.jsonl"
        )
        assert (
            bedrock_request["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
            == "s3://test-bucket/claude-model/output/"
        )

        # Simulate Bedrock response
        bedrock_response = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/integration-test-789",
            "jobName": "integration-test",
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "status": "Completed",
            "submitTime": datetime(2024, 1, 15, 12, 0, 0),
            "endTime": datetime(2024, 1, 15, 13, 0, 0),
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3Uri": "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/input.jsonl"
                }
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://test-bucket/claude-model/output/"}
            },
        }

        # Transform back to OpenAI format
        openai_response = (
            BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
                bedrock_response=bedrock_response,
                input_file_id=openai_request["input_file_id"],
            )
        )

        # Verify OpenAI response structure
        assert openai_response.id == "integration-test-789"
        assert openai_response.status == "completed"
        assert (
            openai_response.input_file_id
            == "s3://test-bucket/anthropic.claude-3-sonnet-20240229-v1:0/input.jsonl"
        )
        assert openai_response.output_file_id == "s3://test-bucket/claude-model/output/"
        assert openai_response.created_at == int(
            datetime(2024, 1, 15, 12, 0, 0).timestamp()
        )
        assert openai_response.completed_at == int(
            datetime(2024, 1, 15, 13, 0, 0).timestamp()
        )

    def test_status_mapping_coverage(self):
        """Test that all Bedrock statuses are properly mapped"""
        all_bedrock_statuses = [
            "Submitted",
            "InProgress",
            "Completed",
            "Failed",
            "Stopping",
            "Stopped",
            "PartiallyCompleted",
            "Expired",
            "Validating",
            "Scheduled",
        ]

        expected_openai_statuses = [
            "validating",
            "in_progress",
            "completed",
            "failed",
            "cancelling",
            "cancelled",
            "completed",
            "expired",
            "validating",
            "validating",
        ]

        for bedrock_status, expected_openai_status in zip(
            all_bedrock_statuses, expected_openai_statuses
        ):
            response = {
                "status": bedrock_status,
                "jobArn": "test-arn",
                "submitTime": datetime(
                    2024, 1, 15, 10, 0, 0
                ),  # Required for created_at
            }
            result = (
                BedrockBatchTransformation.transform_bedrock_response_to_openai_batch(
                    response
                )
            )
            assert (
                result.status == expected_openai_status
            ), f"Status {bedrock_status} should map to {expected_openai_status}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
