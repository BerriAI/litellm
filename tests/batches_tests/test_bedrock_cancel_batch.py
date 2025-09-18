"""
Test Bedrock cancel batch functionality
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.types.llms.openai import CancelBatchRequest
from openai.types.batch import Batch


class TestBedrockCancelBatch:
    """
    Test Bedrock cancel batch functionality
    """

    def test_bedrock_cancel_batch_transformation_request(self):
        """
        Test that cancel batch request is properly transformed for Bedrock StopModelInvocationJob API
        """
        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
        
        config = BedrockBatchesConfig()
        batch_id = "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-name"
        optional_params = {}
        litellm_params = {}
        
        with patch.object(config.common_utils, 'sign_aws_request') as mock_sign:
            mock_sign.return_value = ({"Authorization": "AWS4-HMAC-SHA256 ..."}, b'{}')
            
            result = config.transform_cancel_batch_request(
                batch_id=batch_id,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )
            
            # Verify the URL format
            expected_url = "https://bedrock.us-west-2.amazonaws.com/model-invocation-job/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Amodel-invocation-job/test-job-name/stop"
            assert result["url"] == expected_url
            assert result["method"] == "POST"
            assert "Authorization" in result["headers"]
            
            # Verify AWS signing was called with correct parameters
            mock_sign.assert_called_once_with(
                service_name="bedrock",
                data={},
                endpoint_url=expected_url,
                optional_params=optional_params,
                method="POST"
            )

    def test_bedrock_cancel_batch_transformation_response(self):
        """
        Test that cancel batch response is properly transformed to OpenAI Batch format
        """
        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
        
        config = BedrockBatchesConfig()
        
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "jobArn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-name",
            "status": "Stopping",
            "lastModifiedTime": "2024-01-15T10:30:00Z",
            "submitTime": "2024-01-15T09:00:00Z",
        }
        
        result = config.transform_cancel_batch_response(
            model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            raw_response=mock_response,
            logging_obj=None,
            litellm_params={},
        )
        
        # Verify the result is a Batch object with correct properties
        assert isinstance(result, Batch)
        assert result.id == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-name"
        assert result.object == "batch"
        assert result.status == "cancelling"  # "Stopping" maps to "cancelling"
        assert result.metadata["bedrock_job_arn"] == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job-name"
        assert result.metadata["bedrock_status"] == "Stopping"

    def test_bedrock_cancel_batch_invalid_arn(self):
        """
        Test that invalid ARN format raises appropriate error
        """
        from litellm.llms.bedrock.batches.transformation import BedrockBatchesConfig
        
        config = BedrockBatchesConfig()
        invalid_batch_id = "invalid-batch-id"
        
        with pytest.raises(ValueError, match="Invalid batch_id format. Expected ARN"):
            config.transform_cancel_batch_request(
                batch_id=invalid_batch_id,
                optional_params={},
                litellm_params={},
            )

    @patch('litellm.batches.main.bedrock_batches_instance')
    def test_litellm_cancel_batch_bedrock_provider(self, mock_bedrock_instance):
        """
        Test that litellm.cancel_batch works with bedrock provider
        """
        # Mock the bedrock instance cancel_batch method
        mock_batch = Batch(
            id="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job",
            object="batch",
            endpoint="/v1/chat/completions",
            errors=None,
            input_file_id="",
            completion_window="24h",
            status="cancelling",  # type: ignore
            output_file_id=None,
            error_file_id=None,
            created_at=1640995200,
            in_progress_at=None,
            expires_at=None,
            finalizing_at=None,
            completed_at=None,
            failed_at=None,
            expired_at=None,
            cancelling_at=1640995800,
            cancelled_at=None,
            request_counts=None,
            metadata={"bedrock_job_arn": "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job"},
        )
        mock_bedrock_instance.cancel_batch.return_value = mock_batch
        
        # Call litellm.cancel_batch with bedrock provider
        result = litellm.cancel_batch(
            batch_id="arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job",
            custom_llm_provider="bedrock",
            model="us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        
        # Verify the result
        assert result.id == "arn:aws:bedrock:us-west-2:123456789012:model-invocation-job/test-job"
        assert result.status == "cancelling"
        
        # Verify the bedrock instance was called
        mock_bedrock_instance.cancel_batch.assert_called_once()

    def test_bedrock_cancel_batch_unsupported_provider_error_updated(self):
        """
        Test that the error message for unsupported providers now includes bedrock as supported
        """
        with pytest.raises(litellm.exceptions.BadRequestError) as exc_info:
            litellm.cancel_batch(
                batch_id="test-batch-id",
                custom_llm_provider="unsupported_provider"  # type: ignore
            )
        
        error_message = str(exc_info.value.message)
        assert "bedrock" in error_message
        assert "Only 'openai', 'azure', and 'bedrock' are supported" in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])