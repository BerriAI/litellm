from unittest.mock import MagicMock, patch
import pytest

import litellm
from litellm.llms.bedrock.batches.handler import BedrockBatchesHandler


@patch("boto3.client")
def test_bedrock_cancel_batch_handler(mock_boto_client):
    mock_client_instance = MagicMock()
    mock_boto_client.return_value = mock_client_instance

    mock_client_instance.stop_model_invocation_job.return_value = {}
    mock_client_instance.get_model_invocation_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
        "status": "Stopping",
        "submitTime": 1700000000,
        "lastModifiedTime": 1700000100,
    }

    res = BedrockBatchesHandler.cancel_batch(
        batch_id="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
        aws_region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    mock_client_instance.stop_model_invocation_job.assert_called_once_with(
        jobIdentifier="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id"
    )
    assert res.status == "cancelling"


@patch("boto3.client")
def test_litellm_cancel_batch_bedrock_dispatcher(mock_boto_client):
    mock_client_instance = MagicMock()
    mock_boto_client.return_value = mock_client_instance

    mock_client_instance.stop_model_invocation_job.return_value = {}
    mock_client_instance.get_model_invocation_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
        "status": "Stopped",
        "submitTime": 1700000000,
        "lastModifiedTime": 1700000100,
    }

    res = litellm.cancel_batch(
        batch_id="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/test-job-id",
        custom_llm_provider="bedrock",
        aws_region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    assert res.status == "cancelled"
