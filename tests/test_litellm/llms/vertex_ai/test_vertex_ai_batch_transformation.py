from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.vertex_ai.batches.handler import VertexAIBatchPrediction
from litellm.llms.vertex_ai.batches.transformation import VertexAIBatchTransformation


def test_output_file_id_uses_predictions_jsonl_with_output_info():
    response = {
        "outputInfo": {
            "gcsOutputDirectory": "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-123"
        }
    }

    output_file_id = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )

    assert (
        output_file_id
        == "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-123/predictions.jsonl"
    )


def test_output_file_id_falls_back_to_output_uri_prefix_with_predictions_jsonl():
    response = {
        "outputInfo": {},
        "outputConfig": {
            "gcsDestination": {
                "outputUriPrefix": "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-456"
            }
        },
    }

    output_file_id = VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
        response
    )

    assert (
        output_file_id
        == "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-456/predictions.jsonl"
    )


@pytest.mark.asyncio
def test_vertex_ai_cancel_batch():
    """Test that vertex_ai cancel_batch calls the correct API endpoint"""
    handler = VertexAIBatchPrediction(gcs_bucket_name="test-bucket")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456",
        "state": "JOB_STATE_CANCELLING",
        "createTime": "2024-03-17T10:00:00.000000Z",
        "inputConfig": {
            "gcsSource": {
                "uris": ["gs://test-bucket/input.jsonl"]
            }
        },
        "outputConfig": {
            "gcsDestination": {
                "outputUriPrefix": "gs://test-bucket/output"
            }
        }
    }
    
    with patch("litellm.llms.vertex_ai.batches.handler._get_httpx_client") as mock_client:
        mock_client.return_value.post.return_value = mock_response
        mock_client.return_value.get.return_value = mock_response
        
        with patch.object(handler, "_ensure_access_token") as mock_auth:
            mock_auth.return_value = ("fake-token", "test-project")
            
            response = handler.cancel_batch(
                _is_async=False,
                batch_id="123456",
                api_base=None,
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=600.0,
                max_retries=None,
            )
            
            assert response.id == "123456"
            assert response.status == "cancelling"
            
            mock_client.return_value.post.assert_called_once()
            mock_client.return_value.get.assert_called_once()
            call_args = mock_client.return_value.post.call_args
            assert ":cancel" in call_args.kwargs["url"]


@pytest.mark.asyncio
async def test_litellm_cancel_batch_vertex_ai():
    """Test that litellm.cancel_batch works with vertex_ai provider"""
    mock_response = MagicMock()
    mock_response.id = "batch_123"
    mock_response.status = "cancelling"
    
    with patch("litellm.batches.main.vertex_ai_batches_instance") as mock_instance:
        mock_instance.cancel_batch.return_value = mock_response
        
        response = litellm.cancel_batch(
            batch_id="batch_123",
            custom_llm_provider="vertex_ai",
            vertex_project="test-project",
            vertex_location="us-central1",
        )
        
        assert mock_instance.cancel_batch.called
        assert response.id == "batch_123"
        assert response.status == "cancelling"
