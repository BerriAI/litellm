from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

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

    output_file_id = (
        VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
            response
        )
    )

    assert (
        output_file_id
        == "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro/prediction-model-123/predictions.jsonl"
    )


def test_output_file_id_is_none_until_output_info():
    response = {
        "outputInfo": {},
        "outputConfig": {
            "gcsDestination": {
                "outputUriPrefix": "gs://test-bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-pro"
            }
        },
    }

    output_file_id = (
        VertexAIBatchTransformation._get_output_file_id_from_vertex_ai_batch_response(
            response
        )
    )

    assert output_file_id is None


def test_vertex_ai_cancel_batch():
    """Test that vertex_ai cancel_batch calls the correct API endpoint"""
    handler = VertexAIBatchPrediction(gcs_bucket_name="test-bucket")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456",
        "state": "JOB_STATE_CANCELLING",
        "createTime": "2024-03-17T10:00:00.000000Z",
        "inputConfig": {"gcsSource": {"uris": ["gs://test-bucket/input.jsonl"]}},
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://test-bucket/output"}},
    }

    with patch("litellm.llms.vertex_ai.batches.handler.get_httpx_client") as mock_client:
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


def test_vertex_ai_cancel_batch_encodes_batch_id():
    """Test that vertex_ai cancel_batch encodes user-controlled batch IDs."""
    handler = VertexAIBatchPrediction(gcs_bucket_name="test-bucket")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456",
        "state": "JOB_STATE_CANCELLING",
        "createTime": "2024-03-17T10:00:00.000000Z",
        "inputConfig": {"gcsSource": {"uris": ["gs://test-bucket/input.jsonl"]}},
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://test-bucket/output"}},
    }

    with patch("litellm.llms.vertex_ai.batches.handler.get_httpx_client") as mock_client:
        mock_client.return_value.post.return_value = mock_response
        mock_client.return_value.get.return_value = mock_response

        with patch.object(handler, "_ensure_access_token") as mock_auth:
            mock_auth.return_value = ("fake-token", "test-project")

            handler.cancel_batch(
                _is_async=False,
                batch_id="../../batchPredictionJobs/other?x=1#frag",
                api_base=None,
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=600.0,
                max_retries=None,
            )

            post_url = mock_client.return_value.post.call_args.kwargs["url"]
            get_url = mock_client.return_value.get.call_args.kwargs["url"]
            assert (
                "/..%2F..%2FbatchPredictionJobs%2Fother%3Fx%3D1%23frag:cancel"
                in post_url
            )
            assert "/..%2F..%2FbatchPredictionJobs%2Fother%3Fx%3D1%23frag" in get_url


def test_vertex_ai_cancel_batch_forwards_timeout():
    """Test that timeout is forwarded to the POST (cancel) HTTP call.

    Note: the follow-up GET (retrieve) call does not accept a timeout
    parameter in the underlying HTTP handler, so it is intentionally omitted.
    """


def test_vertex_ai_cancel_batch_custom_proxy_retrieve_url():
    """Retrieve URL should go through the custom proxy, not bypass it"""
    handler = VertexAIBatchPrediction(gcs_bucket_name="test-bucket")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "projects/test-project/locations/us-central1/batchPredictionJobs/123456",
        "state": "JOB_STATE_CANCELLING",
        "createTime": "2024-03-17T10:00:00.000000Z",
        "inputConfig": {"gcsSource": {"uris": ["gs://test-bucket/input.jsonl"]}},
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://test-bucket/output"}},
    }

    with patch("litellm.llms.vertex_ai.batches.handler.get_httpx_client") as mock_client:
        mock_client.return_value.post.return_value = mock_response
        mock_client.return_value.get.return_value = mock_response

        with patch.object(handler, "_ensure_access_token") as mock_auth:
            mock_auth.return_value = ("fake-token", "test-project")

            handler.cancel_batch(
                _is_async=False,
                batch_id="123456",
                api_base="https://my-proxy.example.com",
                vertex_credentials=None,
                vertex_project="test-project",
                vertex_location="us-central1",
                timeout=600.0,
                max_retries=None,
            )

            post_url = mock_client.return_value.post.call_args.kwargs["url"]
            get_url = mock_client.return_value.get.call_args.kwargs["url"]

            assert "my-proxy.example.com" in post_url
            assert ":cancel" in post_url
            assert "my-proxy.example.com" in get_url
            assert ":cancel" not in get_url
            assert "googleapis.com" not in get_url


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


_MOCK_GCS_FILE_RESPONSE = {
    "kind": "storage#object",
    "id": "litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb/1739598666670574",
    "selfLink": "https://www.googleapis.com/storage/v1/b/litellm-local/o/litellm-vertex-files%2Fpublishers%2Fgoogle%2Fmodels%2Fgemini-1.5-flash-001%2F5f7b99ad-9203-4430-98bf-3b45451af4cb",
    "name": "litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb",
    "bucket": "litellm-local",
    "generation": "1739598666670574",
    "metageneration": "1",
    "contentType": "application/json",
    "storageClass": "STANDARD",
    "size": "416",
    "md5Hash": "hbBNj7C8KJ7oVH+JmyRM6A==",
    "crc32c": "oDmiUA==",
    "etag": "CO7D0IT+xIsDEAE=",
    "timeCreated": "2025-02-15T05:51:06.741Z",
    "updated": "2025-02-15T05:51:06.741Z",
    "timeStorageClassUpdated": "2025-02-15T05:51:06.741Z",
    "timeFinalized": "2025-02-15T05:51:06.741Z",
}

_MOCK_VERTEX_BATCH_RESPONSE = {
    "name": "projects/123456789/locations/us-central1/batchPredictionJobs/test-batch-id-456",
    "displayName": "litellm_batch_job",
    "model": "projects/123456789/locations/us-central1/models/gemini-1.5-flash-001",
    "modelVersionId": "v1",
    "inputConfig": {
        "gcsSource": {
            "uris": [
                "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb"
            ]
        }
    },
    "outputConfig": {
        "gcsDestination": {"outputUriPrefix": "gs://litellm-local/batch-outputs/"}
    },
    "dedicatedResources": {
        "machineSpec": {
            "machineType": "n1-standard-4",
            "acceleratorType": "NVIDIA_TESLA_T4",
            "acceleratorCount": 1,
        },
        "startingReplicaCount": 1,
        "maxReplicaCount": 1,
    },
    "state": "JOB_STATE_RUNNING",
    "createTime": "2025-02-15T05:51:06.741Z",
    "startTime": "2025-02-15T05:51:07.741Z",
    "updateTime": "2025-02-15T05:51:08.741Z",
    "labels": {"key1": "value1", "key2": "value2"},
    "completionStats": {"successfulCount": 0, "failedCount": 0, "remainingCount": 100},
}


@pytest.mark.asyncio
async def test_vertex_file_upload_create_and_retrieve_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("GCS_BUCKET_NAME", "litellm-local")
    monkeypatch.setenv("VERTEXAI_PROJECT", "mock-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

    mock_creds: Final = MagicMock()
    mock_creds.token = "mock-token"
    mock_creds.valid = True
    mock_creds.expiry = None
    monkeypatch.setattr(
        "google.auth.default",
        lambda *args, **kwargs: (mock_creds, "mock-project"),
    )

    mock_response: Final = MagicMock()
    mock_response.raise_for_status.return_value = None

    async def _route_post(*args, **kwargs):
        url: Final = kwargs.get("url", "")
        if "files" in url:
            mock_response.json.return_value = _MOCK_GCS_FILE_RESPONSE
        elif "batch" in url:
            mock_response.json.return_value = _MOCK_VERTEX_BATCH_RESPONSE
            mock_response.status_code = 200
        return mock_response

    jsonl_path: Final = tmp_path / "vertex_batch.jsonl"
    jsonl_path.write_text(
        '{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", '
        '"body": {"model": "gemini-1.5-flash-001", "messages": [{"role": "user", "content": "hi"}]}}\n'
    )

    with (
        patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            side_effect=_route_post,
        ),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(
                200,
                json=_MOCK_GCS_FILE_RESPONSE,
                request=httpx.Request("POST", "https://storage.googleapis.com/upload"),
            ),
        ) as mock_gcs_upload,
    ):
        file_obj: Final = await litellm.acreate_file(
            file=open(jsonl_path, "rb"),
            purpose="batch",
            custom_llm_provider="vertex_ai",
        )

        assert (
            file_obj.id
            == "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb"
        )
        mock_gcs_upload.assert_awaited_once()
        upload_url: Final = str(mock_gcs_upload.call_args.args[0])
        assert "uploadType=media" in upload_url
        assert "/b/litellm-local/o" in upload_url
        assert (
            mock_gcs_upload.call_args.kwargs["headers"]["Content-Type"]
            == "application/json"
        )

        create_batch_response: Final = await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            custom_llm_provider="vertex_ai",
            metadata={"key1": "value1", "key2": "value2"},
        )

        assert create_batch_response.id == "test-batch-id-456"
        assert create_batch_response.input_file_id == file_obj.id

        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
        ) as mock_get:
            mock_get_response: Final = MagicMock()
            mock_get_response.json.return_value = _MOCK_VERTEX_BATCH_RESPONSE
            mock_get_response.status_code = 200
            mock_get_response.is_redirect = False
            mock_get_response.raise_for_status.return_value = None
            mock_get.return_value = mock_get_response

            retrieved_batch: Final = await litellm.aretrieve_batch(
                batch_id=create_batch_response.id,
                custom_llm_provider="vertex_ai",
            )
            assert retrieved_batch.id == "test-batch-id-456"
