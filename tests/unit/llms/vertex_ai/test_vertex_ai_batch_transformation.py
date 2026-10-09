import json
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx

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


_MOCK_GCS_FILE_RESPONSE: Final = MappingProxyType(
    {
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
)

_MOCK_VERTEX_BATCH_RESPONSE: Final = MappingProxyType(
    {
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
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://litellm-local/batch-outputs/"}},
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
)


@pytest.mark.asyncio
async def test_vertex_file_upload_create_and_retrieve_batch(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
):
    monkeypatch.setenv("GCS_BUCKET_NAME", "litellm-local")
    monkeypatch.setenv("VERTEXAI_PROJECT", "mock-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    mock_creds: Final = MagicMock(token="mock-token", valid=True, expiry=None)
    monkeypatch.setattr("google.auth.default", lambda *args, **kwargs: (mock_creds, "mock-project"))
    jobs_url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/mock-project/locations/us-central1"
        "/batchPredictionJobs"
    )
    upload_route: Final = respx_mock.post(
        url__startswith="https://storage.googleapis.com/upload/storage/v1/b/litellm-local/o"
    ).mock(return_value=httpx.Response(200, json=dict(_MOCK_GCS_FILE_RESPONSE)))
    create_route: Final = respx_mock.post(jobs_url).mock(
        return_value=httpx.Response(200, json=dict(_MOCK_VERTEX_BATCH_RESPONSE))
    )
    retrieve_route: Final = respx_mock.get(f"{jobs_url}/test-batch-id-456").mock(
        return_value=httpx.Response(200, json=dict(_MOCK_VERTEX_BATCH_RESPONSE))
    )
    gcs_object_uri: Final = (
        "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/"
        "5f7b99ad-9203-4430-98bf-3b45451af4cb"
    )

    file_obj: Final = await litellm.acreate_file(
        file=(
            "vertex_batch.jsonl",
            b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", '
            b'"body": {"model": "gemini-1.5-flash-001", "messages": [{"role": "user", "content": "hi"}]}}\n',
            "application/jsonl",
        ),
        purpose="batch",
        custom_llm_provider="vertex_ai",
    )

    assert file_obj.id == gcs_object_uri
    assert upload_route.call_count == 1
    upload_request: Final = upload_route.calls.last.request
    assert upload_request.url.params["uploadType"] == "media"
    assert upload_request.url.params["name"].startswith(
        "litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/"
    )
    assert upload_request.headers["Content-Type"] == "application/json"
    uploaded_row: Final = json.loads(upload_request.content)
    assert uploaded_row["request"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert uploaded_row["request"]["labels"]["litellm_custom_id"] == "request-1"

    create_batch_response: Final = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=file_obj.id,
        custom_llm_provider="vertex_ai",
        metadata={"key1": "value1", "key2": "value2"},
    )

    create_body: Final = json.loads(create_route.calls.last.request.content)
    assert create_body["inputConfig"] == {"gcsSource": {"uris": [gcs_object_uri]}, "instancesFormat": "jsonl"}
    assert create_body["model"] == "publishers/google/models/gemini-1.5-flash-001"
    assert create_body["outputConfig"]["predictionsFormat"] == "jsonl"
    assert create_body["outputConfig"]["gcsDestination"]["outputUriPrefix"].startswith("gs://litellm-local/")
    assert create_batch_response.id == "test-batch-id-456"
    assert create_batch_response.input_file_id == gcs_object_uri

    retrieved_batch: Final = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider="vertex_ai"
    )

    assert retrieve_route.call_count == 1
    assert retrieved_batch.id == "test-batch-id-456"
    assert retrieved_batch.input_file_id == gcs_object_uri
