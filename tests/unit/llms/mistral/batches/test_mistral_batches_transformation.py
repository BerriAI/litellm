"""
Regression tests for ``MistralBatchesConfig``, the BaseBatchesConfig implementation
behind ``custom_llm_provider="mistral"`` on /v1/batches.

Locks the request shape Mistral's ``POST /v1/batch/jobs`` accepts (input_files list,
model set on the job, endpoint passed through untouched so ``/v1/ocr`` batches work),
the Mistral -> OpenAI status mapping, request-count and file-id mapping, and auth.
Everything runs for real against canned httpx responses; only the API key env var is
set.
"""

import json

import httpx
import pytest

from litellm.llms.mistral.batches.transformation import MistralBatchesConfig
from litellm.llms.mistral.common_utils import MistralError
from litellm.types.llms.openai import CreateBatchRequest
from litellm.types.utils import LiteLLMBatch, LlmProviders

STATUS_MAP = {
    "QUEUED": "validating",
    "RUNNING": "in_progress",
    "SUCCESS": "completed",
    "FAILED": "failed",
    "TIMEOUT_EXCEEDED": "expired",
    "CANCELLATION_REQUESTED": "cancelling",
    "CANCELLED": "cancelled",
}


def _job(**overrides):
    base = {
        "id": "8ff5e0d1-6bc2-4c3a-9f7d-0d1c2e3f4a5b",
        "object": "batch",
        "input_files": ["c1a2b3d4-0000-4000-8000-000000000001"],
        "endpoint": "/v1/ocr",
        "model": "mistral-ocr-latest",
        "status": "SUCCESS",
        "created_at": 1_757_400_000,
        "started_at": 1_757_400_010,
        "completed_at": 1_757_400_500,
        "total_requests": 3,
        "completed_requests": 3,
        "succeeded_requests": 2,
        "failed_requests": 1,
        "output_file": "out-0000-4000-8000-000000000002",
        "error_file": "err-0000-4000-8000-000000000003",
        "errors": [],
        "metadata": {"job_type": "testing"},
    }
    return {**base, **overrides}


def _response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(payload).encode(),
        request=httpx.Request("GET", "https://api.mistral.ai/v1/batch/jobs/x"),
    )


@pytest.fixture
def config() -> MistralBatchesConfig:
    return MistralBatchesConfig()


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-test")
    return "sk-mistral-test"


def test_custom_llm_provider(config):
    assert config.custom_llm_provider == LlmProviders.MISTRAL


def test_create_request_maps_openai_fields_onto_mistral_job(config):
    data = CreateBatchRequest(
        completion_window="24h",
        endpoint="/v1/ocr",
        input_file_id="file-123",
        metadata={"team": "docs"},
    )
    body = config.transform_create_batch_request(
        model="mistral-ocr-latest", create_batch_data=data, optional_params={}, litellm_params={}
    )
    assert body == {
        "input_files": ("file-123",),
        "endpoint": "/v1/ocr",
        "model": "mistral-ocr-latest",
        "metadata": {"team": "docs"},
    }


def test_create_request_omits_empty_metadata(config):
    data = CreateBatchRequest(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id="file-123",
        metadata=None,
    )
    body = config.transform_create_batch_request(
        model="mistral-small-latest", create_batch_data=data, optional_params={}, litellm_params={}
    )
    assert "metadata" not in body


def test_create_request_requires_input_file_and_endpoint(config):
    with pytest.raises(ValueError, match="input_file_id and endpoint are required"):
        config.transform_create_batch_request(
            model="m",
            create_batch_data=CreateBatchRequest(completion_window="24h"),
            optional_params={},
            litellm_params={},
        )


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (None, "https://api.mistral.ai/v1/batch/jobs"),
        ("https://api.mistral.ai/v1", "https://api.mistral.ai/v1/batch/jobs"),
        ("https://proxy.example.com/", "https://proxy.example.com/v1/batch/jobs"),
    ],
)
def test_create_url(config, api_base, expected):
    url = config.get_complete_batch_url(
        api_base=api_base, api_key="k", model="m", optional_params={}, litellm_params={}, data={}
    )
    assert url == expected


def test_validate_environment_uses_bearer_auth(config, api_key):
    headers = config.validate_environment(
        headers={"x-extra": "1"}, model="m", messages=[], optional_params={}, litellm_params={}
    )
    assert headers == {"x-extra": "1", "Authorization": f"Bearer {api_key}"}


def test_validate_environment_explicit_key_wins(config, api_key):
    headers = config.validate_environment(
        headers={}, model="m", messages=[], optional_params={}, litellm_params={}, api_key="sk-explicit"
    )
    assert headers["Authorization"] == "Bearer sk-explicit"


def test_validate_environment_without_key_raises(config, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing Mistral API Key"):
        config.validate_environment(headers={}, model="m", messages=[], optional_params={}, litellm_params={})


def test_create_response_maps_job_onto_openai_batch(config):
    batch = config.transform_create_batch_response(
        model="mistral-ocr-latest",
        raw_response=_response(_job(status="QUEUED", started_at=None, completed_at=None)),
        logging_obj=None,
        litellm_params={},
    )
    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "8ff5e0d1-6bc2-4c3a-9f7d-0d1c2e3f4a5b"
    assert batch.endpoint == "/v1/ocr"
    assert batch.input_file_id == "c1a2b3d4-0000-4000-8000-000000000001"
    assert batch.status == "validating"
    assert batch.created_at == 1_757_400_000
    assert batch.in_progress_at is None
    assert batch.completed_at is None
    assert batch.metadata == {"job_type": "testing"}


def test_retrieve_request_is_presigned_get_with_auth(config, api_key):
    req = config.transform_retrieve_batch_request(
        batch_id="job/with slash", optional_params={}, litellm_params={"api_base": "https://api.mistral.ai"}
    )
    assert req["method"] == "GET"
    assert req["url"] == "https://api.mistral.ai/v1/batch/jobs/job%2Fwith%20slash"
    assert req["headers"] == {"Authorization": f"Bearer {api_key}"}


def test_retrieve_request_prefers_litellm_params_api_key(config, api_key):
    req = config.transform_retrieve_batch_request(
        batch_id="job-1", optional_params={}, litellm_params={"api_key": "sk-from-deployment"}
    )
    assert req["headers"]["Authorization"] == "Bearer sk-from-deployment"


@pytest.mark.parametrize("mistral_status,openai_status", sorted(STATUS_MAP.items()))
def test_retrieve_response_status_mapping(config, mistral_status, openai_status):
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(_job(status=mistral_status)), logging_obj=None, litellm_params={}
    )
    assert batch.status == openai_status


@pytest.mark.parametrize(
    "mistral_status,populated_field",
    [
        ("SUCCESS", "completed_at"),
        ("FAILED", "failed_at"),
        ("TIMEOUT_EXCEEDED", "expired_at"),
        ("CANCELLED", "cancelled_at"),
    ],
)
def test_retrieve_response_terminal_timestamp_lands_on_matching_field(config, mistral_status, populated_field):
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(_job(status=mistral_status)), logging_obj=None, litellm_params={}
    )
    terminal_fields = {"completed_at", "failed_at", "expired_at", "cancelled_at"}
    assert getattr(batch, populated_field) == 1_757_400_500
    for other in terminal_fields - {populated_field}:
        assert getattr(batch, other) is None
    assert batch.in_progress_at == 1_757_400_010


def test_retrieve_response_maps_counts_and_files(config):
    batch = config.transform_retrieve_batch_response(
        model=None, raw_response=_response(_job()), logging_obj=None, litellm_params={}
    )
    assert batch.request_counts.total == 3
    assert batch.request_counts.completed == 2
    assert batch.request_counts.failed == 1
    assert batch.output_file_id == "out-0000-4000-8000-000000000002"
    assert batch.error_file_id == "err-0000-4000-8000-000000000003"
    assert batch.errors is None


def test_retrieve_response_surfaces_job_errors(config):
    batch = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_response(
            _job(status="FAILED", errors=[{"message": "invalid document", "count": 2}, {"message": "timeout"}])
        ),
        logging_obj=None,
        litellm_params={},
    )
    assert [e.message for e in batch.errors.data] == ["invalid document (x2)", "timeout"]


def test_retrieve_response_without_files_or_input(config):
    batch = config.transform_retrieve_batch_response(
        model=None,
        raw_response=_response(_job(input_files=[], output_file=None, error_file=None, metadata=None)),
        logging_obj=None,
        litellm_params={},
    )
    assert batch.input_file_id == ""
    assert batch.output_file_id is None
    assert batch.error_file_id is None
    assert batch.metadata is None


def test_get_error_class(config):
    err = config.get_error_class("nope", 401, {"x-request-id": "r1"})
    assert isinstance(err, MistralError)
    assert err.status_code == 401
    assert err.message == "nope"
