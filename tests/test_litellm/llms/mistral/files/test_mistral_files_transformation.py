"""
Regression tests for ``MistralFilesConfig``, the BaseFilesConfig implementation behind
``custom_llm_provider="mistral"`` on /v1/files.

Locks the URL routing for each file operation, the multipart upload shape Mistral's
``POST /v1/files`` accepts (purpose restricted to fine-tune/batch/ocr), and the
Mistral -> OpenAI file object mapping. Runs against canned httpx responses.
"""

import json

import httpx
import pytest
from openai.types.file_deleted import FileDeleted

from litellm.llms.mistral.files.transformation import MistralFilesConfig
from litellm.types.llms.openai import CreateFileRequest, FileContentRequest, OpenAIFileObject
from litellm.types.utils import LlmProviders

FILE_ID = "497f6eca-6276-4993-bfeb-53cbbbba6f09"


def _file(**overrides):
    base = {
        "id": FILE_ID,
        "object": "file",
        "bytes": 13000,
        "created_at": 1_716_963_433,
        "filename": "batch_input.jsonl",
        "purpose": "batch",
        "sample_type": "batch_request",
        "num_lines": 3,
        "source": "upload",
    }
    return {**base, **overrides}


def _response(payload) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        content=json.dumps(payload).encode(),
        request=httpx.Request("GET", "https://api.mistral.ai/v1/files"),
    )


@pytest.fixture
def config() -> MistralFilesConfig:
    return MistralFilesConfig()


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-test")
    return "sk-mistral-test"


def test_custom_llm_provider(config):
    assert config.custom_llm_provider == LlmProviders.MISTRAL


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (None, "https://api.mistral.ai/v1/files"),
        ("https://api.mistral.ai/v1/", "https://api.mistral.ai/v1/files"),
        ("https://proxy.example.com", "https://proxy.example.com/v1/files"),
    ],
)
def test_upload_url(config, api_base, expected):
    url = config.get_complete_url(api_base=api_base, api_key="k", model="", optional_params={}, litellm_params={})
    assert url == expected


def test_validate_environment_uses_bearer_auth(config, api_key):
    headers = config.validate_environment(headers={}, model="", messages=[], optional_params={}, litellm_params={})
    assert headers == {"Authorization": f"Bearer {api_key}"}


def test_upload_request_is_multipart_with_batch_purpose(config):
    body = config.transform_create_file_request(
        model="",
        create_file_data=CreateFileRequest(
            file=("in.jsonl", b'{"custom_id":"0"}\n', "application/jsonl"), purpose="batch"
        ),
        optional_params={},
        litellm_params={},
    )
    assert body == {
        "file": ("in.jsonl", b'{"custom_id":"0"}\n', "application/jsonl"),
        "purpose": (None, "batch"),
    }


@pytest.mark.parametrize("purpose", ["batch", "fine-tune", "ocr"])
def test_upload_request_passes_mistral_purposes_through(config, purpose):
    body = config.transform_create_file_request(
        model="",
        create_file_data=CreateFileRequest(file=("f.bin", b"x"), purpose=purpose),
        optional_params={},
        litellm_params={},
    )
    assert body["purpose"] == (None, purpose)


@pytest.mark.parametrize("purpose", ["assistants", "user_data", "vision", "evals"])
def test_upload_request_rejects_purposes_mistral_lacks(config, purpose):
    """Regression: these used to be silently rewritten to ``batch``, so an upload that skipped the
    proxy's batch-only validation and guardrails still landed on Mistral as a batch input file."""
    with pytest.raises(ValueError, match=f"purpose={purpose!r}"):
        config.transform_create_file_request(
            model="",
            create_file_data=CreateFileRequest(file=("f.bin", b"x"), purpose=purpose),
            optional_params={},
            litellm_params={},
        )


def test_upload_request_requires_file(config):
    with pytest.raises(ValueError, match="File data is required"):
        config.transform_create_file_request(
            model="", create_file_data=CreateFileRequest(purpose="batch"), optional_params={}, litellm_params={}
        )


def test_upload_response_maps_onto_openai_file_object(config):
    obj = config.transform_create_file_response(
        model=None, raw_response=_response(_file()), logging_obj=None, litellm_params={}
    )
    assert obj == OpenAIFileObject(
        id=FILE_ID,
        bytes=13000,
        created_at=1_716_963_433,
        filename="batch_input.jsonl",
        object="file",
        purpose="batch",
        status="uploaded",
    )


def test_file_response_with_ocr_purpose_maps_onto_user_data(config):
    obj = config.transform_retrieve_file_response(
        raw_response=_response(_file(purpose="ocr", expires_at=1_800_000_000)), logging_obj=None, litellm_params={}
    )
    assert obj.purpose == "user_data"
    assert obj.expires_at == 1_800_000_000


@pytest.mark.parametrize(
    "method,suffix",
    [
        ("transform_retrieve_file_request", ""),
        ("transform_delete_file_request", ""),
    ],
)
def test_single_file_urls_encode_id_and_honor_api_base(config, method, suffix):
    url, params = getattr(config, method)(
        file_id="id/with slash", optional_params={}, litellm_params={"api_base": "https://mistral.internal/v1"}
    )
    assert url == f"https://mistral.internal/v1/files/id%2Fwith%20slash{suffix}"
    assert params == {}


def test_file_content_url(config):
    url, params = config.transform_file_content_request(
        file_content_request=FileContentRequest(file_id=FILE_ID), optional_params={}, litellm_params={}
    )
    assert url == f"https://api.mistral.ai/v1/files/{FILE_ID}/content"
    assert params == {}


def test_file_content_response_is_binary_passthrough(config):
    raw = httpx.Response(
        200, content=b'{"custom_id":"0","response":{"status_code":200}}\n', request=httpx.Request("GET", "https://x")
    )
    out = config.transform_file_content_response(raw_response=raw, logging_obj=None, litellm_params={})
    assert out.content == b'{"custom_id":"0","response":{"status_code":200}}\n'


def test_delete_response(config):
    out = config.transform_delete_file_response(
        raw_response=_response({"id": FILE_ID, "object": "file", "deleted": True}), logging_obj=None, litellm_params={}
    )
    assert out == FileDeleted(id=FILE_ID, deleted=True, object="file")


def test_list_request_filters_by_mapped_purpose(config):
    url, params = config.transform_list_files_request(purpose="batch", optional_params={}, litellm_params={})
    assert url == "https://api.mistral.ai/v1/files"
    assert params == {"purpose": "batch"}
    _, no_params = config.transform_list_files_request(purpose=None, optional_params={}, litellm_params={})
    assert no_params == {}


def test_list_request_rejects_purposes_mistral_lacks(config):
    with pytest.raises(ValueError, match="purpose='assistants'"):
        config.transform_list_files_request(purpose="assistants", optional_params={}, litellm_params={})


def test_list_response(config):
    out = config.transform_list_files_response(
        raw_response=_response(
            {"data": [_file(), _file(id="second", filename="b.jsonl")], "object": "list", "total": 2}
        ),
        logging_obj=None,
        litellm_params={},
    )
    assert [f.id for f in out] == [FILE_ID, "second"]
    assert out[1].filename == "b.jsonl"
