import httpx
import pytest

from litellm.llms.openai.vector_store_files.transformation import (
    OpenAIVectorStoreFilesConfig,
)
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture()
def config() -> OpenAIVectorStoreFilesConfig:
    return OpenAIVectorStoreFilesConfig()


def test_validate_environment_sets_headers(config: OpenAIVectorStoreFilesConfig):
    headers: dict = {}
    params = GenericLiteLLMParams(api_key="sk-test")

    result = config.validate_environment(headers=headers, litellm_params=params)

    assert result["Authorization"] == "Bearer sk-test"
    assert result[config.ASSISTANTS_HEADER_KEY] == config.ASSISTANTS_HEADER_VALUE
    assert result["Content-Type"] == "application/json"


def test_get_complete_url(config: OpenAIVectorStoreFilesConfig):
    url = config.get_complete_url(
        api_base="https://api.example.com/v1",
        vector_store_id="vs_123",
        litellm_params={},
    )
    assert url == "https://api.example.com/v1/vector_stores/vs_123/files"


def test_transform_create_request(config: OpenAIVectorStoreFilesConfig):
    api_base = "https://api.example.com/v1/vector_stores/vs_123/files"
    url, payload = config.transform_create_vector_store_file_request(
        vector_store_id="vs_123",
        create_request={
            "file_id": "file-abc",
            "attributes": {"key": "value"},
        },
        api_base=api_base,
    )

    assert url == api_base
    assert payload["file_id"] == "file-abc"
    assert payload["attributes"]["key"] == "value"


def test_transform_list_request(config: OpenAIVectorStoreFilesConfig):
    api_base = "https://api.example.com/v1/vector_stores/vs_123/files"
    url, params = config.transform_list_vector_store_files_request(
        vector_store_id="vs_123",
        query_params={"limit": 2, "order": "asc"},
        api_base=api_base,
    )

    assert url == api_base
    assert params == {"limit": 2, "order": "asc"}


def test_transform_create_response(config: OpenAIVectorStoreFilesConfig):
    response = httpx.Response(
        status_code=200,
        json={
            "id": "file-abc",
            "object": "vector_store.file",
            "vector_store_id": "vs_123",
            "status": "completed",
            "created_at": 123,
        },
    )

    result = config.transform_create_vector_store_file_response(response=response)
    assert result["id"] == "file-abc"
    assert result["status"] == "completed"


def test_transform_list_response(config: OpenAIVectorStoreFilesConfig):
    response = httpx.Response(
        status_code=200,
        json={
            "object": "list",
            "data": [
                {
                    "id": "file-abc",
                    "object": "vector_store.file",
                    "vector_store_id": "vs_123",
                    "status": "completed",
                    "created_at": 123,
                }
            ],
            "first_id": "file-abc",
            "last_id": "file-abc",
            "has_more": False,
        },
    )

    result = config.transform_list_vector_store_files_response(response=response)
    assert result["data"][0]["id"] == "file-abc"
    assert result["has_more"] is False


def test_transform_delete_response(config: OpenAIVectorStoreFilesConfig):
    response = httpx.Response(
        status_code=200,
        json={"id": "file-abc", "object": "vector_store.file.deleted", "deleted": True},
    )

    result = config.transform_delete_vector_store_file_response(response=response)
    assert result["deleted"] is True
