import json
from typing import Final

import httpx
import pytest

from litellm.llms.openai.vector_store_files.transformation import (
    OpenAIVectorStoreFilesConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.vector_store_files import (
    VectorStoreFileCreateRequest,
    VectorStoreFileUpdateRequest,
)


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


def test_get_complete_url_encodes_vector_store_id(
    config: OpenAIVectorStoreFilesConfig,
):
    url = config.get_complete_url(
        api_base="https://api.example.com/v1",
        vector_store_id="../vs_123?x=1#frag",
        litellm_params={},
    )

    assert url == "https://api.example.com/v1/vector_stores/..%2Fvs_123%3Fx%3D1%23frag/files"


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


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "category": "manual",
            "year": 2024,
            "score": 0.75,
            "published": True,
            "archived": False,
            "count": 0,
            "weight": 0.0,
            "label": "",
        },
        {"year": 2024, "archived": False},
        {},
    ],
    ids=["mixed-types", "no-strings", "empty"],
)
def test_create_request_preserves_attribute_types(
    config: OpenAIVectorStoreFilesConfig,
    attributes: dict[str, str | int | float | bool],
) -> None:
    api_base: Final = "https://api.example.com/v1/vector_stores/vs_123/files"
    create_request: Final[VectorStoreFileCreateRequest] = {
        "file_id": "file-abc",
        "attributes": attributes,
    }
    expected_json: Final = json.dumps(create_request, sort_keys=True)

    url, payload = config.transform_create_vector_store_file_request(
        vector_store_id="vs_123",
        create_request=create_request,
        api_base=api_base,
    )

    assert url == api_base
    assert json.dumps(payload, sort_keys=True) == expected_json
    assert json.dumps(create_request, sort_keys=True) == expected_json


@pytest.mark.parametrize(
    "attributes",
    [
        {
            "category": "manual",
            "year": 2024,
            "score": 0.75,
            "published": True,
            "archived": False,
            "count": 0,
            "weight": 0.0,
            "label": "",
        },
        {"year": 2024, "archived": False},
        {},
    ],
    ids=["mixed-types", "no-strings", "empty"],
)
def test_update_request_preserves_attribute_types(
    config: OpenAIVectorStoreFilesConfig,
    attributes: dict[str, str | int | float | bool],
) -> None:
    api_base: Final = "https://api.example.com/v1/vector_stores/vs_123/files"
    update_request: Final[VectorStoreFileUpdateRequest] = {"attributes": attributes}
    expected_json: Final = json.dumps(update_request, sort_keys=True)

    url, payload = config.transform_update_vector_store_file_request(
        vector_store_id="vs_123",
        file_id="file-abc",
        update_request=update_request,
        api_base=api_base,
    )

    assert url == f"{api_base}/file-abc"
    assert json.dumps(payload, sort_keys=True) == expected_json
    assert json.dumps(update_request, sort_keys=True) == expected_json


def test_transform_list_request(config: OpenAIVectorStoreFilesConfig):
    api_base = "https://api.example.com/v1/vector_stores/vs_123/files"
    url, params = config.transform_list_vector_store_files_request(
        vector_store_id="vs_123",
        query_params={"limit": 2, "order": "asc"},
        api_base=api_base,
    )

    assert url == api_base
    assert params == {"limit": 2, "order": "asc"}


def test_transform_file_request_encodes_file_id(config: OpenAIVectorStoreFilesConfig):
    api_base = "https://api.example.com/v1/vector_stores/vs_123/files"

    url, params = config.transform_retrieve_vector_store_file_content_request(
        vector_store_id="vs_123",
        file_id="../../files?x=1#frag",
        api_base=api_base,
    )

    assert url == "https://api.example.com/v1/vector_stores/vs_123/files/..%2F..%2Ffiles%3Fx%3D1%23frag/content"
    assert params == {}


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
