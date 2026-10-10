from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.llms.openai import OpenAIFileObject

NATIVE_VERTEX_ROWS: Final = (
    b'{"request": {"contents": [{"role": "user", "parts": [{"text": "Who won the 2024 Tour de France?"}]}],'
    b' "tools": [{"googleSearch": {"excludeDomains": ["example.com"]}}]}}\n'
    b'{"request": {"contents": [{"role": "user", "parts": [{"text": "What is the tallest building in Tokyo?"}]}],'
    b' "tools": [{"googleSearch": {}}]}}\n'
)


@pytest.mark.parametrize(
    "custom_llm_provider, purpose",
    [("openai", "batch"), ("vertex_ai", "assistants")],
    ids=["non-vertex-provider", "non-batch-purpose"],
)
def test_create_file_passthrough_is_rejected_outside_a_vertex_batch(custom_llm_provider, purpose):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.create_file(
            file=("batch.jsonl", b'{"request": {"contents": []}}\n', "application/jsonl"),
            purpose=purpose,
            custom_llm_provider=custom_llm_provider,
            passthrough=True,
            api_key="sk-test",
            api_base="http://127.0.0.1:9",
        )

    assert "vertex_ai" in str(exc_info.value)
    assert "batch" in str(exc_info.value)


def _gcs_upload_transport(uploads: list[httpx.Request]) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        uploads.append(request)
        object_name: Final = parse_qs(urlparse(str(request.url)).query)["name"][0]
        return httpx.Response(
            200,
            json={
                "id": f"my-bucket/{object_name}/1758585600000000",
                "name": object_name,
                "size": str(len(request.read())),
                "timeCreated": "2026-09-23T00:00:00.000Z",
            },
        )

    return httpx.MockTransport(respond)


def test_create_file_passthrough_kwarg_ships_native_rows_byte_for_byte_under_the_passthrough_prefix():
    uploads: Final[list[httpx.Request]] = []
    file_object = litellm.create_file(
        file=("batch.jsonl", NATIVE_VERTEX_ROWS, "application/jsonl"),
        purpose="batch",
        custom_llm_provider="vertex_ai",
        passthrough=True,
        model="vertex_ai/gemini-2.5-flash",
        gcs_bucket_name="my-bucket",
        api_key="test-token",
        client=HTTPHandler(client=httpx.Client(transport=_gcs_upload_transport(uploads))),
    )
    (upload,) = uploads
    object_name: Final = parse_qs(urlparse(str(upload.url)).query)["name"][0]
    assert upload.read() == NATIVE_VERTEX_ROWS
    assert object_name.startswith("litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/")
    assert file_object.id == f"gs://my-bucket/{object_name}"


OPENAI_FILES_API_BASE: Final = "https://files.test/v1"
PROVIDER_FILE: Final = {
    "id": "file-abc123",
    "bytes": 120,
    "created_at": 1700000000,
    "filename": "batch.jsonl",
    "object": "file",
    "purpose": "batch",
}
UNSET_OPTIONAL_FIELDS: Final = {"status": None, "expires_at": None, "status_details": None}


@pytest.mark.parametrize(
    "provider_extras, expected",
    [
        ({}, {**PROVIDER_FILE, **UNSET_OPTIONAL_FIELDS}),
        (
            {"status": "processed", "expires_at": 1800000000, "status_details": "ok", "provider_only": {"a": [1]}},
            {**PROVIDER_FILE, "status": "processed", "expires_at": 1800000000, "status_details": "ok"},
        ),
    ],
    ids=["required-fields-only", "optional-and-unknown-fields"],
)
@respx.mock
async def test_afile_retrieve_returns_the_provider_file_as_an_openai_file_object(provider_extras, expected):
    respx.get(f"{OPENAI_FILES_API_BASE}/files/file-abc123").respond(200, json={**PROVIDER_FILE, **provider_extras})

    file_object: Final = await litellm.afile_retrieve(
        file_id="file-abc123", custom_llm_provider="openai", api_key="sk-test", api_base=OPENAI_FILES_API_BASE
    )

    assert type(file_object) is OpenAIFileObject
    assert file_object.model_dump() == expected


@respx.mock
async def test_afile_retrieve_rejects_a_provider_file_without_its_size():
    provider_file: Final = {key: value for key, value in PROVIDER_FILE.items() if key != "bytes"}
    respx.get(f"{OPENAI_FILES_API_BASE}/files/file-abc123").respond(200, json=provider_file)

    with pytest.raises(ValidationError) as exc_info:
        await litellm.afile_retrieve(
            file_id="file-abc123", custom_llm_provider="openai", api_key="sk-test", api_base=OPENAI_FILES_API_BASE
        )

    assert exc_info.value.title == "OpenAIFileObject"
    assert [error["loc"] for error in exc_info.value.errors()] == [("bytes",)]


_FILE_BODY: Final = b'{"prompt": "Hello", "completion": "Hi"}'
_FINE_TUNE_FILE_JSON: Final = MappingProxyType(
    {
        "id": "file-abc123",
        "object": "file",
        "bytes": len(_FILE_BODY),
        "created_at": 1699000000,
        "filename": "mydata.jsonl",
        "purpose": "fine-tune",
    }
)


@pytest.mark.asyncio
async def test_openai_file_operations_roundtrip(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    files_route: Final = respx_mock.post("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json=dict(_FINE_TUNE_FILE_JSON))
    )
    list_route: Final = respx_mock.get("https://api.openai.com/v1/files").mock(
        return_value=httpx.Response(200, json={"object": "list", "data": [dict(_FINE_TUNE_FILE_JSON)]})
    )
    retrieve_route: Final = respx_mock.get("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(200, json=dict(_FINE_TUNE_FILE_JSON))
    )
    content_route: Final = respx_mock.get("https://api.openai.com/v1/files/file-abc123/content").mock(
        return_value=httpx.Response(200, content=_FILE_BODY)
    )
    delete_route: Final = respx_mock.delete("https://api.openai.com/v1/files/file-abc123").mock(
        return_value=httpx.Response(200, json={"id": "file-abc123", "object": "file", "deleted": True})
    )

    uploaded: Final = await litellm.acreate_file(
        file=("mydata.jsonl", _FILE_BODY), purpose="fine-tune", custom_llm_provider="openai", api_key="fake-key"
    )
    assert files_route.call_count == 1
    upload_body: Final = files_route.calls.last.request.content
    assert b'name="purpose"\r\n\r\nfine-tune' in upload_body
    assert b'filename="mydata.jsonl"' in upload_body
    assert _FILE_BODY in upload_body
    assert uploaded.id == "file-abc123"

    listed: Final = await litellm.afile_list(custom_llm_provider="openai", api_key="fake-key")
    assert list_route.call_count == 1
    assert [file.id for file in listed.data] == ["file-abc123"]

    retrieved: Final = await litellm.afile_retrieve(
        file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert retrieve_route.call_count == 1
    assert retrieved.filename == "mydata.jsonl"
    assert retrieved.purpose == "fine-tune"

    content: Final = await litellm.afile_content(
        file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key"
    )
    assert content_route.call_count == 1
    assert content.content == _FILE_BODY

    deleted: Final = await litellm.afile_delete(file_id="file-abc123", custom_llm_provider="openai", api_key="fake-key")
    assert delete_route.call_count == 1
    assert deleted.id == "file-abc123"
    assert deleted.deleted is True
