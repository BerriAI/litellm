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
