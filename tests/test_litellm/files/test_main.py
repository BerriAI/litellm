from typing import Final
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler

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
