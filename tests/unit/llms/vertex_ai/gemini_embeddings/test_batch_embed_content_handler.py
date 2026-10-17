import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler import GoogleBatchEmbeddings

FILES_URI: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
FILE_METADATA_URL: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
GEMINI_BATCH_EMBEDDINGS_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
)


def _files_api_metadata(request: httpx.Request) -> httpx.Response:
    if str(request.url) != FILE_METADATA_URL or request.headers.get("x-goog-api-key") != "gemini-key":
        return httpx.Response(404, json={"error": {"message": "no such file"}})
    return httpx.Response(200, json={"mimeType": "video/mp4", "uri": FILES_URI})


@pytest.mark.parametrize("reference", ["files/clip123", FILES_URI])
def test_resolve_file_references_fetches_the_file_metadata_for_both_reference_forms(reference: str) -> None:
    resolved: Final = GoogleBatchEmbeddings()._resolve_file_references(
        input=[reference, "a red bus"],
        api_key="gemini-key",
        sync_handler=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_files_api_metadata))),
    )
    assert resolved == {reference: {"mime_type": "video/mp4", "uri": FILES_URI}}


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("embedding_input", ["good morning from litellm", ["good morning from litellm"]])
@pytest.mark.asyncio
async def test_gemini_embedding_sends_batch_request_and_parses_vectors(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    embedding_input: str | list[str],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(GEMINI_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )
    )

    response: Final = (
        litellm.embedding(model="gemini/gemini-embedding-001", input=embedding_input, api_key="gemini-test-key")
        if sync_mode
        else await litellm.aembedding(
            model="gemini/gemini-embedding-001", input=embedding_input, api_key="gemini-test-key"
        )
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "good morning from litellm"}]},
            }
        ]
    }
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2, 0.3]]
    local_token_count: Final = litellm.token_counter(model="gemini-embedding-001", text="good morning from litellm")
    assert local_token_count > 0
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (
        local_token_count,
        local_token_count,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["files/clip123", FILES_URI])
async def test_async_resolve_file_references_fetches_the_file_metadata_for_both_reference_forms(
    reference: str,
) -> None:
    resolved: Final = await GoogleBatchEmbeddings()._async_resolve_file_references(
        input=[reference, "a red bus"],
        api_key="gemini-key",
        async_handler=AsyncHTTPHandler(transport=httpx.MockTransport(_files_api_metadata)),
    )
    assert resolved == {reference: {"mime_type": "video/mp4", "uri": FILES_URI}}


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_files_prefixed_text_block_embeds_as_text_without_a_files_api_lookup(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
) -> None:
    """Regression for #46719: a typed text block whose text starts with files/ is text, not a file reference.

    No Files API GET may fire for it, and the batch request must carry the text verbatim.
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(GEMINI_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]})
    )

    embedding_input: Final = [{"type": "text", "text": "files/README.md is the docs entry point"}]
    response: Final = (
        litellm.embedding(model="gemini/gemini-embedding-001", input=embedding_input, api_key="gemini-test-key")
        if sync_mode
        else await litellm.aembedding(
            model="gemini/gemini-embedding-001", input=embedding_input, api_key="gemini-test-key"
        )
    )

    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": "files/README.md is the docs entry point"}]},
            }
        ]
    }
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_input",
    [
        ["data:application/octet-stream;base64,SGVsbG8=", "plain text chunk"],
        ["gs://my-bucket/build/output.txt", "plain text chunk"],
    ],
)
async def test_media_prefix_validation_fails_as_bad_request_without_sending_a_request(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    bad_input: list[str],
) -> None:
    """Regression for #46719: local input validation is a 400 BadRequestError, not a 500 connection error.

    These are client input errors raised before any request: retry logic must not
    retry them until its time limit, and no HTTP call may be made.
    """
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(GEMINI_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]})
    )

    with pytest.raises(BadRequestError) as excinfo:
        await litellm.aembedding(model="gemini/gemini-embedding-001", input=bad_input, api_key="gemini-test-key")

    assert excinfo.value.status_code == 400
    assert route.call_count == 0
