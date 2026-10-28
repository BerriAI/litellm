import json
from typing import Final

import httpx
import pytest
import respx
from google.oauth2.credentials import Credentials

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler import GoogleBatchEmbeddings
from litellm.main import google_batch_embeddings

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
@pytest.mark.parametrize("model", ["gemini/gemini-embedding-001", "vertex_ai/gemini-embedding-2"])
@pytest.mark.asyncio
async def test_google_embedding_applies_timeout_and_parses_vectors(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    embedding_input: str | list[str],
    model: str,
) -> None:
    provider, model_name = model.split("/", 1)
    is_vertex: Final = provider == "vertex_ai"
    timeout: Final = 0.125
    vertex_params: Final = (
        {
            "vertex_project": "embedding-timeout-tests",
            "vertex_location": "us-central1",
            "vertex_credentials": "embedding-timeout-test-credentials",
        }
        if is_vertex else {}
    )
    monkeypatch.setitem(
        google_batch_embeddings._credentials_project_mapping,
        ("embedding-timeout-test-credentials", "embedding-timeout-tests"),
        (Credentials(token="test-vertex-token"), "embedding-timeout-tests"),
    )
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/embedding-timeout-tests/"
        "locations/us-central1/publishers/google/models/gemini-embedding-2:embedContent"
        if is_vertex else GEMINI_BATCH_EMBEDDINGS_URL
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={"embedding": {"values": [0.1, 0.2, 0.3]}}
            if is_vertex else {"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )
    )

    response: Final = (
        litellm.embedding(
            model=model, input=embedding_input, api_key="gemini-test-key", timeout=timeout, **vertex_params
        )
        if sync_mode
        else await litellm.aembedding(
            model=model, input=embedding_input, api_key="gemini-test-key", timeout=timeout, **vertex_params
        )
    )

    assert route.call_count == 1
    content: Final = {"parts": [{"text": "good morning from litellm"}]}
    expected_body: Final = (
        {"content": content}
        if is_vertex else {"requests": [{"model": f"models/{model_name}", "content": content}]}
    )
    assert json.loads(route.calls[0].request.content) == expected_body
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2, 0.3]]
    local_token_count: Final = litellm.token_counter(
        model=model_name, text="good morning from litellm"
    )
    assert local_token_count > 0
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (
        local_token_count,
        local_token_count,
    )
    assert route.calls[0].request.extensions["timeout"] == httpx.Timeout(timeout).as_dict()


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
