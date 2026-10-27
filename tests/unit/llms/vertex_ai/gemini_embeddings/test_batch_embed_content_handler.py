import json
from typing import Final

import httpx
import pytest
import respx
from google.oauth2.credentials import Credentials

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler import GoogleBatchEmbeddings
from litellm.types.llms.openai import ChatCompletionTextObject
from litellm.types.llms.vertex_ai import GeminiEmbeddingElement, GeminiEmbeddingInput

FILES_URI: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
FILE_METADATA_URL: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
GEMINI_BATCH_EMBEDDINGS_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
)
GEMINI_2_BATCH_EMBEDDINGS_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
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
@pytest.mark.parametrize(
    "text",
    [
        "data:image/png;base64,iVBORw0KGgo= is an example",
        "data:application/octet-stream;base64,SGVsbG8= and surrounding prose",
        "gs://my-bucket/build/output.txt",
        "files/README.md is the docs entry point",
        FILES_URI,
    ],
)
async def test_media_prefixed_text_block_embeds_verbatim_without_a_files_api_lookup(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    text: str,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(GEMINI_2_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]})
    )

    embedding_input: Final = [ChatCompletionTextObject(type="text", text=text)]
    response: Final = (
        litellm.embedding(model="gemini/gemini-embedding-2", input=embedding_input, api_key="gemini-test-key")
        if sync_mode
        else await litellm.aembedding(
            model="gemini/gemini-embedding-2", input=embedding_input, api_key="gemini-test-key"
        )
    )

    assert route.call_count == 1
    assert len(respx_mock.calls) == 1
    assert json.loads(route.calls[0].request.content) == {
        "requests": [
            {
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"text": text}]},
            }
        ]
    }
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2, 0.3]]
    assert response.usage.prompt_tokens == litellm.token_counter(model="gemini-embedding-2", text=text)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "text_block",
    [
        {"type": "text"},
        {"type": "text", "text": 123},
        {"type": "text", "text": "literal text", "unexpected": True},
    ],
)
async def test_invalid_text_block_fails_before_resolving_other_file_references(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
    text_block: dict[str, object],
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    embedding_input: Final = ["files/clip123", text_block]

    with pytest.raises(BadRequestError, match="text") as excinfo:
        (
            litellm.embedding(
                model="gemini/gemini-embedding-2",
                input=embedding_input,
                api_key="gemini-test-key",
                drop_params=False,
            )
            if sync_mode
            else await litellm.aembedding(
                model="gemini/gemini-embedding-2",
                input=embedding_input,
                api_key="gemini-test-key",
                drop_params=False,
            )
        )

    assert excinfo.value.status_code == 400
    assert len(respx_mock.calls) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_drop_params_ignores_extra_text_fields_and_preserves_other_file_references(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    metadata_route: Final = respx_mock.get(FILE_METADATA_URL).mock(
        return_value=httpx.Response(200, json={"mimeType": "video/mp4", "uri": FILES_URI})
    )
    route: Final = respx_mock.post(GEMINI_2_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": [0.1]}, {"values": [0.2]}]})
    )
    embedding_input: Final = [
        {"type": "text", "text": "files/README.md", "unexpected": True},
        "files/clip123",
    ]

    response: Final = (
        litellm.embedding(
            model="gemini/gemini-embedding-2", input=embedding_input, api_key="gemini-test-key", drop_params=True
        )
        if sync_mode
        else await litellm.aembedding(
            model="gemini/gemini-embedding-2", input=embedding_input, api_key="gemini-test-key", drop_params=True
        )
    )

    assert metadata_route.call_count == 1
    assert route.call_count == 1
    assert len(respx_mock.calls) == 2
    assert json.loads(route.calls[0].request.content) == {
        "requests": [
            {"model": "models/gemini-embedding-2", "content": {"parts": [{"text": "files/README.md"}]}},
            {
                "model": "models/gemini-embedding-2",
                "content": {"parts": [{"file_data": {"mime_type": "video/mp4", "file_uri": FILES_URI}}]},
            },
        ]
    }
    assert [item["embedding"] for item in response.data] == [[0.1], [0.2]]


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", [False, True])
async def test_text_and_legacy_media_preserve_embedding_groups_and_order(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    combined: bool,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    media: Final = "data:image/png;base64,iVBORw0KGgo="
    elements: Final[list[GeminiEmbeddingElement]] = [ChatCompletionTextObject(type="text", text=media), media]
    embedding_input: Final[GeminiEmbeddingInput] = [elements] if combined else elements
    vectors: Final = [[0.1]] if combined else [[0.1], [0.2]]
    route: Final = respx_mock.post(GEMINI_2_BATCH_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"embeddings": [{"values": vector} for vector in vectors]})
    )

    response: Final = await litellm.aembedding(
        model="gemini/gemini-embedding-2", input=embedding_input, api_key="gemini-test-key"
    )

    parts: Final = [{"text": media}, {"inline_data": {"mime_type": "image/png", "data": "iVBORw0KGgo="}}]
    groups: Final = [parts] if combined else [[part] for part in parts]
    assert json.loads(route.calls[0].request.content) == {
        "requests": [{"model": "models/gemini-embedding-2", "content": {"parts": group}} for group in groups]
    }
    assert [item["embedding"] for item in response.data] == vectors
    assert [item["index"] for item in response.data] == list(range(len(vectors)))


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_vertex_embedding_sends_media_prefixed_text_as_literal_parts(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    sync_mode: bool,
) -> None:
    from litellm.main import google_batch_embeddings

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    credentials_key: Final = "test-vertex-embedding-credentials"
    monkeypatch.setitem(
        google_batch_embeddings._credentials_project_mapping,
        (credentials_key, "test-project"),
        (Credentials(token="test-vertex-token"), "test-project"),
    )
    url: Final = (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/"
        "locations/us-central1/publishers/google/models/gemini-embedding-2:embedContent"
    )
    route: Final = respx_mock.post(url).mock(
        return_value=httpx.Response(200, json={"embedding": {"values": [0.1, 0.2]}})
    )
    text: Final = "gs://my-bucket/build/output.txt"
    embedding_input: Final = [ChatCompletionTextObject(type="text", text=text)]

    response: Final = (
        litellm.embedding(
            model="vertex_ai/gemini-embedding-2",
            input=embedding_input,
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=credentials_key,
        )
        if sync_mode
        else await litellm.aembedding(
            model="vertex_ai/gemini-embedding-2",
            input=embedding_input,
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_credentials=credentials_key,
        )
    )

    assert route.call_count == 1
    assert len(respx_mock.calls) == 1
    assert json.loads(route.calls[0].request.content) == {"content": {"parts": [{"text": text}]}}
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2]]
    assert response.usage.prompt_tokens == litellm.token_counter(model="gemini-embedding-2", text=text)


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
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with pytest.raises(BadRequestError) as excinfo:
        await litellm.aembedding(model="gemini/gemini-embedding-2", input=bad_input, api_key="gemini-test-key")

    assert excinfo.value.status_code == 400
    assert len(respx_mock.calls) == 0
