import json
from typing import Final

import httpx
import pytest
import respx
from google.oauth2.credentials import Credentials

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.gemini_embeddings.batch_embed_content_handler import GoogleBatchEmbeddings
from litellm.main import google_batch_embeddings
from litellm.types.utils import EmbeddingResponse

FILES_URI: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
FILE_METADATA_URL: Final = "https://generativelanguage.googleapis.com/v1beta/files/clip123"
GEMINI_BATCH_EMBEDDINGS_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
)
ERROR_TEST_PROJECT: Final = "embedding-error-tests"
ERROR_TEST_CREDENTIALS: Final = "embedding-error-test-credentials"
VERTEX_EMBED_CONTENT_URL: Final = (
    f"https://us-central1-aiplatform.googleapis.com/v1/projects/{ERROR_TEST_PROJECT}/"
    "locations/us-central1/publishers/google/models/gemini-embedding-2:embedContent"
)
SHARED_HANDLER_ROUTES: Final = (
    pytest.param("vertex_ai/gemini-embedding-2", VERTEX_EMBED_CONTENT_URL, id="vertex-embedContent"),
    pytest.param("gemini/gemini-embedding-001", GEMINI_BATCH_EMBEDDINGS_URL, id="gemini-batchEmbedContents"),
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


@pytest.fixture
def error_test_vertex_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        google_batch_embeddings._credentials_project_mapping,
        (ERROR_TEST_CREDENTIALS, ERROR_TEST_PROJECT),
        (Credentials(token="test-vertex-token"), ERROR_TEST_PROJECT),
    )


def _vertex_params(model: str) -> dict[str, str]:
    if not model.startswith("vertex_ai/"):
        return {}
    return {
        "vertex_project": ERROR_TEST_PROJECT,
        "vertex_location": "us-central1",
        "vertex_credentials": ERROR_TEST_CREDENTIALS,
    }


def _upstream_error(url: str, status: int) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) != url:
            return httpx.Response(404, json={"error": {"message": f"unexpected url {request.url}"}})
        return httpx.Response(
            status,
            headers={"retry-after": "7"},
            stream=httpx.ByteStream(json.dumps({"error": {"code": status, "message": "upstream says no"}}).encode()),
        )

    return httpx.MockTransport(respond)


@pytest.mark.usefixtures("error_test_vertex_credentials")
@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, litellm.RateLimitError), (503, litellm.ServiceUnavailableError)],
)
@pytest.mark.parametrize(("model", "url"), SHARED_HANDLER_ROUTES)
def test_custom_httpx_client_error_response_keeps_upstream_status(
    model: str, url: str, status: int, error_type: type[litellm.RateLimitError | litellm.ServiceUnavailableError]
) -> None:
    with httpx.Client(transport=_upstream_error(url, status)) as client, pytest.raises(error_type) as exc_info:
        litellm.embedding(
            model=model,
            input=["good morning from litellm"],
            api_key="gemini-test-key",
            client=client,
            **_vertex_params(model),
        )

    assert exc_info.value.status_code == status
    assert exc_info.value.litellm_response_headers["retry-after"] == "7"
    assert "upstream says no" in exc_info.value.message


@pytest.mark.usefixtures("error_test_vertex_credentials")
@pytest.mark.parametrize(("model", "url"), SHARED_HANDLER_ROUTES)
def test_custom_httpx_client_raising_on_error_status_keeps_upstream_status(model: str, url: str) -> None:
    with (
        httpx.Client(
            transport=_upstream_error(url, 429),
            event_hooks={"response": [lambda response: response.raise_for_status()]},
        ) as raising_client,
        pytest.raises(litellm.RateLimitError) as exc_info,
    ):
        litellm.embedding(
            model=model,
            input=["good morning from litellm"],
            api_key="gemini-test-key",
            client=raising_client,
            **_vertex_params(model),
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.litellm_response_headers["retry-after"] == "7"
    assert httpx.codes.TOO_MANY_REQUESTS.phrase in exc_info.value.message


async def _embed_with_default_client(model: str, sync_mode: bool) -> EmbeddingResponse:
    if sync_mode:
        return litellm.embedding(
            model=model, input=["good morning from litellm"], api_key="gemini-test-key", **_vertex_params(model)
        )
    return await litellm.aembedding(
        model=model, input=["good morning from litellm"], api_key="gemini-test-key", **_vertex_params(model)
    )


@pytest.mark.usefixtures("error_test_vertex_credentials")
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(("model", "url"), SHARED_HANDLER_ROUTES)
@pytest.mark.asyncio
async def test_default_client_error_response_keeps_upstream_status(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool, model: str, url: str
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    respx_mock.post(url).mock(
        return_value=httpx.Response(
            429, headers={"retry-after": "7"}, json={"error": {"code": 429, "message": "upstream says no"}}
        )
    )

    with pytest.raises(litellm.RateLimitError) as exc_info:
        await _embed_with_default_client(model, sync_mode)

    assert exc_info.value.status_code == 429
    assert exc_info.value.litellm_response_headers["retry-after"] == "7"
    assert "upstream says no" in exc_info.value.message


def _direct_handler_url(use_embed_content: bool) -> str:
    return VERTEX_EMBED_CONTENT_URL if use_embed_content else GEMINI_BATCH_EMBEDDINGS_URL


async def _direct_async_embedding(client: httpx.AsyncClient, use_embed_content: bool) -> EmbeddingResponse:
    return await GoogleBatchEmbeddings().async_batch_embeddings(
        model="gemini-embedding-2" if use_embed_content else "gemini-embedding-001",
        api_base=None,
        url=_direct_handler_url(use_embed_content),
        data=None,
        model_response=EmbeddingResponse(),
        input=["good morning from litellm"],
        timeout=None,
        client=client,
        use_embed_content=use_embed_content,
        api_key=None if use_embed_content else "gemini-test-key",
    )


async def _raise_for_status(response: httpx.Response) -> None:
    response.raise_for_status()


@pytest.mark.parametrize("use_embed_content", [True, False], ids=["vertex-embedContent", "gemini-batchEmbedContents"])
@pytest.mark.asyncio
async def test_async_handler_custom_httpx_client_parses_vectors(use_embed_content: bool) -> None:
    body: Final = (
        {"embedding": {"values": [0.1, 0.2, 0.3]}}
        if use_embed_content
        else {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}
    )
    url: Final = _direct_handler_url(use_embed_content)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body) if str(request.url) == url else httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        response: Final = await _direct_async_embedding(client, use_embed_content)

    assert [item["embedding"] for item in response.data] == [[0.1, 0.2, 0.3]]


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("use_embed_content", [True, False], ids=["vertex-embedContent", "gemini-batchEmbedContents"])
@pytest.mark.asyncio
async def test_async_handler_custom_httpx_client_error_response_keeps_upstream_status(
    use_embed_content: bool, status: int
) -> None:
    async with httpx.AsyncClient(transport=_upstream_error(_direct_handler_url(use_embed_content), status)) as client:
        with pytest.raises(VertexAIError) as exc_info:
            await _direct_async_embedding(client, use_embed_content)

    assert exc_info.value.status_code == status
    assert exc_info.value.headers["retry-after"] == "7"
    assert "upstream says no" in exc_info.value.message


@pytest.mark.parametrize("use_embed_content", [True, False], ids=["vertex-embedContent", "gemini-batchEmbedContents"])
@pytest.mark.asyncio
async def test_async_handler_custom_httpx_client_raising_on_error_status_keeps_upstream_status(
    use_embed_content: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=_upstream_error(_direct_handler_url(use_embed_content), 429),
        event_hooks={"response": [_raise_for_status]},
    ) as client:
        with pytest.raises(VertexAIError) as exc_info:
            await _direct_async_embedding(client, use_embed_content)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers["retry-after"] == "7"
    assert exc_info.value.message == httpx.codes.TOO_MANY_REQUESTS.phrase


@pytest.mark.asyncio
async def test_async_handler_custom_httpx_client_error_redacts_credentials_echoed_by_upstream() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"bad request for {request.url}?key=not-a-real-secret-123")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(VertexAIError) as exc_info:
            await _direct_async_embedding(client, use_embed_content=False)

    assert exc_info.value.status_code == 400
    assert "bad request for" in exc_info.value.message
    assert "not-a-real-secret-123" not in exc_info.value.message
