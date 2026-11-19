from collections import UserString
from types import SimpleNamespace
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion
from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion
from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion
from litellm.rag.ingestion.vertex_ai_ingestion import VertexAIRAGIngestion
from litellm.types.utils import CredentialItem

_FILE_URL: Final = "https://files.example/docs/report.pdf"
_GEMINI_API_BASE: Final = "https://gemini.example"
_GEMINI_STORE: Final = "fileSearchStores/docs-1"
_GEMINI_UPLOAD_SESSION_URL: Final = "https://gemini.example/upload/session-1"
_TEXT_FILE: Final = ("report.txt", b"plain text", "text/plain")
_STORED_CREDENTIAL: Final = CredentialItem(
    credential_name="gemini-prod",
    credential_info={},
    credential_values={"api_key": "stored-key"},
)


class _UnauthorizedIngestionError(Exception):
    status_code: Final = 401


class _FailingStoreIngestion(BaseRAGIngestion):
    def __init__(self, failure: Exception) -> None:
        super().__init__(ingest_options={"vector_store": {}})
        self._failure = failure

    async def store(
        self,
        file_content: bytes | None,
        filename: str | None,
        content_type: str | None,
        chunks: list[str],
        embeddings: list[list[float]] | None,
        existing_file_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        raise self._failure


class _EmbeddingRouter:
    async def aembedding(self, model: str, input: list[str]) -> SimpleNamespace:
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]} for _ in input])


def _vector_store_options(litellm_credential_name: object) -> dict[str, object]:
    return {
        "custom_llm_provider": "gemini",
        "litellm_credential_name": litellm_credential_name,
        "api_key": "caller-key",
        "api_base": "https://caller.example",
    }


def test_a_stored_credential_named_by_the_vector_store_replaces_the_caller_supplied_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "credential_list", [_STORED_CREDENTIAL])
    vector_store: Final = _vector_store_options("gemini-prod")

    ingestion: Final = GeminiRAGIngestion(ingest_options={"vector_store": vector_store})

    assert ingestion.vector_store_config is vector_store
    assert vector_store == {
        "custom_llm_provider": "gemini",
        "litellm_credential_name": "gemini-prod",
        "api_key": "stored-key",
    }


@pytest.mark.parametrize(
    "litellm_credential_name",
    ["gemini-staging", "", None, 7, True, ["gemini-prod"], {"credential_name": "gemini-prod"}],
)
def test_a_credential_name_that_matches_no_stored_credential_leaves_the_vector_store_config_alone(
    monkeypatch: pytest.MonkeyPatch, litellm_credential_name: object
):
    monkeypatch.setattr(litellm, "credential_list", [_STORED_CREDENTIAL])

    ingestion: Final = GeminiRAGIngestion(
        ingest_options={"vector_store": _vector_store_options(litellm_credential_name)}
    )

    assert ingestion.vector_store_config == _vector_store_options(litellm_credential_name)


def test_a_credential_name_that_is_not_a_string_is_not_resolved_even_when_it_equals_a_stored_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "credential_list", [_STORED_CREDENTIAL])
    litellm_credential_name: Final = UserString("gemini-prod")

    ingestion: Final = GeminiRAGIngestion(
        ingest_options={"vector_store": _vector_store_options(litellm_credential_name)}
    )

    assert litellm_credential_name == "gemini-prod"
    assert ingestion.vector_store_config == _vector_store_options(litellm_credential_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "expected_content_type"),
    [
        ([("content-type", "application/pdf")], "application/pdf"),
        ([("Content-Type", "text/plain; charset=utf-8")], "text/plain; charset=utf-8"),
        ([("content-type", "")], ""),
        ([("content-type", "text/plain"), ("content-type", "text/html")], "text/plain, text/html"),
        ([], "application/octet-stream"),
        ([("content-length", "8")], "application/octet-stream"),
    ],
)
async def test_upload_from_a_url_takes_the_content_type_from_the_response_header(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    headers: list[tuple[str, str]],
    expected_content_type: str,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "user_url_validation", False)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.get(_FILE_URL).mock(return_value=httpx.Response(200, content=b"%PDF-1.7", headers=headers))
    ingestion: Final = GeminiRAGIngestion(ingest_options={"vector_store": {"custom_llm_provider": "gemini"}})

    uploaded: Final = await ingestion.upload(file_url=_FILE_URL)

    assert uploaded == ("report.pdf", b"%PDF-1.7", expected_content_type, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status_code"),
    [
        (_UnauthorizedIngestionError("unauthorized"), 401),
        (RuntimeError("ingestion failed"), None),
    ],
)
async def test_ingest_response_preserves_integer_exception_status_code(
    failure: Exception,
    expected_status_code: int | None,
) -> None:
    response: Final = await _FailingStoreIngestion(failure).ingest(
        file_data=("report.txt", b"plain text", "text/plain")
    )

    assert response["error_status_code"] == expected_status_code


async def test_an_existing_file_id_on_a_store_that_cannot_take_one_fails_as_a_bad_request() -> None:
    response: Final = await _FailingStoreIngestion(RuntimeError("store must not run")).ingest(file_id="file-abc")

    assert response["status"] == "failed"
    assert response["error_status_code"] == 400
    assert "does not support ingesting an existing file_id" in str(response["error"])


@pytest.mark.parametrize(
    ("ingestion_class", "ingest_kwargs"),
    [
        (OpenAIRAGIngestion, {"file_id": "file-abc"}),
        (VertexAIRAGIngestion, {"file_data": ("report.txt", b"plain text", "text/plain")}),
    ],
)
async def test_a_vector_store_config_missing_a_caller_field_fails_as_a_bad_request(
    ingestion_class: type[BaseRAGIngestion],
    ingest_kwargs: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERTEXAI_PROJECT", raising=False)
    ingestion: Final = ingestion_class(ingest_options={"vector_store": {"api_key": "caller-key"}})

    response: Final = await ingestion.ingest(**ingest_kwargs)

    assert response["status"] == "failed"
    assert response["error_status_code"] == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vector_store_id", "upstream_statuses", "expected_status_code"),
    [
        (None, {"create_store": 401}, 401),
        (_GEMINI_STORE, {"start_upload": 429}, 429),
        (_GEMINI_STORE, {"upload": 503}, 503),
    ],
)
async def test_a_gemini_upstream_error_keeps_its_status_on_the_failed_ingest(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    vector_store_id: str | None,
    upstream_statuses: dict[str, int],
    expected_status_code: int,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(f"{_GEMINI_API_BASE}/v1beta/fileSearchStores").mock(
        return_value=httpx.Response(upstream_statuses.get("create_store", 200), json={"name": _GEMINI_STORE})
    )
    respx_mock.post(f"{_GEMINI_API_BASE}/upload/v1beta/{_GEMINI_STORE}:uploadToFileSearchStore").mock(
        return_value=httpx.Response(
            upstream_statuses.get("start_upload", 200), headers={"x-goog-upload-url": _GEMINI_UPLOAD_SESSION_URL}
        )
    )
    respx_mock.put(_GEMINI_UPLOAD_SESSION_URL).mock(
        return_value=httpx.Response(upstream_statuses.get("upload", 200), json={"name": f"{_GEMINI_STORE}/documents/d"})
    )
    vector_store: Final = {
        "custom_llm_provider": "gemini",
        "api_key": "test-key",
        "api_base": _GEMINI_API_BASE,
        **({"vector_store_id": vector_store_id} if vector_store_id else {}),
    }

    response: Final = await GeminiRAGIngestion(ingest_options={"vector_store": vector_store}).ingest(
        file_data=_TEXT_FILE
    )

    assert response["status"] == "failed"
    assert response["error_status_code"] == expected_status_code


@pytest.mark.asyncio
async def test_a_gemini_store_with_no_api_key_anywhere_fails_without_a_caller_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ingestion: Final = GeminiRAGIngestion(ingest_options={"vector_store": {"custom_llm_provider": "gemini"}})

    response: Final = await ingestion.ingest(file_data=_TEXT_FILE)

    assert response["status"] == "failed"
    assert response.get("error_status_code") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_bucket_name",
    ["ab", "bucket_with_underscores"],
)
async def test_an_invalid_s3_vectors_bucket_name_fails_as_a_bad_request(vector_bucket_name: str) -> None:
    ingestion: Final = S3VectorsRAGIngestion(
        ingest_options={
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": {
                "custom_llm_provider": "s3_vectors",
                "aws_region_name": "us-west-2",
                "vector_bucket_name": vector_bucket_name,
                "dimension": 2,
            },
        },
        router=_EmbeddingRouter(),
    )

    response: Final = await ingestion.ingest(file_data=_TEXT_FILE)

    assert response["status"] == "failed"
    assert response["error_status_code"] == 400
