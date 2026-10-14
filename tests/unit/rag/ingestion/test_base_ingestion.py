from collections import UserString
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.rag.ingestion.base_ingestion import BaseRAGIngestion
from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion
from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion
from litellm.rag.ingestion.vertex_ai_ingestion import VertexAIRAGIngestion
from litellm.types.utils import CredentialItem

_FILE_URL: Final = "https://files.example/docs/report.pdf"
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
