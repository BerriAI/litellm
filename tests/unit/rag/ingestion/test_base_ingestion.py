from collections import UserString
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion
from litellm.types.utils import CredentialItem

_FILE_URL: Final = "https://files.example/docs/report.pdf"
_STORED_CREDENTIAL: Final = CredentialItem(
    credential_name="gemini-prod",
    credential_info={},
    credential_values={"api_key": "stored-key"},
)


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
