import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.rag.ingestion.gemini_ingestion import GeminiRAGIngestion

_API_BASE: Final = "https://gemini.example"
_STORE: Final = "fileSearchStores/docs-1"
_START_UPLOAD_URL: Final = f"{_API_BASE}/upload/v1beta/{_STORE}:uploadToFileSearchStore"
_UPLOAD_SESSION_URL: Final = "https://gemini.example/upload/session-1"
_DOCUMENT: Final = "fileSearchStores/docs-1/documents/notes-1"


def _ingestion(chunking_strategy: object) -> GeminiRAGIngestion:
    return GeminiRAGIngestion(
        ingest_options={
            "chunking_strategy": chunking_strategy,
            "vector_store": {
                "custom_llm_provider": "gemini",
                "vector_store_id": _STORE,
                "api_key": "test-key",
                "api_base": _API_BASE,
            },
        }
    )


async def _store_notes(ingestion: GeminiRAGIngestion) -> tuple[str | None, str | None]:
    return await ingestion.store(
        file_content=b"first note",
        filename="notes.txt",
        content_type="text/plain",
        chunks=[],
        embeddings=None,
    )


@pytest.fixture
def start_upload(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.put(_UPLOAD_SESSION_URL).mock(return_value=httpx.Response(200, json={"name": _DOCUMENT}))
    return respx_mock.post(_START_UPLOAD_URL).mock(
        return_value=httpx.Response(200, headers={"x-goog-upload-url": _UPLOAD_SESSION_URL})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("white_space_config", "expected"),
    [
        (
            {"max_tokens_per_chunk": 200, "max_overlap_tokens": 20},
            {"maxTokensPerChunk": 200, "maxOverlapTokens": 20},
        ),
        ({"max_tokens_per_chunk": 200}, {"maxTokensPerChunk": 200, "maxOverlapTokens": 400}),
        ({"unrelated": True}, {"maxTokensPerChunk": 800, "maxOverlapTokens": 400}),
        (
            {"max_tokens_per_chunk": "200", "max_overlap_tokens": None},
            {"maxTokensPerChunk": "200", "maxOverlapTokens": None},
        ),
        ({7: "not a field", "max_overlap_tokens": 1.5}, {"maxTokensPerChunk": 800, "maxOverlapTokens": 1.5}),
        (MappingProxyType({"max_overlap_tokens": 0}), {"maxTokensPerChunk": 800, "maxOverlapTokens": 0}),
    ],
)
async def test_white_space_config_is_sent_as_the_chunking_config_of_the_upload(
    start_upload: respx.Route, white_space_config: Mapping[object, object], expected: Mapping[str, object]
):
    stored: Final = await _store_notes(_ingestion({"white_space_config": white_space_config}))

    assert stored == (_STORE, _DOCUMENT)
    assert json.loads(start_upload.calls.last.request.content) == {
        "displayName": "notes.txt",
        "chunkingConfig": {"whiteSpaceConfig": expected},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunking_strategy",
    [None, {"type": "auto"}, {"white_space_config": None}, {"white_space_config": {}}, {"white_space_config": 0}],
)
async def test_upload_without_a_white_space_config_sends_no_chunking_config(
    start_upload: respx.Route, chunking_strategy: object
):
    stored: Final = await _store_notes(_ingestion(chunking_strategy))

    assert stored == (_STORE, _DOCUMENT)
    assert json.loads(start_upload.calls.last.request.content) == {"displayName": "notes.txt"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "white_space_config",
    ["private-setting", [800, 400], [("max_tokens_per_chunk", 200)], 800, True, 1.5],
)
async def test_a_white_space_config_that_is_not_a_mapping_is_rejected_before_any_upload(
    start_upload: respx.Route, white_space_config: object
):
    with pytest.raises(ValidationError) as raised:
        await _store_notes(_ingestion({"white_space_config": white_space_config}))

    assert "private-setting" not in str(raised.value)
    assert not start_upload.called
