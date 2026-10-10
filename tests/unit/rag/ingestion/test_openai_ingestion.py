import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.rag.ingestion.openai_ingestion import OpenAIRAGIngestion

_API_BASE: Final = "https://openai.example/v1"
_STATIC_CHUNKING: Final = {"type": "static", "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}}
_ATTACHED_FILE: Final = {
    "id": "file-1",
    "object": "vector_store.file",
    "created_at": 1767323045,
    "vector_store_id": "vs_1",
    "status": "completed",
    "usage_bytes": 10,
}
_UPLOADED_FILE: Final = {
    "id": "file-1",
    "object": "file",
    "bytes": 10,
    "created_at": 1767323045,
    "filename": "notes.txt",
    "purpose": "assistants",
    "status": "processed",
}


def _ingestion(chunking_strategy: object) -> OpenAIRAGIngestion:
    return OpenAIRAGIngestion(
        ingest_options={
            "chunking_strategy": chunking_strategy,
            "vector_store": {
                "custom_llm_provider": "openai",
                "vector_store_id": "vs_1",
                "api_key": "sk-test",
                "api_base": _API_BASE,
            },
        }
    )


@pytest.fixture
def attach_file(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.Route:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post(f"{_API_BASE}/vector_stores/vs_1/files").mock(
        return_value=httpx.Response(200, json=_ATTACHED_FILE)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunking_strategy", "sent_chunking_strategy"),
    [(None, {"type": "auto"}), ({"type": "auto"}, {"type": "auto"}), (_STATIC_CHUNKING, _STATIC_CHUNKING)],
)
async def test_an_uploaded_file_is_attached_to_the_vector_store_with_the_chunking_strategy(
    attach_file: respx.Route, respx_mock: respx.MockRouter, chunking_strategy: object, sent_chunking_strategy: object
):
    respx_mock.post(f"{_API_BASE}/files").mock(return_value=httpx.Response(200, json=_UPLOADED_FILE))

    stored: Final = await _ingestion(chunking_strategy).store(
        file_content=b"first note",
        filename="notes.txt",
        content_type="text/plain",
        chunks=[],
        embeddings=None,
    )

    assert stored == ("vs_1", "file-1")
    assert json.loads(attach_file.calls.last.request.content) == {
        "file_id": "file-1",
        "chunking_strategy": sent_chunking_strategy,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunking_strategy", "sent_chunking_strategy"),
    [(None, {"type": "auto"}), (_STATIC_CHUNKING, _STATIC_CHUNKING)],
)
async def test_an_existing_file_is_attached_to_the_vector_store_with_the_chunking_strategy(
    attach_file: respx.Route, chunking_strategy: object, sent_chunking_strategy: object
):
    stored: Final = await _ingestion(chunking_strategy).store(
        file_content=None,
        filename=None,
        content_type=None,
        chunks=[],
        embeddings=None,
        existing_file_id="file-9",
    )

    assert stored == ("vs_1", "file-9")
    assert json.loads(attach_file.calls.last.request.content) == {
        "file_id": "file-9",
        "chunking_strategy": sent_chunking_strategy,
    }
