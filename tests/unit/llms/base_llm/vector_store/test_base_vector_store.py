"""Cross-provider base vector store contract tests.

Exercises litellm.vector_stores create/search against a real request/response
cycle with the upstream scripted at the provider's own URL.
"""

import json
from typing import Final

import httpx
import pytest
import respx

import litellm

OPENAI_VECTOR_STORES_URL: Final = "https://api.openai.com/v1/vector_stores"
OPENAI_VECTOR_STORE_SEARCH_URL: Final = "https://api.openai.com/v1/vector_stores/vs_abc123/search"


@pytest.mark.asyncio
async def test_basic_create_vector_store(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    create_route = respx_mock.post(OPENAI_VECTOR_STORES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "vs_abc123",
                "object": "vector_store",
                "created_at": 1745836841,
                "name": "support-docs",
                "usage_bytes": 0,
                "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                "status": "completed",
                "last_active_at": 1745836841,
                "metadata": {},
            },
        )
    )

    response = await litellm.vector_stores.acreate(
        name="support-docs",
        custom_llm_provider="openai",
        api_key="sk-unit-test",
    )

    create_request = create_route.calls.last.request
    assert create_request.headers["authorization"] == "Bearer sk-unit-test"
    create_body = json.loads(create_request.read())
    assert create_body["name"] == "support-docs"

    assert isinstance(response, dict)
    assert response["id"] == "vs_abc123"
    assert response["object"] == "vector_store"
    assert response["name"] == "support-docs"
    assert response["created_at"] == 1745836841


@pytest.mark.asyncio
async def test_basic_search_vector_store(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    search_route = respx_mock.post(OPENAI_VECTOR_STORE_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "vector_store.search_results.page",
                "search_query": "what is machine learning?",
                "data": [
                    {
                        "file_id": "file-1",
                        "filename": "ml-notes.txt",
                        "score": 0.87,
                        "attributes": {},
                        "content": [{"type": "text", "text": "machine learning answer chunk"}],
                    }
                ],
            },
        )
    )

    response = await litellm.vector_stores.asearch(
        query="what is machine learning?",
        vector_store_id="vs_abc123",
        custom_llm_provider="openai",
        api_key="sk-unit-test",
    )

    search_request = search_route.calls.last.request
    assert search_request.headers["authorization"] == "Bearer sk-unit-test"
    assert json.loads(search_request.read())["query"] == "what is machine learning?"

    assert response["object"] == "vector_store.search_results.page"
    assert len(response["data"]) == 1
    first_result = response["data"][0]
    assert first_result["file_id"] == "file-1"
    assert first_result["filename"] == "ml-notes.txt"
    assert first_result["score"] == 0.87
    assert first_result["content"][0]["type"] == "text"
    assert first_result["content"][0]["text"] == "machine learning answer chunk"
