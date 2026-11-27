import datetime
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.vector_stores import create as vector_store_create
from litellm.vector_stores import search as vector_store_search

GENERATE_CONTENT_URL: Final = r"https://generativelanguage\.googleapis\.com/v1beta/models/[^/]+:generateContent"
FILE_SEARCH_STORES_URL: Final = "https://generativelanguage.googleapis.com/v1beta/fileSearchStores"


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def test_search_returns_file_chunks_and_new_grounding_support_texts(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(url__regex=GENERATE_CONTENT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "groundingMetadata": {
                            "groundingChunks": [
                                {
                                    "retrievedContext": {
                                        "text": "Refunds take five days.",
                                        "uri": "fileSearchStores/store-1/documents/doc-1",
                                        "title": "refunds.md",
                                    }
                                },
                                {"web": {"uri": "https://example.com"}},
                            ],
                            "groundingSupports": [
                                {
                                    "segment": {"text": "Refunds take five days."},
                                    "groundingChunkIndices": [0],
                                    "confidenceScores": [0.9],
                                },
                                {
                                    "segment": {"text": "Contact support for faster refunds."},
                                    "groundingChunkIndices": [0, 1],
                                    "confidenceScores": [0.42, 0.1],
                                },
                            ],
                        }
                    }
                ]
            },
        )
    )

    response = vector_store_search(
        vector_store_id="fileSearchStores/store-1",
        query="how long do refunds take",
        custom_llm_provider="gemini",
        api_key="test-key",
    )

    assert response["search_query"] == "how long do refunds take"
    assert [
        (result["content"][0]["text"], result["score"], result.get("file_id"), result.get("filename"))
        for result in response["data"]
    ] == [
        (
            "Refunds take five days.",
            None,
            "fileSearchStores/store-1/documents/doc-1",
            "refunds.md",
        ),
        ("Contact support for faster refunds.", 0.42, None, None),
    ]
    assert response["data"][0]["attributes"] == {
        "uri": "fileSearchStores/store-1/documents/doc-1",
        "title": "refunds.md",
    }
    assert response["data"][1]["attributes"] == {"grounding_chunk_indices": [0, 1]}


def test_search_reports_a_non_object_body_as_a_parse_failure(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(url__regex=GENERATE_CONTENT_URL).mock(
        return_value=httpx.Response(200, json=["not", "an", "object"])
    )

    with pytest.raises(litellm.APIConnectionError, match="Failed to parse Gemini response"):
        vector_store_search(
            vector_store_id="fileSearchStores/store-1",
            query="refunds",
            custom_llm_provider="gemini",
            api_key="test-key",
        )


def test_create_maps_the_file_search_store_to_a_vector_store(respx_mock: respx.MockRouter) -> None:
    created_at: Final = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    respx_mock.post(FILE_SEARCH_STORES_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "fileSearchStores/store-1",
                "displayName": "support docs",
                "createTime": "2026-01-02T03:04:05Z",
            },
        )
    )

    response = vector_store_create(name="support docs", custom_llm_provider="gemini", api_key="test-key")

    assert (response["id"], response["name"], response["created_at"]) == (
        "fileSearchStores/store-1",
        "support docs",
        int(created_at.timestamp()),
    )
