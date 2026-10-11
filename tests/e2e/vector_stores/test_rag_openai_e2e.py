"""
OpenAI RAG ingestion and query end to end, with real OpenAI credentials.
"""

import asyncio
import uuid

import pytest

import litellm
from litellm.types.rag import OpenAIVectorStoreOptions, RAGIngestOptions

pytestmark = pytest.mark.e2e


def _unique_filename(prefix: str) -> tuple[str, str]:
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique_id}.txt", unique_id


def _base_ingest_options() -> RAGIngestOptions:
    return {
        "vector_store": OpenAIVectorStoreOptions(
            custom_llm_provider="openai",
        ),
    }


async def _ingest_document(prefix: str) -> tuple[str, str]:
    filename, unique_id = _unique_filename(prefix)
    text_content = f"LiteLLM is a unified interface for 100+ LLMs. ID: {unique_id}".encode()
    ingest_response = await litellm.rag.aingest(
        ingest_options=_base_ingest_options(),
        file_data=(filename, text_content, "text/plain"),
    )
    assert ingest_response["status"] == "completed", (
        f"Ingestion failed with status: {ingest_response['status']}, "
        f"error: {ingest_response.get('error', 'Unknown')}"
    )
    vector_store_id = ingest_response["vector_store_id"]
    assert vector_store_id
    return vector_store_id, unique_id


@pytest.mark.asyncio
async def test_rag_query_basic():
    """Ingest a document, then a RAG query answers from it and cites search results."""
    vector_store_id, unique_id = await _ingest_document("rag_query")

    await asyncio.sleep(10)

    response = await litellm.rag.aquery(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is LiteLLM?"}],
        retrieval_config={
            "vector_store_id": vector_store_id,
            "custom_llm_provider": "openai",
            "top_k": 5,
        },
    )

    content = response.choices[0].message.content
    assert content
    assert isinstance(content, str)
    assert "search_results" in response.choices[0].message.provider_specific_fields
    search_results = response.choices[0].message.provider_specific_fields["search_results"]
    assert len(search_results) > 0


@pytest.mark.asyncio
async def test_rag_query_with_rerank():
    """Ingest a document, then a reranked RAG query reports rerank results."""
    vector_store_id, unique_id = await _ingest_document("rag_query_rerank")

    await asyncio.sleep(10)

    response = await litellm.rag.aquery(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is LiteLLM?"}],
        retrieval_config={
            "vector_store_id": vector_store_id,
            "custom_llm_provider": "openai",
            "top_k": 5,
        },
        rerank={
            "enabled": True,
            "model": "cohere/rerank-english-v3.0",
            "top_n": 3,
        },
    )

    content = response.choices[0].message.content
    assert content
    assert isinstance(content, str)
    provider_fields = response.choices[0].message.provider_specific_fields
    assert "search_results" in provider_fields
    assert "rerank_results" in provider_fields
    assert len(provider_fields["rerank_results"]) <= 3
