"""
Vertex AI RAG Engine ingestion tests.

Requires gcloud auth application-default login (ADC) and VERTEX_PROJECT set.
"""

import asyncio
import os
import uuid

import pytest

import litellm
from litellm.types.rag import RAGIngestOptions

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def check_env_vars():
    if not os.environ.get("VERTEX_PROJECT"):
        pytest.skip("Skipping Vertex AI test: VERTEX_PROJECT required")


def _unique_filename(prefix: str) -> tuple[str, str]:
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique_id}.txt", unique_id


def _base_ingest_options() -> RAGIngestOptions:
    vertex_project = os.environ.get("VERTEX_PROJECT")
    vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")
    corpus_id = os.environ.get("VERTEX_CORPUS_ID")

    options: RAGIngestOptions = {
        "chunking_strategy": {
            "chunk_size": 512,
            "chunk_overlap": 100,
        },
        "vector_store": {
            "custom_llm_provider": "vertex_ai",
            "vertex_project": vertex_project,
            "vertex_location": vertex_location,
        },
    }

    if corpus_id:
        options["vector_store"]["vector_store_id"] = corpus_id

    return options


async def _query_vector_store(vector_store_id: str, query: str):
    vertex_project = os.environ.get("VERTEX_PROJECT")
    vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")

    search_response = await litellm.vector_stores.asearch(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=5,
        custom_llm_provider="vertex_ai",
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )

    if search_response and search_response.get("data"):
        results = []
        for item in search_response["data"]:
            text = ""
            if item.get("content"):
                for content_item in item["content"]:
                    if content_item.get("text"):
                        text += content_item["text"]
            results.append(
                {
                    "text": text,
                    "score": item.get("score", 0.0),
                    "file_id": item.get("file_id", ""),
                    "filename": item.get("filename", ""),
                }
            )
        if results:
            return results
    return None


@pytest.mark.asyncio
async def test_basic_ingest():
    """Test basic text file ingestion to vector store."""
    filename, unique_id = _unique_filename("basic_ingest")
    text_content = f"Test document {unique_id} for RAG ingestion.".encode("utf-8")
    file_data = (filename, text_content, "text/plain")

    ingest_options = _base_ingest_options()
    ingest_options["name"] = f"test-basic-ingest-{unique_id}"

    response = await litellm.rag.aingest(
        ingest_options=ingest_options,
        file_data=file_data,
    )

    assert "id" in response
    assert response["id"].startswith("ingest_")
    assert response["status"] in ["completed", "failed"]
    assert "vector_store_id" in response

    if response["status"] == "completed":
        assert response["vector_store_id"]


@pytest.mark.asyncio
async def test_ingest_and_query():
    """Test full RAG flow: ingest a document and then query it."""
    filename, unique_id = _unique_filename("ingest_query")
    text_content = f"""
    Test document {unique_id} for RAG ingestion and query.
    LiteLLM provides a unified interface for 100+ LLMs.
    This content should be retrievable via semantic search.
    """.encode(
        "utf-8"
    )
    file_data = (filename, text_content, "text/plain")

    ingest_options = _base_ingest_options()
    ingest_options["name"] = f"test-ingest-query-{unique_id}"

    ingest_response = await litellm.rag.aingest(
        ingest_options=ingest_options,
        file_data=file_data,
    )

    assert ingest_response["status"] == "completed"
    vector_store_id = ingest_response["vector_store_id"]
    assert vector_store_id

    search_results = None
    max_retries = 10
    for attempt in range(max_retries):
        await asyncio.sleep(3)

        search_results = await _query_vector_store(
            vector_store_id=vector_store_id,
            query=f"Test document {unique_id}",
        )

        if search_results:
            break

    assert search_results is not None, "Document not found after retries"
