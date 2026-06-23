"""
Milvus RAG ingestion tests.

Requires environment variables:
- MILVUS_API_BASE (e.g. http://localhost:19530)

Optional:
- MILVUS_API_KEY (token, e.g. "root:Milvus"); omit for a Milvus without auth
- MILVUS_COLLECTION_NAME (default: litellm_rag_test)

These tests are skipped unless MILVUS_API_BASE is set.
"""

import os
import sys
from typing import Any, Dict, Optional

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import RAGIngestOptions
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest

COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_NAME", "litellm_rag_test")


class TestRAGMilvus(BaseRAGTest):
    """Test RAG Ingest with self-hosted Milvus."""

    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        if not os.environ.get("MILVUS_API_BASE"):
            pytest.skip("Skipping Milvus test: MILVUS_API_BASE required")

    def get_base_ingest_options(self) -> RAGIngestOptions:
        return {
            "chunking_strategy": {"chunk_size": 512, "chunk_overlap": 100},
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": {
                "custom_llm_provider": "milvus",
                "collection_name": COLLECTION_NAME,
                "api_base": os.environ["MILVUS_API_BASE"],
                "api_key": os.environ.get("MILVUS_API_KEY"),
                "vector_field": "vector",
                "text_field": "text",
                "metric_type": "COSINE",
            },
        }

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Vector-search the Milvus collection via the REST API to verify ingestion."""
        embedding_response = await litellm.aembedding(
            model="text-embedding-3-small", input=[query]
        )
        query_vector = embedding_response.data[0]["embedding"]

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("MILVUS_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{os.environ['MILVUS_API_BASE'].rstrip('/')}/v2/vectordb/entities/search"
        body = {
            "collectionName": vector_store_id,
            "data": [query_vector],
            "annsField": "vector",
            "limit": 5,
            "outputFields": ["text", "filename", "chunk_index"],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=body, headers=headers)
            response.raise_for_status()
            data = response.json()

        if data.get("code") not in (0, None):
            return None
        results = data.get("data") or []
        return data if results else None
