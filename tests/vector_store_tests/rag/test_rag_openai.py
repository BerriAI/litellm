"""
OpenAI RAG ingestion tests.
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import RAGIngestOptions, OpenAIVectorStoreOptions
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest


class TestRAGOpenAI(BaseRAGTest):
    """Test RAG Ingest with OpenAI provider."""

    def get_base_ingest_options(self) -> RAGIngestOptions:
        """Return OpenAI-specific ingest options."""
        return {
            "vector_store": OpenAIVectorStoreOptions(
                custom_llm_provider="openai",
            ),
        }

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Query OpenAI vector store."""
        search_response = await litellm.vector_stores.asearch(
            vector_store_id=vector_store_id,
            query=query,
            custom_llm_provider="openai",
        )

        if search_response.get("data") and len(search_response["data"]) > 0:
            return search_response
        return None

    @pytest.mark.asyncio
    async def test_rag_query_basic(self):
        """Test basic RAG query flow."""
        import asyncio

        litellm._turn_on_debug()

        # First ingest a document
        filename, unique_id = self.get_unique_filename("rag_query")
        text_content = (
            f"LiteLLM is a unified interface for 100+ LLMs. ID: {unique_id}".encode()
        )

        ingest_response = await litellm.rag.aingest(
            ingest_options=self.get_base_ingest_options(),
            file_data=(filename, text_content, "text/plain"),
        )
        
        # Check if ingestion succeeded
        if ingest_response["status"] != "completed":
            pytest.fail(
                f"Ingestion failed with status: {ingest_response['status']}, "
                f"error: {ingest_response.get('error', 'Unknown')}"
            )
        
        vector_store_id = ingest_response["vector_store_id"]
        assert vector_store_id, "vector_store_id should not be empty"

        # Wait for indexing
        await asyncio.sleep(10)

        # Query with RAG
        response = await litellm.rag.aquery(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is LiteLLM?"}],
            retrieval_config={
                "vector_store_id": vector_store_id,
                "custom_llm_provider": "openai",
                "top_k": 5,
            },
        )

        print(f"RAG Query Response: {response}")

        assert response.choices[0].message.content
        assert (
            "search_results" in response.choices[0].message.provider_specific_fields
        )

    @pytest.mark.asyncio
    async def test_rag_query_with_rerank(self):
        """Test RAG query with reranking."""
        import asyncio

        litellm._turn_on_debug()

        # First ingest a document
        filename, unique_id = self.get_unique_filename("rag_query_rerank")
        text_content = (
            f"LiteLLM is a unified interface for 100+ LLMs. ID: {unique_id}".encode()
        )

        ingest_response = await litellm.rag.aingest(
            ingest_options=self.get_base_ingest_options(),
            file_data=(filename, text_content, "text/plain"),
        )
        
        # Check if ingestion succeeded
        if ingest_response["status"] != "completed":
            pytest.fail(
                f"Ingestion failed with status: {ingest_response['status']}, "
                f"error: {ingest_response.get('error', 'Unknown')}"
            )
        
        vector_store_id = ingest_response["vector_store_id"]
        assert vector_store_id, "vector_store_id should not be empty"

        # Wait for indexing
        await asyncio.sleep(10)

        # Query with RAG and rerank
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

        print(f"RAG Query Response with Rerank: {response.model_dump_json(indent=4)}")

        assert response.choices[0].message.content
        assert (
            "search_results" in response.choices[0].message.provider_specific_fields
        )
        assert (
            "rerank_results" in response.choices[0].message.provider_specific_fields
        )

    