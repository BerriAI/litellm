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

    