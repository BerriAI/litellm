"""
Base RAG test class that enforces common tests across all providers.

Providers should inherit from BaseRAGTest and implement the abstract methods.
"""

import os
import sys
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import (
    RAGIngestOptions,
    OpenAIVectorStoreOptions,
    BedrockVectorStoreOptions,
)


class BaseRAGTest(ABC):
    """
    Abstract base test class for RAG ingestion tests.

    Providers should inherit from this class and implement:
    - get_base_ingest_options(): Returns provider-specific ingest options
    - query_vector_store(): Queries the vector store after ingestion
    """

    @abstractmethod
    def get_base_ingest_options(self) -> RAGIngestOptions:
        """
        Must return the base ingest options for the provider.

        Example for OpenAI:
            return {
                "vector_store": OpenAIVectorStoreOptions(
                    custom_llm_provider="openai",
                )
            }

        Example for Bedrock:
            return {
                "vector_store": BedrockVectorStoreOptions(
                    custom_llm_provider="bedrock",
                )
            }
        """
        pass

    @abstractmethod
    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query the vector store to verify ingestion.

        Args:
            vector_store_id: The ID of the vector store to query
            query: The search query

        Returns:
            Search results dict or None if no results found
        """
        pass

    def get_unique_filename(self, prefix: str = "test") -> str:
        """Generate a unique filename for test documents."""
        unique_id = uuid.uuid4().hex[:8]
        return f"{prefix}_{unique_id}.txt", unique_id

    @pytest.mark.asyncio
    async def test_basic_ingest(self):
        """
        Test basic text file ingestion to vector store.
        """
        litellm._turn_on_debug()

        filename, unique_id = self.get_unique_filename("basic_ingest")
        text_content = f"Test document {unique_id} for RAG ingestion.".encode("utf-8")
        file_data = (filename, text_content, "text/plain")

        ingest_options = self.get_base_ingest_options()
        ingest_options["name"] = f"test-basic-ingest-{unique_id}"

        try:
            response = await litellm.rag.aingest(
                ingest_options=ingest_options,
                file_data=file_data,
            )

            print(f"RAG Ingest Response: {response}")

            assert "id" in response
            assert response["id"].startswith("ingest_")
            assert "status" in response
            assert response["status"] in ["completed", "failed"]
            assert "vector_store_id" in response

            if response["status"] == "completed":
                assert response["vector_store_id"]
                print(f"Vector store ID: {response['vector_store_id']}")

        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")

    @pytest.mark.asyncio
    async def test_ingest_and_query(self):
        """
        Test full RAG flow: ingest a document and then query it.
        """
        import asyncio

        litellm._turn_on_debug()

        filename, unique_id = self.get_unique_filename("ingest_query")
        text_content = f"""
        Test document {unique_id} for RAG ingestion and query.
        LiteLLM provides a unified interface for 100+ LLMs.
        This content should be retrievable via semantic search.
        """.encode("utf-8")
        file_data = (filename, text_content, "text/plain")

        ingest_options = self.get_base_ingest_options()
        ingest_options["name"] = f"test-ingest-query-{unique_id}"

        try:
            # Step 1: Ingest
            ingest_response = await litellm.rag.aingest(
                ingest_options=ingest_options,
                file_data=file_data,
            )

            print(f"Ingest Response: {ingest_response}")
            assert ingest_response["status"] == "completed"
            vector_store_id = ingest_response["vector_store_id"]
            assert vector_store_id

            # Step 2: Query with retry (indexing may take time)
            search_results = None
            max_retries = 10
            for attempt in range(max_retries):
                await asyncio.sleep(3)

                search_results = await self.query_vector_store(
                    vector_store_id=vector_store_id,
                    query=f"Test document {unique_id}",
                )

                if search_results:
                    break

                print(
                    f"Attempt {attempt + 1}/{max_retries}: "
                    "Waiting for document to be indexed..."
                )

            print(f"Search Results: {search_results}")

            # Validate search results
            assert search_results is not None, "Document not found after retries"

            print("Query successful!")

        except litellm.InternalServerError:
            pytest.skip("Skipping test due to litellm.InternalServerError")

