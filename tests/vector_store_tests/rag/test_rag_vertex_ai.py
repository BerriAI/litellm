"""
Vertex AI RAG Engine ingestion tests.

Tests the Vertex AI RAG ingestion implementation that:
- Creates RAG corpora automatically (or uses existing ones)
- Uploads files directly to Vertex AI RAG Engine
- Handles long-running operations for corpus creation
- Supports both file upload and GCS import

Requires:
- gcloud auth application-default login (for ADC authentication)

Environment variables:
- VERTEX_PROJECT: GCP project ID (required)
- VERTEX_LOCATION: GCP region (optional, defaults to us-central1)
- VERTEX_CORPUS_ID: Existing RAG corpus ID (optional - will create if not provided)
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import RAGIngestOptions
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest


class TestRAGVertexAI(BaseRAGTest):
    """Test RAG Ingest with Vertex AI RAG Engine."""

    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        """Check required environment variables before each test."""
        vertex_project = os.environ.get("VERTEX_PROJECT")

        if not vertex_project:
            pytest.skip("Skipping Vertex AI test: VERTEX_PROJECT required")

    def get_base_ingest_options(self) -> RAGIngestOptions:
        """
        Return Vertex AI-specific ingest options.

        Chunking is configured via chunking_strategy (unified interface),
        not inside vector_store.
        
        If VERTEX_CORPUS_ID is not set, a new corpus will be created automatically.
        """
        vertex_project = os.environ.get("VERTEX_PROJECT")
        vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")
        corpus_id = os.environ.get("VERTEX_CORPUS_ID")  # Optional

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
        
        # Add corpus ID if provided (otherwise will create new corpus)
        if corpus_id:
            options["vector_store"]["vector_store_id"] = corpus_id

        return options

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Query Vertex AI RAG corpus using LiteLLM's vector store search.
        
        Args:
            vector_store_id: The RAG corpus ID (can be full path or just the ID)
            query: The search query
            
        Returns:
            Search results dict or None if no results found
        """
        vertex_project = os.environ.get("VERTEX_PROJECT")
        vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")

        try:
            # Use LiteLLM's vector store search
            search_response = await litellm.vector_stores.asearch(
                vector_store_id=vector_store_id,
                query=query,
                max_num_results=5,
                custom_llm_provider="vertex_ai",
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )

            # Check if we got results
            if search_response and search_response.get("data"):
                results = []
                for item in search_response["data"]:
                    # Extract text from content
                    text = ""
                    if item.get("content"):
                        for content_item in item["content"]:
                            if content_item.get("text"):
                                text += content_item["text"]
                    
                    results.append({
                        "text": text,
                        "score": item.get("score", 0.0),
                        "file_id": item.get("file_id", ""),
                        "filename": item.get("filename", ""),
                    })

                # Check if query terms appear in results
                for result in results:
                    if query.lower() in result["text"].lower():
                        return {"results": results}

                # Return results even if exact match not found
                return {"results": results}

            return None

        except Exception as e:
            print(f"Query failed: {e}")
            return None

    @pytest.mark.asyncio
    async def test_create_corpus_and_ingest(self):
        """
        Test creating a new RAG corpus and ingesting a file.
        
        This test specifically validates:
        - Automatic corpus creation when vector_store_id is not provided
        - Long-running operation polling for corpus creation
        - File upload to the newly created corpus
        """
        litellm._turn_on_debug()

        filename, unique_id = self.get_unique_filename("create_corpus")
        text_content = f"""
        Test document {unique_id} for Vertex AI RAG corpus creation.
        This tests the automatic corpus creation feature.
        The corpus should be created and the file should be uploaded successfully.
        """.encode("utf-8")
        file_data = (filename, text_content, "text/plain")

        # Get base options WITHOUT corpus_id to trigger creation
        ingest_options = self.get_base_ingest_options()
        # Remove corpus_id if it was set from env var
        if "vector_store_id" in ingest_options.get("vector_store", {}):
            del ingest_options["vector_store"]["vector_store_id"]
        
        ingest_options["name"] = f"test-create-corpus-{unique_id}"

        try:
            response = await litellm.rag.aingest(
                ingest_options=ingest_options,
                file_data=file_data,
            )

            print(f"Create Corpus Response: {response}")

            # Validate response
            assert "id" in response
            assert response["id"].startswith("ingest_")
            assert "status" in response
            assert response["status"] == "completed", f"Expected completed, got {response['status']}"
            assert "vector_store_id" in response
            assert response["vector_store_id"], "vector_store_id should not be empty"
            
            # The vector_store_id should be a full corpus path
            corpus_id = response["vector_store_id"]
            assert "projects/" in corpus_id, "Corpus ID should be a full resource path"
            assert "ragCorpora/" in corpus_id, "Corpus ID should contain ragCorpora"
            
            print(f"✓ Successfully created corpus: {corpus_id}")
            print(f"✓ Successfully uploaded file: {response.get('file_id')}")

        except litellm.InternalServerError as e:
            pytest.skip(f"Skipping test due to litellm.InternalServerError: {e}")
        except Exception as e:
            print(f"Test failed with error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_ingest_with_existing_corpus(self):
        """
        Test ingesting a file to an existing RAG corpus.
        
        This test validates:
        - Using an existing corpus_id from environment variable
        - Direct file upload without corpus creation
        """
        corpus_id = os.environ.get("VERTEX_CORPUS_ID")
        if not corpus_id:
            pytest.skip("Skipping test: VERTEX_CORPUS_ID not set")

        litellm._turn_on_debug()

        filename, unique_id = self.get_unique_filename("existing_corpus")
        text_content = f"""
        Test document {unique_id} for existing Vertex AI RAG corpus.
        This tests file upload to a pre-existing corpus.
        """.encode("utf-8")
        file_data = (filename, text_content, "text/plain")

        ingest_options = self.get_base_ingest_options()
        ingest_options["name"] = f"test-existing-corpus-{unique_id}"

        try:
            response = await litellm.rag.aingest(
                ingest_options=ingest_options,
                file_data=file_data,
            )

            print(f"Existing Corpus Ingest Response: {response}")

            assert response["status"] == "completed"
            assert response["vector_store_id"] == corpus_id or corpus_id in response["vector_store_id"]
            assert response.get("file_id"), "file_id should be present"
            
            print(f"✓ Successfully uploaded to existing corpus: {corpus_id}")
            print(f"✓ File ID: {response.get('file_id')}")

        except litellm.InternalServerError as e:
            pytest.skip(f"Skipping test due to litellm.InternalServerError: {e}")

