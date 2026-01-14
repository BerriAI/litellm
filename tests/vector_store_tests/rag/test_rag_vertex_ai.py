"""
Vertex AI RAG Engine ingestion tests.

Requires:
- gcloud auth application-default login (for ADC authentication)

Environment variables:
- VERTEX_PROJECT: GCP project ID (required)
- VERTEX_LOCATION: GCP region (optional, defaults to europe-west1)
- VERTEX_CORPUS_ID: Existing RAG corpus ID (required for Vertex AI)
- GCS_BUCKET_NAME: GCS bucket for file uploads (required)
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
        corpus_id = os.environ.get("VERTEX_CORPUS_ID")
        gcs_bucket = os.environ.get("GCS_BUCKET_NAME")

        if not vertex_project:
            pytest.skip("Skipping Vertex AI test: VERTEX_PROJECT required")

        if not corpus_id:
            pytest.skip("Skipping Vertex AI test: VERTEX_CORPUS_ID required")

        if not gcs_bucket:
            pytest.skip("Skipping Vertex AI test: GCS_BUCKET_NAME required")

        # Check if vertexai is installed
        try:
            from vertexai import rag
        except ImportError:
            pytest.skip("Skipping Vertex AI test: google-cloud-aiplatform>=1.60.0 required")

    def get_base_ingest_options(self) -> RAGIngestOptions:
        """
        Return Vertex AI-specific ingest options.

        Chunking is configured via chunking_strategy (unified interface),
        not inside vector_store.
        """
        corpus_id = os.environ.get("VERTEX_CORPUS_ID")
        vertex_project = os.environ.get("VERTEX_PROJECT")
        vertex_location = os.environ.get("VERTEX_LOCATION", "europe-west1")
        gcs_bucket = os.environ.get("GCS_BUCKET_NAME")

        return {
            "chunking_strategy": {
                "chunk_size": 512,
                "chunk_overlap": 100,
            },
            "vector_store": {
                "custom_llm_provider": "vertex_ai",
                "vertex_project": vertex_project,
                "vertex_location": vertex_location,
                "vector_store_id": corpus_id,
                "gcs_bucket": gcs_bucket,
                "wait_for_import": True,
            },
        }

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Query Vertex AI RAG corpus."""
        try:
            from vertexai import init as vertexai_init
            from vertexai import rag
        except ImportError:
            pytest.skip("vertexai required for Vertex AI tests")

        vertex_project = os.environ.get("VERTEX_PROJECT")
        vertex_location = os.environ.get("VERTEX_LOCATION", "europe-west1")

        # Initialize Vertex AI
        vertexai_init(project=vertex_project, location=vertex_location)

        # Build corpus name
        corpus_name = f"projects/{vertex_project}/locations/{vertex_location}/ragCorpora/{vector_store_id}"

        # Query the corpus
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(rag_corpus=corpus_name)
            ],
            text=query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=5,
            ),
        )

        if hasattr(response, 'contexts') and response.contexts.contexts:
            # Convert to dict format
            results = []
            for ctx in response.contexts.contexts:
                results.append({
                    "text": ctx.text,
                    "score": ctx.score,
                    "source_uri": ctx.source_uri,
                })

            # Check if query terms appear in results
            for result in results:
                if query.lower() in result["text"].lower():
                    return {"results": results}

            # Return results even if exact match not found
            return {"results": results}

        return None

