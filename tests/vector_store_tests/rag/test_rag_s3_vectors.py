"""
S3 Vectors RAG ingestion tests.

Requires environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION_NAME (optional, defaults to us-west-2)

Optional:
- S3_VECTOR_BUCKET_NAME (optional, auto-generates if not set)
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import RAGIngestOptions
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest


class TestRAGS3Vectors(BaseRAGTest):
    """Test RAG Ingest with AWS S3 Vectors."""

    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        """Check required environment variables before each test."""
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not aws_key or not aws_secret:
            pytest.skip("Skipping S3 Vectors test: AWS credentials required")

    def get_base_ingest_options(self) -> RAGIngestOptions:
        """
        Return S3 Vectors-specific ingest options.

        Chunking is configured via chunking_strategy (unified interface).
        Embeddings are generated using LiteLLM's embedding API.
        """
        vector_bucket_name = os.environ.get(
            "S3_VECTOR_BUCKET_NAME", "test-litellm-vectors"
        )
        aws_region = os.environ.get("AWS_REGION_NAME", "us-west-2")

        return {
            "chunking_strategy": {
                "chunk_size": 512,
                "chunk_overlap": 100,
            },
            "embedding": {
                "model": "text-embedding-3-small"  # Can use any LiteLLM-supported model
            },
            "vector_store": {
                "custom_llm_provider": "s3_vectors",
                "vector_bucket_name": vector_bucket_name,
                "index_name": "test-index",
                # dimension is auto-detected from embedding model (text-embedding-3-small = 1536)
                "distance_metric": "cosine",
                "non_filterable_metadata_keys": ["source_text"],
                "aws_region_name": aws_region,
            },
        }

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Query S3 Vectors index."""
        try:
            # Import the ingestion class to use its query method
            from litellm.rag.ingestion.s3_vectors_ingestion import (
                S3VectorsRAGIngestion,
            )
        except ImportError:
            pytest.skip("S3 Vectors ingestion not available")

        vector_bucket_name = os.environ.get(
            "S3_VECTOR_BUCKET_NAME", "test-litellm-vectors"
        )
        aws_region = os.environ.get("AWS_REGION_NAME", "us-west-2")

        # Create ingestion instance to use query method
        ingest_options = {
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": {
                "custom_llm_provider": "s3_vectors",
                "vector_bucket_name": vector_bucket_name,
                "aws_region_name": aws_region,
            },
        }

        ingestion = S3VectorsRAGIngestion(ingest_options=ingest_options)

        # Query the index
        results = await ingestion.query_vector_store(
            vector_store_id=vector_store_id,
            query=query,
            top_k=5,
        )

        return results
