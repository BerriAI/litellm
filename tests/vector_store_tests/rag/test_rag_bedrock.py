"""
Bedrock Knowledge Base RAG ingestion tests.

Requires environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION_NAME (optional, defaults to us-west-2)

Optional (for using existing KB instead of auto-creating):
- BEDROCK_KNOWLEDGE_BASE_ID
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.types.rag import RAGIngestOptions, BedrockVectorStoreOptions
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest


class TestRAGBedrock(BaseRAGTest):
    """Test RAG Ingest with Bedrock Knowledge Base."""

    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        """Check required environment variables before each test."""
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not aws_key or not aws_secret:
            pytest.skip("Skipping Bedrock test: AWS credentials required")

    def get_base_ingest_options(self) -> RAGIngestOptions:
        """
        Return Bedrock-specific ingest options.

        Uses unified interface - no vector_store_id means auto-create KB.
        If BEDROCK_KNOWLEDGE_BASE_ID is set, uses existing KB.
        """
        # Use existing KB if provided, otherwise auto-create
        existing_kb_id = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID")

        return {
            "vector_store": BedrockVectorStoreOptions(
                custom_llm_provider="bedrock",
                vector_store_id=existing_kb_id,  # None = auto-create
                # wait_for_ingestion defaults to False - returns immediately
            ),
        }

    async def query_vector_store(
        self,
        vector_store_id: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Query Bedrock Knowledge Base."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 required for Bedrock tests")

        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_REGION_NAME", "us-west-2"),
        )
        bedrock_agent_runtime = session.client("bedrock-agent-runtime")

        response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=vector_store_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            },
        )

        if response.get("retrievalResults") and len(response["retrievalResults"]) > 0:
            # Check if query terms appear in results
            for result in response["retrievalResults"]:
                # Extract unique_id from query if present
                if query in result["content"]["text"]:
                    return response
            # Return results even if exact match not found
            return response
        return None

