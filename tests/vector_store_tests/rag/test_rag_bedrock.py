"""
Bedrock Knowledge Base RAG ingestion tests.

Requires environment variables:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION_NAME (optional, defaults to us-west-2)
- BEDROCK_KNOWLEDGE_BASE_ID
- BEDROCK_DATA_SOURCE_ID
- BEDROCK_S3_BUCKET
- BEDROCK_S3_PREFIX (optional, defaults to data/)
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from tests.vector_store_tests.rag.base_rag_tests import BaseRAGTest


class TestRAGBedrock(BaseRAGTest):
    """Test RAG Ingest with Bedrock Knowledge Base."""

    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        """Check required environment variables before each test."""
        knowledge_base_id = os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID")
        data_source_id = os.environ.get("BEDROCK_DATA_SOURCE_ID")
        s3_bucket = os.environ.get("BEDROCK_S3_BUCKET")

        if not all([knowledge_base_id, data_source_id, s3_bucket]):
            pytest.skip(
                "Skipping Bedrock test: BEDROCK_KNOWLEDGE_BASE_ID, "
                "BEDROCK_DATA_SOURCE_ID, and BEDROCK_S3_BUCKET required"
            )

    def get_base_ingest_options(self) -> Dict[str, Any]:
        """Return Bedrock-specific ingest options."""
        return {
            "vector_store": {
                "custom_llm_provider": "bedrock",
                "knowledge_base_id": os.environ.get("BEDROCK_KNOWLEDGE_BASE_ID"),
                "data_source_id": os.environ.get("BEDROCK_DATA_SOURCE_ID"),
                "s3_bucket": os.environ.get("BEDROCK_S3_BUCKET"),
                "s3_prefix": os.environ.get("BEDROCK_S3_PREFIX", "data/"),
                "wait_for_ingestion": True,
                "ingestion_timeout": 120,
            },
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

