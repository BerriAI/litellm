from base_vector_store_test import BaseVectorStoreTest
import os
import pytest


class TestS3VectorsVectorStore(BaseVectorStoreTest):
    @pytest.fixture(autouse=True)
    def check_env_vars(self):
        """Check if required environment variables are set"""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {', '.join(missing_vars)}")

    def get_base_request_args(self) -> dict:
        """
        Must return the base request args for searching.
        For S3 Vectors, vector_store_id should be in format: bucket_name:index_name
        """
        return {
            "custom_llm_provider": "s3_vectors",
            "vector_store_id": os.getenv(
                "S3_VECTORS_VECTOR_STORE_ID", "test-litellm-vectors:test-index"
            ),
            "query": "What is machine learning?",
            "aws_region_name": os.getenv("AWS_REGION_NAME", "us-west-2"),
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }

    def get_base_create_vector_store_args(self) -> dict:
        """
        Vector store creation is not yet implemented for S3 Vectors.
        This test will be skipped.
        """
        return {}

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_create_vector_store(self, sync_mode):
        """S3 Vectors doesn't support vector store creation via this API yet"""
        pytest.skip("Vector store creation not yet implemented for S3 Vectors")
