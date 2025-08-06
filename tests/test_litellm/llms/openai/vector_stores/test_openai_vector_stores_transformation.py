import pytest

from litellm.llms.openai.vector_stores.transformation import OpenAIVectorStoreConfig
from litellm.types.vector_stores import (
    VectorStoreCreateOptionalRequestParams,
)


class TestOpenAIVectorStoreAPIConfig:

    @pytest.mark.parametrize(
        "metadata", [{}, None]
    )
    def test_transform_create_vector_store_request_with_metadata_empty_or_none(self, metadata):
        """
        Test transform_create_vector_store_request when metadata is None or empty dict.
        """
        config = OpenAIVectorStoreConfig()
        api_base = "https://api.openai.com/v1/vector_stores"
        
        vector_store_create_params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-vector-store",
            "file_ids": ["file-123", "file-456"],
            "metadata": metadata,
        }
        
        url, request_body = config.transform_create_vector_store_request(
            vector_store_create_params, api_base
        )

        assert url == api_base
        assert request_body["name"] == "test-vector-store"
        assert request_body["file_ids"] == ["file-123", "file-456"]
        assert request_body["metadata"] == metadata


    def test_transform_create_vector_store_request_with_large_metadata(self):
        """
        Test transform_create_vector_store_request with metadata exceeding 16 keys.
        
        OpenAI limits metadata to 16 keys maximum.
        """
        config = OpenAIVectorStoreConfig()
        api_base = "https://api.openai.com/v1/vector_stores"
        
        # Create metadata with more than 16 keys
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(20)}
        
        vector_store_create_params: VectorStoreCreateOptionalRequestParams = {
            "name": "test-vector-store",
            "metadata": large_metadata,
        }
        
        url, request_body = config.transform_create_vector_store_request(
            vector_store_create_params, api_base
        )
        
        assert url == api_base
        assert request_body["name"] == "test-vector-store"
        
        # Should be trimmed to 16 keys
        assert len(request_body["metadata"]) == 16
        
        # Should contain the first 16 keys (as per add_openai_metadata implementation)
        for i in range(16):
            assert f"key_{i}" in request_body["metadata"]
            assert request_body["metadata"][f"key_{i}"] == f"value_{i}"
