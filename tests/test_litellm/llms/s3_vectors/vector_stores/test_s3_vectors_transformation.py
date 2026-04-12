from unittest.mock import MagicMock, Mock

import httpx
import pytest

from litellm.llms.s3_vectors.vector_stores.transformation import (
    S3VectorsVectorStoreConfig,
)
from litellm.types.vector_stores import VectorStoreSearchResponse


class TestS3VectorsVectorStoreConfig:
    def test_init(self):
        """Test that S3VectorsVectorStoreConfig initializes correctly"""
        config = S3VectorsVectorStoreConfig()
        assert config is not None

    def test_get_supported_openai_params(self):
        """Test that supported OpenAI params are returned"""
        config = S3VectorsVectorStoreConfig()
        params = config.get_supported_openai_params("test-model")
        assert "max_num_results" in params

    def test_get_complete_url(self):
        """Test URL generation for S3 Vectors"""
        config = S3VectorsVectorStoreConfig()
        litellm_params = {"aws_region_name": "us-west-2"}
        url = config.get_complete_url(None, litellm_params)
        assert url == "https://s3vectors.us-west-2.api.aws"

    def test_get_complete_url_missing_region(self):
        """Test that missing region raises error"""
        config = S3VectorsVectorStoreConfig()
        litellm_params = {}
        with pytest.raises(ValueError, match="aws_region_name is required"):
            config.get_complete_url(None, litellm_params)

    @pytest.mark.skip(reason="Requires embedding API call, tested in integration tests")
    def test_transform_search_request(self):
        """Test search request transformation"""
        # This test requires making an actual embedding API call
        # It's better tested in integration tests
        pass

    def test_transform_search_request_invalid_vector_store_id(self):
        """Test that invalid vector_store_id format raises error"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {}

        with pytest.raises(
            ValueError, match="vector_store_id must be in format 'bucket_name:index_name'"
        ):
            config.transform_search_vector_store_request(
                vector_store_id="invalid-format",
                query="test query",
                vector_store_search_optional_params={},
                api_base="https://s3vectors.us-west-2.api.aws",
                litellm_logging_obj=mock_logging_obj,
                litellm_params={},
            )

    def test_transform_search_response(self):
        """Test search response transformation"""
        config = S3VectorsVectorStoreConfig()
        mock_logging_obj = Mock()
        mock_logging_obj.model_call_details = {"query": "test query"}

        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "vectors": [
                {
                    "distance": 0.05,  # S3 Vectors returns distance, not score
                    "metadata": {
                        "source_text": "This is test content",
                        "chunk_index": "0",
                        "filename": "test.pdf",
                    },
                },
                {
                    "distance": 0.15,
                    "metadata": {
                        "source_text": "More test content",
                        "chunk_index": "1",
                    },
                },
            ]
        }
        mock_response.status_code = 200
        mock_response.headers = {}

        result = config.transform_search_vector_store_response(
            mock_response, mock_logging_obj
        )

        # VectorStoreSearchResponse is a TypedDict, so check structure instead of isinstance
        assert result["object"] == "vector_store.search_results.page"
        assert result["search_query"] == "test query"
        assert len(result["data"]) == 2
        # Score should be 1 - distance (cosine similarity)
        assert result["data"][0]["score"] == 0.95  # 1 - 0.05
        assert result["data"][0]["content"][0]["text"] == "This is test content"
        assert result["data"][0]["filename"] == "test.pdf"
        assert result["data"][1]["score"] == 0.85  # 1 - 0.15
        assert result["data"][1]["content"][0]["text"] == "More test content"

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping"""
        config = S3VectorsVectorStoreConfig()
        non_default_params = {"max_num_results": 5}
        optional_params = {}

        result = config.map_openai_params(non_default_params, optional_params, False)

        assert result["maxResults"] == 5
