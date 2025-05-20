"""
Unit tests for Morph rerank configuration.

These tests validate the MorphRerankConfig class which extends BaseRerankConfig.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.morph.rerank.transformation import MorphRerankConfig, MorphError


class TestMorphRerankConfig:
    """Test class for MorphRerankConfig functionality"""

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        config = MorphRerankConfig()
        headers = {}
        api_key = "fake-morph-key"

        result = config.validate_environment(
            headers=headers,
            model="morph/morph-rerank-v2",
            api_key=api_key,
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["accept"] == "application/json"
        assert result["content-type"] == "application/json"

    def test_complete_url(self):
        """Test the get_complete_url method"""
        config = MorphRerankConfig()
        
        # Test with custom API base
        custom_base = "https://custom.morphllm.com"
        url = config.get_complete_url(api_base=custom_base, model="morph/morph-rerank-v2")
        assert url == "https://custom.morphllm.com/v1/rerank"
        
        # Test with API base already having /v1/rerank
        custom_base_with_path = "https://custom.morphllm.com/v1/rerank"
        url = config.get_complete_url(api_base=custom_base_with_path, model="morph/morph-rerank-v2")
        assert url == "https://custom.morphllm.com/v1/rerank"
        
        # Test with no API base
        url = config.get_complete_url(api_base=None, model="morph/morph-rerank-v2")
        assert url == "https://api.morphllm.com/v1/rerank"

    def test_missing_api_key(self):
        """Test error handling when API key is missing"""
        config = MorphRerankConfig()
        
        with pytest.raises(ValueError) as excinfo:
            config.validate_environment(
                headers={},
                model="morph/morph-rerank-v2",
                api_key=None,
            )

        assert "Morph API key is required" in str(excinfo.value)
        
    def test_transform_rerank_request(self):
        """Test the transform_rerank_request method"""
        config = MorphRerankConfig()
        
        # Test with documents
        optional_params = {
            "query": "test query",
            "documents": ["doc1", "doc2"],
            "top_n": 2,
            "return_documents": True
        }
        
        result = config.transform_rerank_request(
            model="morph/morph-rerank-v2",
            optional_rerank_params=optional_params,
            headers={}
        )
        
        assert result["model"] == "morph-rerank-v2"
        assert result["query"] == "test query"
        assert result["documents"] == ["doc1", "doc2"]
        assert result["top_n"] == 2
        assert result["return_documents"] is True
        
        # Test with embedding_ids
        optional_params = {
            "query": "test query",
            "embedding_ids": ["id1", "id2"],
            "top_n": 2
        }
        
        result = config.transform_rerank_request(
            model="morph/morph-rerank-v2",
            optional_rerank_params=optional_params,
            headers={}
        )
        
        assert result["model"] == "morph-rerank-v2"
        assert result["query"] == "test query"
        assert result["embedding_ids"] == ["id1", "id2"]
        assert result["top_n"] == 2
        
    def test_transform_rerank_request_missing_query(self):
        """Test transform_rerank_request with missing query"""
        config = MorphRerankConfig()
        
        optional_params = {
            "documents": ["doc1", "doc2"]
        }
        
        with pytest.raises(ValueError) as excinfo:
            config.transform_rerank_request(
                model="morph/morph-rerank-v2",
                optional_rerank_params=optional_params,
                headers={}
            )
            
        assert "query is required for Morph rerank" in str(excinfo.value)
        
    def test_transform_rerank_request_missing_documents_and_embedding_ids(self):
        """Test transform_rerank_request with missing documents and embedding_ids"""
        config = MorphRerankConfig()
        
        optional_params = {
            "query": "test query"
        }
        
        with pytest.raises(ValueError) as excinfo:
            config.transform_rerank_request(
                model="morph/morph-rerank-v2",
                optional_rerank_params=optional_params,
                headers={}
            )
            
        assert "Either documents or embedding_ids is required" in str(excinfo.value)
    
    def test_get_supported_cohere_rerank_params(self):
        """Test get_supported_cohere_rerank_params method"""
        config = MorphRerankConfig()
        
        params = config.get_supported_cohere_rerank_params(model="morph/morph-rerank-v2")
        
        assert "query" in params
        assert "documents" in params
        assert "top_n" in params
        assert "return_documents" in params
        assert "embedding_ids" in params
    
    def test_transform_rerank_response(self):
        """Test the transform_rerank_response method"""
        config = MorphRerankConfig()
        
        # Create a mock response
        raw_response = httpx.Response(
            status_code=200,
            content=b"""
            {
                "model": "morph-rerank-v2",
                "results": [
                    {
                        "index": 0,
                        "document": "This is a test document",
                        "relevance_score": 0.92
                    },
                    {
                        "index": 1,
                        "document": "This is another document",
                        "relevance_score": 0.75
                    }
                ]
            }
            """
        )
        
        from litellm.types.utils import RerankResponse
        
        model_response = RerankResponse()
        logging_obj = None  # Mock logging object
        
        result = config.transform_rerank_response(
            model="morph/morph-rerank-v2",
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj
        )
        
        # Verify response transformation
        assert result.id == "morph-rerank-v2"  # id should be copied from model
        assert len(result.results) == 2
        assert result.results[0].index == 0
        assert result.results[0].document == "This is a test document"
        assert result.results[0].relevance_score == 0.92
        assert result.results[1].index == 1
        assert result.results[1].document == "This is another document"
        assert result.results[1].relevance_score == 0.75 