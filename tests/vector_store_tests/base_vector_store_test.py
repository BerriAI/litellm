import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os
import uuid
import time
import base64

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from abc import ABC, abstractmethod
from litellm.integrations.custom_logger import CustomLogger
import json
from litellm.types.utils import StandardLoggingPayload

class BaseVectorStoreTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """
    @abstractmethod
    def get_base_request_args(self) -> dict:
        """Must return the base request args"""
        pass

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_search_vector_store(self, sync_mode):
        litellm._turn_on_debug()
        litellm.set_verbose = True
        base_request_args = self.get_base_request_args()
        try: 
            if sync_mode:
                response = litellm.vector_stores.search(
                    query="Basic ping", 
                    **base_request_args
                )
            else:
                response = await litellm.vector_stores.asearch(
                    query="Basic ping", 
                    **base_request_args
                )
        except litellm.InternalServerError: 
            pytest.skip("Skipping test due to litellm.InternalServerError")
        
        print("litellm response=", json.dumps(response, indent=4, default=str))
        
        # Validate response structure
        self._validate_vector_store_response(response)

    def _validate_vector_store_response(self, response):
        """Validate the structure and content of a vector store search response"""
        
        # Check that response is a dictionary
        assert isinstance(response, dict), f"Response should be a dict, got {type(response)}"
        
        # Check required top-level fields
        required_fields = ['object', 'search_query', 'data']
        for field in required_fields:
            assert field in response, f"Missing required field '{field}' in response"
        
        # Validate object field
        assert response['object'] == 'vector_store.search_results.page', \
            f"Expected object to be 'vector_store.search_results.page', got '{response['object']}'"
        
        # Validate search_query field
        assert isinstance(response['search_query'], list), \
            f"search_query should be a list, got {type(response['search_query'])}"
        assert len(response['search_query']) > 0, "search_query should not be empty"
        assert all(isinstance(query, str) for query in response['search_query']), \
            "All items in search_query should be strings"
        
        # Validate data field
        assert isinstance(response['data'], list), \
            f"data should be a list, got {type(response['data'])}"
        
        # Validate each result in data
        for i, result in enumerate(response['data']):
            self._validate_search_result(result, i)
        
        print(f"✅ Response validation passed: Found {len(response['data'])} search results")

    def _validate_search_result(self, result, index):
        """Validate an individual search result"""
        
        # Check that result is a dictionary
        assert isinstance(result, dict), f"Result {index} should be a dict, got {type(result)}"
        
        # Check required fields in each result
        required_result_fields = ['file_id', 'filename', 'score', 'attributes', 'content']
        for field in required_result_fields:
            assert field in result, f"Missing required field '{field}' in result {index}"
        
        # Validate file_id
        assert isinstance(result['file_id'], str), \
            f"file_id should be a string, got {type(result['file_id'])} in result {index}"
        assert len(result['file_id']) > 0, f"file_id should not be empty in result {index}"
        
        # Validate filename
        assert isinstance(result['filename'], str), \
            f"filename should be a string, got {type(result['filename'])} in result {index}"
        assert len(result['filename']) > 0, f"filename should not be empty in result {index}"
        
        # Validate score
        assert isinstance(result['score'], (int, float)), \
            f"score should be a number, got {type(result['score'])} in result {index}"
        assert 0.0 <= result['score'] <= 1.0, \
            f"score should be between 0.0 and 1.0, got {result['score']} in result {index}"
        
        # Validate attributes
        assert isinstance(result['attributes'], dict), \
            f"attributes should be a dict, got {type(result['attributes'])} in result {index}"
        
        # Validate content
        assert isinstance(result['content'], list), \
            f"content should be a list, got {type(result['content'])} in result {index}"
        assert len(result['content']) > 0, f"content should not be empty in result {index}"
        
        # Validate each content item
        for j, content_item in enumerate(result['content']):
            assert isinstance(content_item, dict), \
                f"Content item {j} in result {index} should be a dict, got {type(content_item)}"
            assert 'type' in content_item, \
                f"Content item {j} in result {index} missing 'type' field"
            assert 'text' in content_item, \
                f"Content item {j} in result {index} missing 'text' field"
            assert isinstance(content_item['text'], str), \
                f"Content text should be a string in item {j} of result {index}"
            assert len(content_item['text']) > 0, \
                f"Content text should not be empty in item {j} of result {index}"
        
        print(f"✅ Result {index} validation passed: {result['filename']} (score: {result['score']:.4f})")
