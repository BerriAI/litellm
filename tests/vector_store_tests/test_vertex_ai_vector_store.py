import os
import pytest
from unittest.mock import Mock, patch

from litellm.llms.vertex_ai.vector_stores.transformation import VertexVectorStoreConfig
from litellm.types.vector_stores import (
    VectorStoreCreateResponse,
    VectorStoreSearchResponse,
    VectorStoreSearchResult,
    VectorStoreResultContent,
)
from tests.vector_store_tests.base_vector_store_test import BaseVectorStoreTest


class TestVertexAIVectorStore(BaseVectorStoreTest):
    def get_base_create_vector_store_args(self) -> dict:
        """Must return the base create vector store args"""
        return {}
    
    def get_base_request_args(self):
        return {
            "vector_store_id": "6917529027641081856",
            "custom_llm_provider": "vertex_ai",
            "vertex_project": "reliablekeys",
            "vertex_location": "us-central1",
            "query": "what happens after we add a model"
        }
