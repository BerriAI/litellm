"""
Minimal Gemini File Search vector store tests.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from base_vector_store_test import BaseVectorStoreTest


class TestGeminiVectorStore(BaseVectorStoreTest):
    """Reuses the shared vector store smoke suite with Gemini."""

    def get_base_request_args(self) -> dict:
        """Provide arguments for the shared search test."""
        return {
            "vector_store_id": os.getenv("GEMINI_TEST_STORE_ID", "fileSearchStores/example-test-store"),
            "custom_llm_provider": "gemini",
            "query": "LiteLLM",
        }

    def get_base_create_vector_store_args(self) -> dict:
        """Ensure we always call Gemini when creating a vector store."""
        return {
            "custom_llm_provider": "gemini",
        }

