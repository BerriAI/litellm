import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
    get_optional_params,
    get_optional_params_embeddings,
)

# test_example.py
from abc import ABC, abstractmethod


class BaseLLMEmbeddingTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_base_embedding_call_args(self) -> dict:
        """Must return the base embedding call args"""
        pass

    @abstractmethod
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        """Must return the custom llm provider"""
        pass

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_basic_embedding(self, sync_mode):
        litellm.set_verbose = True
        embedding_call_args = self.get_base_embedding_call_args()
        if sync_mode is True:
            response = litellm.embedding(
                **embedding_call_args,
                input=["hello", "world"],
            )

            print("embedding response: ", response)
        else:
            response = await litellm.aembedding(
                **embedding_call_args,
                input=["hello", "world"],
            )

            print("async embedding response: ", response)

    def test_embedding_optional_params_max_retries(self):
        embedding_call_args = self.get_base_embedding_call_args()
        optional_params = get_optional_params_embeddings(
            **embedding_call_args, max_retries=20
        )
        assert optional_params["max_retries"] == 20
