import asyncio
import httpx
import json
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
    get_optional_params,
)
import base64
from pathlib import Path

from abc import ABC, abstractmethod

file_data = (Path(__file__).parent.parent / "white_100x100.png").read_bytes()

encoded_file = base64.b64encode(file_data).decode("utf-8")
base64_image = f"data:image/png;base64,{encoded_file}"

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

        from openai.types.create_embedding_response import CreateEmbeddingResponse

        CreateEmbeddingResponse.model_validate(response.model_dump())

