import json
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from unittest.mock import patch, MagicMock

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from base_rerank_unit_tests import BaseLLMRerankTest
from tests.capturing_transport import CapturingTransport

def test_completion_nvidia_nim():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
                client=client,
            )
        except Exception as e:
            print(e)
        # Add any assertions here to check the response

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            },
        ]
        assert request_body["model"] == "databricks/dbrx-instruct"
        assert request_body["frequency_penalty"] == 0.1
        assert request_body["presence_penalty"] == 0.5

class TestNvidiaNim(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.NVIDIA_NIM

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
        }

    def get_expected_cost(self) -> float:
        """Nvidia NIM rerank models are free (cost = 0.0)"""
        return 0.0

