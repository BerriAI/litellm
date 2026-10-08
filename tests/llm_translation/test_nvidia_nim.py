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

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_basic_rerank(self, sync_mode, monkeypatch):
        """
        Override the base live rerank test with a mocked HTTP layer.

        NVIDIA reached end-of-life for the hosted
        nvidia/llama-3.2-nv-rerankqa-1b-v2 rerank API on 2026-05-18 and
        published no replacement model, so a live call now returns HTTP 410
        ("Gone"). NVIDIA's hosted catalog rotates on a schedule, so pointing
        at another live model would only defer the same failure. Mock the
        transport instead (same pattern as
        test_nvidia_nim_rerank_ranking_endpoint above) so the request/response
        transformation and cost calculation stay covered offline.
        """
        monkeypatch.setenv("NVIDIA_NIM_API_KEY", "fake-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = ""
        mock_response.json.return_value = {
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.75},
            ],
            "usage": {"total_tokens": 7},
        }

        with (
            patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=mock_response,
            ),
            patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                return_value=mock_response,
            ),
        ):
            await super().test_basic_rerank(sync_mode=sync_mode)
