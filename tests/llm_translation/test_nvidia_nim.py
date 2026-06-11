import json
import os
import sys
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import pytest
from unittest.mock import patch, MagicMock

import litellm
from base_rerank_unit_tests import BaseLLMRerankTest


@pytest.mark.asyncio()
async def test_nvidia_nim_rerank_ranking_endpoint():
    """
    Test that using "nvidia_nim/ranking/<model>" forces the /v1/ranking endpoint.

    This allows users to explicitly use the /v1/ranking endpoint for models like
    nvidia/llama-3.2-nv-rerankqa-1b-v2.

    Reference: https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy
    """
    mock_response = AsyncMock()

    def return_val():
        return {
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.75},
            ],
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Use "ranking/" prefix to force /v1/ranking endpoint
        response = await litellm.arerank(
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            query="What is the GPU memory bandwidth?",
            documents=[
                "H100 delivers 3TB/s memory bandwidth",
                "A100 has 2TB/s memory bandwidth",
            ],
            top_n=2,
            api_key="fake-api-key",
        )

        mock_post.assert_called_once()

        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("url = ", _url)

        # Verify URL is /v1/ranking
        assert _url == "https://ai.api.nvidia.com/v1/ranking"

        # Verify request body structure
        request_data = json.loads(args_to_api)
        print("request_data=", request_data)

        # Query should be an object with 'text' field
        assert request_data["query"] == {"text": "What is the GPU memory bandwidth?"}

        # Documents should be 'passages'
        assert request_data["passages"] == [
            {"text": "H100 delivers 3TB/s memory bandwidth"},
            {"text": "A100 has 2TB/s memory bandwidth"},
        ]

        # Model name in body should NOT have "ranking/" prefix
        assert request_data["model"] == "nvidia/llama-3.2-nv-rerankqa-1b-v2"


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
