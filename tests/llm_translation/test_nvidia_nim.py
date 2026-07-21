import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from base_rerank_unit_tests import BaseLLMRerankTest
import litellm


def test_completion_nvidia_nim():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
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


def test_embedding_nvidia_nim():
    litellm.set_verbose = True
    from openai import OpenAI

    client = OpenAI(
        api_key="fake-api-key",
    )
    with patch.object(client.embeddings.with_raw_response, "create") as mock_client:
        try:
            litellm.embedding(
                model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
                input="What is the meaning of life?",
                input_type="passage",
                dimensions=1024,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        print("request_body: ", request_body)
        assert request_body["input"] == "What is the meaning of life?"
        assert request_body["model"] == "nvidia/nv-embedqa-e5-v5"
        assert request_body["extra_body"]["input_type"] == "passage"
        assert request_body["dimensions"] == 1024


def test_chat_completion_nvidia_nim_with_tools():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/meta/llama3-70b-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. EST, PST",
                        },
                    },
                    "required": ["timezone"],
                },
            },
        },
    ]

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today and what time is it in EST?",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0.7,
                client=client,
            )
        except Exception as e:
            print(e)

        # Add assertions to check the request
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today and what time is it in EST?",
            },
        ]
        assert request_body["model"] == "meta/llama3-70b-instruct"
        assert request_body["temperature"] == 0.7
        assert request_body["tools"] == tools
        assert request_body["tool_choice"] == "auto"
        assert request_body["parallel_tool_calls"] == True


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


# ---------------------------------------------------------------------------
# Regression tests for https://github.com/BerriAI/litellm/issues/34165
#
# The native /v1/ranking endpoint accepts only model, query, passages, and
# truncate. Two defects are covered here:
# 1. structured image documents were json.dumps-stringified into text passages
# 2. Cohere top_n was mapped to top_k, which /v1/ranking rejects with a 400
# ---------------------------------------------------------------------------

from litellm.llms.nvidia_nim.rerank.ranking_transformation import (
    NvidiaNimRankingConfig,
)
from litellm.types.rerank import RerankResponse

RANKING_MODEL = "ranking/nvidia/llama-nemotron-rerank-vl-1b-v2"
IMAGE_DOC = {"image": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}
TEXT_DOC = {"text": "a plain text passage"}
MIXED_DOC = {"text": "caption for the image", "image": "data:image/png;base64,iVBORw0KGgo="}


def _build_ranking_request(documents, top_n=None, non_default_params=None):
    """Run map_cohere_rerank_params + transform_rerank_request for /v1/ranking."""
    config = NvidiaNimRankingConfig()
    optional_params = config.map_cohere_rerank_params(
        non_default_params=non_default_params,
        model=RANKING_MODEL,
        drop_params=False,
        query="which passage shows a cat?",
        documents=documents,
        top_n=top_n,
    )
    request_data = config.transform_rerank_request(
        model=RANKING_MODEL,
        optional_rerank_params=optional_params,
        headers={},
    )
    return config, request_data


def _build_ranking_response(config, request_data, rankings):
    """Run transform_rerank_response against a mocked raw ranking response."""
    raw_response = MagicMock()
    raw_response.json.return_value = {"rankings": rankings}
    return config.transform_rerank_response(
        model=RANKING_MODEL,
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
        request_data=request_data,
    )


class TestNvidiaNimRankingRequestTransform:
    def test_string_documents(self):
        _, request_data = _build_ranking_request(["passage one", "passage two"])
        assert request_data["passages"] == [
            {"text": "passage one"},
            {"text": "passage two"},
        ]

    def test_text_object_documents(self):
        _, request_data = _build_ranking_request([TEXT_DOC])
        assert request_data["passages"] == [TEXT_DOC]

    def test_image_object_documents_are_preserved(self):
        _, request_data = _build_ranking_request([IMAGE_DOC, TEXT_DOC])
        assert request_data["passages"] == [IMAGE_DOC, TEXT_DOC]

    def test_mixed_text_image_documents_are_preserved(self):
        _, request_data = _build_ranking_request([MIXED_DOC])
        assert request_data["passages"] == [MIXED_DOC]

    def test_unsupported_dict_documents_are_stringified(self):
        doc = {"title": "no supported fields here"}
        _, request_data = _build_ranking_request([doc])
        assert request_data["passages"] == [{"text": json.dumps(doc)}]

    def test_top_n_is_not_sent_to_the_ranking_endpoint(self):
        _, request_data = _build_ranking_request(["a", "b"], top_n=1)
        assert "top_k" not in request_data
        assert "top_n" not in request_data

    def test_provider_specific_top_k_is_stripped(self):
        _, request_data = _build_ranking_request(["a", "b"], non_default_params={"top_k": 2})
        assert "top_k" not in request_data

    @pytest.mark.parametrize("invalid_top_n", [0, -1, 1.5, "2", True])
    def test_invalid_top_n_raises_value_error(self, invalid_top_n):
        with pytest.raises(ValueError, match="top_n"):
            _build_ranking_request(["a", "b"], top_n=invalid_top_n)


class TestNvidiaNimRankingResponseTransform:
    RANKINGS = [
        {"index": 0, "logit": 0.95},
        {"index": 1, "logit": 0.75},
        {"index": 2, "logit": 0.55},
    ]

    def test_top_n_one_truncates_to_best_result(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=1)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 1
        assert response.results[0]["index"] == 0

    def test_top_n_equal_to_document_count_keeps_all_results(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=3)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 3

    def test_top_n_greater_than_document_count_keeps_all_results(self):
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=10)
        response = _build_ranking_response(config, request_data, self.RANKINGS)
        assert len(response.results) == 3

    def test_top_n_truncation_keeps_most_relevant_results(self):
        unsorted_rankings = [
            {"index": 0, "logit": 0.10},
            {"index": 1, "logit": 0.90},
            {"index": 2, "logit": 0.50},
        ]
        config, request_data = _build_ranking_request(["a", "b", "c"], top_n=2)
        response = _build_ranking_response(config, request_data, unsorted_rankings)
        assert [result["index"] for result in response.results] == [1, 2]

    def test_image_only_passages_do_not_break_document_echo(self):
        config, request_data = _build_ranking_request([IMAGE_DOC, TEXT_DOC])
        response = _build_ranking_response(config, request_data, self.RANKINGS[:2])
        assert len(response.results) == 2
        # Image-only passage has no text to echo back
        assert "document" not in response.results[0]
        assert response.results[1]["document"] == {"text": TEXT_DOC["text"]}


@pytest.mark.asyncio()
async def test_nvidia_nim_ranking_endpoint_image_documents_and_top_n():
    """
    End-to-end (mocked transport): image documents reach /v1/ranking intact
    and top_n is applied client-side instead of being sent as top_k.
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
        response = await litellm.arerank(
            model="nvidia_nim/ranking/nvidia/llama-nemotron-rerank-vl-1b-v2",
            query="which passage shows a cat?",
            documents=[IMAGE_DOC, TEXT_DOC],
            top_n=1,
            api_key="fake-api-key",
        )

        mock_post.assert_called_once()
        request_data = json.loads(mock_post.call_args.kwargs["data"])

        assert mock_post.call_args.kwargs["url"] == "https://ai.api.nvidia.com/v1/ranking"
        # Image passage preserved as-is, not stringified into text
        assert request_data["passages"] == [IMAGE_DOC, TEXT_DOC]
        # Neither top_k nor top_n is sent to the native endpoint
        assert "top_k" not in request_data
        assert "top_n" not in request_data
        # top_n applied client-side on the converted response
        assert len(response.results) == 1
        assert response.results[0]["index"] == 0
