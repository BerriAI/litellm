import asyncio
import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def assert_response_shape(response, custom_llm_provider):
    expected_response_shape = {"id": str, "results": list, "meta": dict}

    expected_results_shape = {"index": int, "relevance_score": float}

    expected_meta_shape = {"api_version": dict, "billed_units": dict}

    expected_api_version_shape = {"version": str}

    expected_billed_units_shape = {"search_units": int}

    assert isinstance(response.id, expected_response_shape["id"])
    assert isinstance(response.results, expected_response_shape["results"])
    for result in response.results:
        assert isinstance(result["index"], expected_results_shape["index"])
        assert isinstance(
            result["relevance_score"], expected_results_shape["relevance_score"]
        )
    assert isinstance(response.meta, expected_response_shape["meta"])

    if custom_llm_provider == "cohere":

        assert isinstance(
            response.meta["api_version"], expected_meta_shape["api_version"]
        )
        assert isinstance(
            response.meta["api_version"]["version"],
            expected_api_version_shape["version"],
        )
        assert isinstance(
            response.meta["billed_units"], expected_meta_shape["billed_units"]
        )
        assert isinstance(
            response.meta["billed_units"]["search_units"],
            expected_billed_units_shape["search_units"],
        )


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_basic_rerank(sync_mode):
    if sync_mode is True:
        response = litellm.rerank(
            model="cohere/rerank-english-v3.0",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("re rank response: ", response)

        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="cohere")
    else:
        response = await litellm.arerank(
            model="cohere/rerank-english-v3.0",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("async re rank response: ", response)

        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="cohere")


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_basic_rerank_together_ai(sync_mode):
    if sync_mode is True:
        response = litellm.rerank(
            model="together_ai/Salesforce/Llama-Rank-V1",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("re rank response: ", response)

        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="together_ai")
    else:
        response = await litellm.arerank(
            model="together_ai/Salesforce/Llama-Rank-V1",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
        )

        print("async re rank response: ", response)

        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="together_ai")


@pytest.mark.asyncio()
async def test_rerank_custom_api_base():
    mock_response = AsyncMock()

    def return_val():
        return {
            "id": "cmpl-mockid",
            "results": [{"index": 0, "relevance_score": 0.95}],
            "meta": {
                "api_version": {"version": "1.0"},
                "billed_units": {"search_units": 1},
            },
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    expected_payload = {
        "model": "Salesforce/Llama-Rank-V1",
        "query": "hello",
        "documents": ["hello", "world"],
        "top_n": 3,
    }

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        response = await litellm.arerank(
            model="cohere/Salesforce/Llama-Rank-V1",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            api_base="https://exampleopenaiendpoint-production.up.railway.app/",
        )

        print("async re rank response: ", response)

        # Assert
        mock_post.assert_called_once()
        _url, kwargs = mock_post.call_args
        args_to_api = kwargs["json"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert _url[0] == "https://exampleopenaiendpoint-production.up.railway.app/"
        assert args_to_api == expected_payload
        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="cohere")


class TestLogger(CustomLogger):

    def __init__(self):
        self.kwargs = None
        self.response_obj = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print("in success event for rerank, kwargs = ", kwargs)
        print("in success event for rerank, response_obj = ", response_obj)
        self.kwargs = kwargs
        self.response_obj = response_obj


@pytest.mark.asyncio()
async def test_rerank_custom_callbacks():
    custom_logger = TestLogger()
    litellm.callbacks = [custom_logger]
    response = await litellm.arerank(
        model="cohere/rerank-english-v3.0",
        query="hello",
        documents=["hello", "world"],
        top_n=3,
    )

    await asyncio.sleep(5)

    print("async re rank response: ", response)
    assert custom_logger.kwargs is not None
    assert custom_logger.kwargs.get("response_cost") > 0.0
    assert custom_logger.response_obj is not None
    assert custom_logger.response_obj.results is not None
