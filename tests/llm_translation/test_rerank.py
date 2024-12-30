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
from litellm.types.rerank import RerankResponse
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
    litellm.set_verbose = True
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

    print("response", response.model_dump_json(indent=4))


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
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_basic_rerank_azure_ai(sync_mode):
    import os

    litellm.set_verbose = True

    if sync_mode is True:
        response = litellm.rerank(
            model="azure_ai/Cohere-rerank-v3-multilingual-ko",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            api_key=os.getenv("AZURE_AI_COHERE_API_KEY"),
            api_base=os.getenv("AZURE_AI_COHERE_API_BASE"),
        )

        print("re rank response: ", response)

        assert response.id is not None
        assert response.results is not None

        assert_response_shape(response, custom_llm_provider="together_ai")
    else:
        response = await litellm.arerank(
            model="azure_ai/Cohere-rerank-v3-multilingual-ko",
            query="hello",
            documents=["hello", "world"],
            top_n=3,
            api_key=os.getenv("AZURE_AI_COHERE_API_KEY"),
            api_base=os.getenv("AZURE_AI_COHERE_API_BASE"),
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
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    expected_payload = {
        "model": "Salesforce/Llama-Rank-V1",
        "query": "hello",
        "top_n": 3,
        "documents": ["hello", "world"],
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
        print("call args", mock_post.call_args)
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("Arguments passed to API=", args_to_api)
        print("url = ", _url)
        assert (
            _url == "https://exampleopenaiendpoint-production.up.railway.app/v1/rerank"
        )

        request_data = json.loads(args_to_api)
        assert request_data["query"] == expected_payload["query"]
        assert request_data["documents"] == expected_payload["documents"]
        assert request_data["top_n"] == expected_payload["top_n"]
        assert request_data["model"] == expected_payload["model"]

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
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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


def test_complete_base_url_cohere():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    litellm.api_base = "http://localhost:4000"
    litellm.set_verbose = True

    text = "Hello there!"
    list_texts = ["Hello there!", "How are you?", "How do you do?"]

    rerank_model = "rerank-multilingual-v3.0"

    with patch.object(client, "post") as mock_post:
        try:
            litellm.rerank(
                model=rerank_model,
                query=text,
                documents=list_texts,
                custom_llm_provider="cohere",
                client=client,
            )
        except Exception as e:
            print(e)

        print("mock_post.call_args", mock_post.call_args)
        mock_post.assert_called_once()
        assert "http://localhost:4000/v1/rerank" in mock_post.call_args.kwargs["url"]


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "top_n_1, top_n_2, expect_cache_hit",
    [
        (3, 3, True),
        (3, None, False),
    ],
)
async def test_basic_rerank_caching(sync_mode, top_n_1, top_n_2, expect_cache_hit):
    from litellm.caching.caching import Cache

    litellm.set_verbose = True
    litellm.cache = Cache(type="local")

    if sync_mode is True:
        for idx in range(2):
            if idx == 0:
                top_n = top_n_1
            else:
                top_n = top_n_2
            response = litellm.rerank(
                model="cohere/rerank-english-v3.0",
                query="hello",
                documents=["hello", "world"],
                top_n=top_n,
            )
    else:
        for idx in range(2):
            if idx == 0:
                top_n = top_n_1
            else:
                top_n = top_n_2
            response = await litellm.arerank(
                model="cohere/rerank-english-v3.0",
                query="hello",
                documents=["hello", "world"],
                top_n=top_n,
            )

            await asyncio.sleep(1)

    if expect_cache_hit is True:
        assert "cache_key" in response._hidden_params
    else:
        assert "cache_key" not in response._hidden_params

    print("re rank response: ", response)

    assert response.id is not None
    assert response.results is not None

    assert_response_shape(response, custom_llm_provider="cohere")


def test_rerank_response_assertions():
    r = RerankResponse(
        **{
            "id": "ab0fcca0-b617-11ef-b292-0242ac110002",
            "results": [
                {"index": 2, "relevance_score": 0.9958819150924683, "document": None},
                {"index": 0, "relevance_score": 0.001293411129154265, "document": None},
                {
                    "index": 1,
                    "relevance_score": 7.641685078851879e-05,
                    "document": None,
                },
                {
                    "index": 3,
                    "relevance_score": 7.621097756782547e-05,
                    "document": None,
                },
            ],
            "meta": {
                "api_version": None,
                "billed_units": None,
                "tokens": None,
                "warnings": None,
            },
        }
    )

    assert_response_shape(r, custom_llm_provider="custom")
