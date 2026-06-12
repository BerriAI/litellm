import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()
from typing import Optional, Dict

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import os

import pytest

import litellm
from litellm.types.rerank import RerankResponse
from litellm.integrations.custom_logger import CustomLogger


def assert_response_shape(response, custom_llm_provider):
    expected_response_shape = {"id": str, "results": list, "meta": dict}

    expected_results_shape = {
        "index": int,
        "relevance_score": float,
        "document": Optional[Dict[str, str]],
    }

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
        if "document" in result:
            assert isinstance(result["document"], Dict)
            assert isinstance(result["document"]["text"], str)
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
@pytest.mark.flaky(retries=3, delay=1)
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
@pytest.mark.skip(reason="Skipping test due to 503 Service Temporarily Unavailable")
async def test_basic_rerank_together_ai(sync_mode):
    try:
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
    except Exception as e:
        if "Service unavailable" in str(e):
            pytest.skip("Skipping test due to 503 Service Temporarily Unavailable")
        raise e


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
@pytest.mark.flaky(retries=3, delay=1)
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

    await asyncio.sleep(8)

    print("async re rank response: ", response)
    assert custom_logger.kwargs is not None
    assert custom_logger.kwargs.get("response_cost") > 0.0
    assert custom_logger.response_obj is not None
    assert custom_logger.response_obj.results is not None


@pytest.mark.asyncio()
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "top_n_1, top_n_2, expect_cache_hit",
    [
        (3, 3, True),
        (3, None, False),
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
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
                {"index": 2, "relevance_score": 0.9958819150924683},
                {"index": 0, "relevance_score": 0.001293411129154265},
                {
                    "index": 1,
                    "relevance_score": 7.641685078851879e-05,
                },
                {
                    "index": 3,
                    "relevance_score": 7.621097756782547e-05,
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


@pytest.mark.flaky(retries=3, delay=1)
def test_rerank_cohere_api():
    response = litellm.rerank(
        model="cohere/rerank-english-v3.0",
        query="hello",
        documents=["hello", "world"],
        return_documents=True,
        top_n=3,
    )
    print("rerank response", response)
    assert response.results[0]["document"] is not None
    assert response.results[0]["document"]["text"] is not None
    assert response.results[0]["document"]["text"] == "hello"
    assert response.results[1]["document"]["text"] == "world"


