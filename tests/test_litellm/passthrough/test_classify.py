import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.classification import ClassificationResponse


CLASSIFICATION_RESPONSE = {
    "id": "classify-test",
    "object": "list",
    "created": 1,
    "model": "classifier",
    "data": [
        {
            "index": 0,
            "label": "positive",
            "probs": [0.1, 0.9],
            "num_classes": 2,
        }
    ],
    "usage": {
        "prompt_tokens": 4,
        "total_tokens": 4,
    },
}


def test_classify_calls_vllm_classify_and_parses_response() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://localhost:8000/classify")
        assert request.method == "POST"
        assert json.loads(request.read()) == {
            "model": "classifier",
            "input": "LiteLLM is useful",
            "truncate_prompt_tokens": 128,
            "priority": 0,
            "add_special_tokens": True,
            "use_activation": False,
            "cache_salt": "request-salt",
        }
        return httpx.Response(status_code=200, json=CLASSIFICATION_RESPONSE)

    response = litellm.classify(
        model="hosted_vllm/classifier",
        input="LiteLLM is useful",
        api_base="http://localhost:8000",
        truncate_prompt_tokens=128,
        use_activation=False,
        extra_body={"cache_salt": "request-salt"},
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(transport))),
    )

    assert isinstance(response, ClassificationResponse)
    assert response.data[0].label == "positive"
    assert response.data[0].probs == [0.1, 0.9]
    assert response.usage.total_tokens == 4


@pytest.mark.asyncio
async def test_aclassify_accepts_token_ids() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://localhost:8000/classify")
        assert json.loads(request.read()) == {
            "model": "classifier",
            "input": [[101, 2023, 102]],
            "priority": 0,
            "add_special_tokens": False,
        }
        return httpx.Response(
            status_code=200,
            json={
                **CLASSIFICATION_RESPONSE,
                "data": [{"index": 0, "probs": [0.1, 0.9], "num_classes": 2}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as async_client:
        response = await litellm.aclassify(
            model="vllm/classifier",
            input=[[101, 2023, 102]],
            api_base="http://localhost:8000",
            add_special_tokens=False,
            client=AsyncHTTPHandler(client=async_client),
        )

    assert isinstance(response, ClassificationResponse)
    assert response.id == "classify-test"
    assert response.data[0].label is None
