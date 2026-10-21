import json

import httpx
import pytest
import respx

from litellm import embedding
from litellm.llms.azure_ai.embed.handler import _foundry_models_route_base

EMBEDDING_PAYLOAD = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (
            "https://my-foundry.services.ai.azure.com",
            "https://my-foundry.services.ai.azure.com/models",
        ),
        (
            "https://my-foundry.services.ai.azure.com/",
            "https://my-foundry.services.ai.azure.com/models",
        ),
        (
            "https://my-foundry.services.ai.azure.com?api-version=2024-05-01-preview",
            "https://my-foundry.services.ai.azure.com/models?api-version=2024-05-01-preview",
        ),
        (
            "https://my-foundry.services.ai.azure.com/models",
            "https://my-foundry.services.ai.azure.com/models",
        ),
        (
            "https://my-foundry.services.ai.azure.com/openai/deployments/text-embedding-3-small",
            "https://my-foundry.services.ai.azure.com/openai/deployments/text-embedding-3-small",
        ),
        (
            "https://my-resource.openai.azure.com",
            "https://my-resource.openai.azure.com",
        ),
        (
            "https://Mistral-serverless.eastus2.models.ai.azure.com",
            "https://Mistral-serverless.eastus2.models.ai.azure.com",
        ),
        (None, None),
    ],
)
def test_foundry_models_route_base(api_base, expected):
    assert _foundry_models_route_base(api_base) == expected


@respx.mock
def test_azure_ai_embedding_calls_foundry_models_route():
    route = respx.post("https://my-foundry.services.ai.azure.com/models/embeddings").mock(
        return_value=httpx.Response(200, json=EMBEDDING_PAYLOAD)
    )

    response = embedding(
        model="azure_ai/text-embedding-3-small",
        input=["hello world"],
        api_base="https://my-foundry.services.ai.azure.com",
        api_key="fake-key",
    )

    assert route.call_count == 1
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2]]


@respx.mock
def test_azure_ai_cohere_image_embedding_calls_image_route():
    route = respx.post("https://my-foundry.services.ai.azure.com/models/images/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": "Cohere-embed-v3-multilingual",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )

    response = embedding(
        model="azure_ai/Cohere-embed-v3-multilingual",
        input=["data:image/png;base64,aGVsbG8="],
        api_base="https://my-foundry.services.ai.azure.com",
        api_key="fake-key",
    )

    assert json.loads(route.calls.last.request.read()) == {"input": [{"image": "data:image/png;base64,aGVsbG8="}]}
    assert set(dict(response).keys()) == {"object", "data", "model", "usage"}
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (2, 2)
