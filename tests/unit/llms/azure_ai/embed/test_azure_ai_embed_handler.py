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

    assert route.called
    assert response.data is not None
    assert len(response.data) == 1
