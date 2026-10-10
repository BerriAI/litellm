import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.fireworks_ai.embed.fireworks_ai_transformation import FireworksAIEmbeddingConfig


def test_fireworks_nomic_embedding_supports_and_maps_dimensions():
    config: Final = FireworksAIEmbeddingConfig()
    params: Final = config.map_openai_params(
        non_default_params={"dimensions": 256, "encoding_format": "float"},
        optional_params={},
        model="nomic-ai/nomic-embed-text-v1.5",
    )

    assert config.get_supported_openai_params("nomic-ai/nomic-embed-text-v1.5") == ["dimensions"]
    assert params == {"dimensions": 256}
    assert config.is_fireworks_embedding_model("nomic-ai/nomic-embed-text-v1.5")


def test_fireworks_non_nomic_embedding_drops_dimensions():
    config: Final = FireworksAIEmbeddingConfig()

    assert config.get_supported_openai_params("other/model") == []
    assert config.map_openai_params(
        non_default_params={"dimensions": 256},
        optional_params={},
        model="other/model",
    ) == {}
    assert not config.is_fireworks_embedding_model("other/model")


@respx.mock
def test_fireworks_embeddings_forwards_supported_dimensions():
    route: Final = respx.post("https://fireworks.example/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )
    )

    response: Final = litellm.embedding(
        model="fireworks_ai/nomic-ai/nomic-embed-text-v1.5",
        input=["hello"],
        api_base="https://fireworks.example/v1",
        api_key="test-key",
        dimensions=2,
    )

    assert json.loads(route.calls.last.request.content) == {
        "input": ["hello"],
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dimensions": 2,
    }
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert response.usage.prompt_tokens == 4
    assert response._hidden_params["response_cost"] == pytest.approx(
        4 * litellm.model_cost["fireworks_ai/nomic-ai/nomic-embed-text-v1.5"]["input_cost_per_token"]
    )
