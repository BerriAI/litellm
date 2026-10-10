import json
from typing import Final

import httpx
import respx

import litellm


def test_mistral_embedding_uses_mistral_embeddings_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.mistral.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "embd-migration",
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "mistral-embed",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )

    response: Final = litellm.embedding(
        model="mistral/mistral-embed",
        input=["hello"],
        api_key="test-mistral-key",
    )

    assert str(route.calls.last.request.url) == "https://api.mistral.ai/v1/embeddings"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-mistral-key"
    assert json.loads(route.calls.last.request.read()) == {"model": "mistral-embed", "input": ["hello"]}
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (2, 2)
