import json
from typing import Final

import httpx
import respx

import litellm


def test_together_ai_embedding_uses_together_embeddings_endpoint(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://api.together.ai/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.3, 0.4]}],
                "model": "BAAI/bge-large-en-v1.5",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )

    response: Final = litellm.embedding(
        model="together_ai/BAAI/bge-large-en-v1.5",
        input=["hello"],
        api_key="test-together-key",
    )

    assert str(route.calls.last.request.url) == "https://api.together.ai/v1/embeddings"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-together-key"
    assert json.loads(route.calls.last.request.read()) == {
        "model": "BAAI/bge-large-en-v1.5",
        "input": ["hello"],
    }
    assert response.data[0]["embedding"] == [0.3, 0.4]
    assert (response.usage.prompt_tokens, response.usage.total_tokens) == (1, 1)
