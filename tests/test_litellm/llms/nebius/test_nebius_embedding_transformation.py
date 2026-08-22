import json

import pytest

import litellm

TOKEN_FACTORY_API_BASE = "https://api.tokenfactory.nebius.com/v1"


@pytest.mark.respx()
def test_nebius_embedding_targets_token_factory(respx_mock, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post(f"{TOKEN_FACTORY_API_BASE}/embeddings").respond(
        json={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "Qwen/Qwen3-Embedding-8B",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
        status_code=200,
    )

    response = litellm.embedding(
        model="nebius/Qwen/Qwen3-Embedding-8B",
        input=["hello from LiteLLM"],
        api_key="test-key",
    )

    assert route.called
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    assert (
        json.loads(route.calls[0].request.content)["model"] == "Qwen/Qwen3-Embedding-8B"
    )
