import json
from pathlib import Path
from typing import Final, Literal

import pytest
from integration._support.client import JSON_OBJECT
from integration._support.wire import Reply, wire_server

import litellm
from litellm.types.utils import EmbeddingResponse

_ENV_KEY: Final = "synthetic-scaleway-env-secret"
_VECTOR: Final = [0.25, 0.5, 0.75]
_PROMPT_TOKENS: Final = 7
# The cost map prices Scaleway embeddings under the vendor-namespaced id
_COST_MAP_KEY: Final = "scaleway/qwen/qwen3-embedding-8b"

Mode = Literal["sync", "async"]


def _reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": _VECTOR}],
                "model": "qwen3-embedding-8b",
                "usage": {"prompt_tokens": _PROMPT_TOKENS, "total_tokens": _PROMPT_TOKENS},
            }
        ).encode()
    )


async def _embed(mode: Mode, **kwargs: object) -> EmbeddingResponse:
    if mode == "sync":
        return litellm.embedding(model="scaleway/qwen3-embedding-8b", **kwargs)
    return await litellm.aembedding(model="scaleway/qwen3-embedding-8b", **kwargs)


@pytest.mark.parametrize("mode", ("sync", "async"))
async def test_regression_scaleway_embedding_reaches_the_embeddings_endpoint_with_the_scw_secret_key(
    monkeypatch: pytest.MonkeyPatch, mode: Mode
) -> None:
    monkeypatch.setenv("SCW_SECRET_KEY", _ENV_KEY)
    with wire_server(lambda _: _reply()) as wire:
        response: Final = await _embed(
            mode, input=["hello"], api_base=f"{wire.url}/v1", dimensions=256, encoding_format="float", user="caller-7"
        )
        received: Final = wire.drain()
    assert [row["embedding"] for row in response.data] == [_VECTOR], response
    input_cost_per_token: Final = json.loads(Path("model_prices_and_context_window.json").read_text())[_COST_MAP_KEY][
        "input_cost_per_token"
    ]
    assert response._hidden_params["response_cost"] == pytest.approx(_PROMPT_TOKENS * input_cost_per_token), (
        response._hidden_params
    )
    assert [
        (request.method, request.target, request.headers["authorization"], JSON_OBJECT.validate_json(request.body))
        for request in received
    ] == [
        (
            "POST",
            "/v1/embeddings",
            f"Bearer {_ENV_KEY}",
            {
                "model": "qwen3-embedding-8b",
                "input": ["hello"],
                "dimensions": 256,
                "encoding_format": "float",
                "user": "caller-7",
            },
        )
    ]
