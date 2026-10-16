from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value
from openai.types.moderation_create_response import ModerationCreateResponse


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=10, trust_env=False) as client:
        yield client


def test_chat_completion_shape(upstream: httpx.Client) -> None:
    response: Final = upstream.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    message: Final = object_value(object_value(list_value(body["choices"])[0])["message"])
    assert message == {
        "role": "assistant",
        "content": "Hello! This is a mock response from the fake OpenAI endpoint.",
    }
    assert body["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 20,
        "total_tokens": 40,
    }


def test_moderations_route_parses_as_an_openai_response(upstream: httpx.Client) -> None:
    response: Final = upstream.post(
        "/v1/moderations",
        json={
            "input": ["I want to harm someone", "hello"],
            "model": "omni-moderation-latest",
        },
    )

    assert response.status_code == 200, response.text
    parsed: Final = ModerationCreateResponse.model_validate_json(response.content)
    assert parsed.model == "omni-moderation-latest"
    assert len(parsed.results) == 2
    assert parsed.results[0].categories.violence is False


def test_triton_embeddings_route(upstream: httpx.Client) -> None:
    response: Final = upstream.post("/triton/embeddings", json={"inputs": []})

    assert response.status_code == 200, response.text
    output: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["outputs"])[0])
    assert output["shape"] == [1, 2]
    assert output["data"] == [0.1, 0.2]


def test_slow_model_blocks_past_client_timeout(upstream: httpx.Client) -> None:
    with pytest.raises(httpx.TimeoutException):
        upstream.post(
            "/v1/chat/completions",
            json={
                "model": "slow-endpoint",
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=0.5,
        )
