from __future__ import annotations

import httpx
import pytest
from integration._support.client import Gateway, JSON_OBJECT, list_value, object_value
from openai.types.moderation_create_response import ModerationCreateResponse


def test_chat_completion_shape(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(model="openai/fake-model")
        body = gateway.chat(model, text="hi")

    choices = list_value(body["choices"])
    message = object_value(object_value(choices[0])["message"])
    usage = object_value(body["usage"])
    assert message["content"] == "Hello! This is a mock response from the fake OpenAI endpoint."
    assert usage == {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}


def test_moderations_route_parses_as_an_openai_response(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(model="openai/omni-moderation-latest")
        response = gateway.request(
            "POST",
            "/v1/moderations",
            {"input": ["I want to harm someone", "hello"], "model": model},
        )

    assert response.status_code == 200, response.text
    parsed = ModerationCreateResponse.model_validate_json(response.content)
    assert len(parsed.results) == 2
    assert parsed.results[0].categories.violence is False


def test_slow_model_blocks_past_client_timeout(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(model="openai/slow-endpoint")
        with pytest.raises(httpx.TimeoutException):
            gateway.client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {gateway.key}"},
                timeout=0.5,
            )


def test_triton_embeddings_route(gateway: Gateway) -> None:
    with httpx.Client(base_url=gateway.upstream_url, timeout=10, trust_env=False) as client:
        response = client.post("/triton/embeddings", json={"inputs": []})

    assert response.status_code == 200, response.text
    body = JSON_OBJECT.validate_json(response.content)
    outputs = list_value(body["outputs"])
    output = object_value(outputs[0])
    assert output["shape"] == [1, 2]
    assert output["data"] == [0.1, 0.2]


def test_lm_studio_completion(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(
            model="lm_studio/typhoon2-quen2.5-7b-instruct",
            api_base=f"{gateway.upstream_url}/v1",
            api_key="synthetic-lm-studio-key",
        )
        response = gateway.chat(model, text="What's the weather like in San Francisco?")

    choices = list_value(response["choices"])
    message = object_value(object_value(choices[0])["message"])
    assert message["content"] == "Hello! This is a mock response from the fake OpenAI endpoint."


def test_openai_embedding_timeouts(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model = scenario.model(model="openai/slow-endpoint")
        with pytest.raises(httpx.TimeoutException):
            gateway.client.post(
                "/v1/embeddings",
                json={"model": model, "input": ["good morning"]},
                headers={"Authorization": f"Bearer {gateway.key}"},
                timeout=0.5,
            )
