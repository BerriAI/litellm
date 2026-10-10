import pytest
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value


def _restricted_key(gateway: Gateway, scenario: Scenario) -> str:
    return scenario.key(
        models=[
            "gpt-4",
            "text-embedding-ada-002",
            "dall-e-2",
            "fake-openai-endpoint-2",
            "mistral-embed",
            "non-existent-model",
        ]
    )


@pytest.mark.asyncio
async def test_async_chat_completion_bad_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        async with AsyncOpenAI(
            api_key=_restricted_key(gateway, scenario), base_url=str(gateway.client.base_url)
        ) as client:
            with pytest.raises(BadRequestError):
                await client.chat.completions.create(
                    model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
                )


def test_chat_completion_bad_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with OpenAI(api_key=_restricted_key(gateway, scenario), base_url=str(gateway.client.base_url)) as client:
            with pytest.raises(BadRequestError):
                client.chat.completions.create(
                    model="non-existent-model", messages=[{"role": "user", "content": "Hello!"}]
                )


def test_completion_bad_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with OpenAI(api_key=_restricted_key(gateway, scenario), base_url=str(gateway.client.base_url)) as client:
            with pytest.raises(BadRequestError):
                client.completions.create(model="non-existent-model", prompt="Hello!")


def test_embeddings_bad_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with OpenAI(api_key=_restricted_key(gateway, scenario), base_url=str(gateway.client.base_url)) as client:
            with pytest.raises(BadRequestError):
                client.embeddings.create(model="non-existent-model", input="Hello world")


def test_images_bad_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        with OpenAI(api_key=_restricted_key(gateway, scenario), base_url=str(gateway.client.base_url)) as client:
            with pytest.raises(BadRequestError):
                client.images.generate(model="non-existent-model", prompt="A cute baby sea otter")


@pytest.mark.parametrize(
    ("path", "body"),
    (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello!"}]}),
        ("/v1/completions", {"prompt": "Hello!"}),
        ("/v1/embeddings", {"input": "Hello world"}),
        ("/v1/images/generations", {"prompt": "A cute baby sea otter"}),
    ),
    ids=("chat", "completions", "embeddings", "images"),
)
def test_missing_model_parameter_returns_bad_request(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    response = gateway.request("POST", path, body)
    assert response.status_code == 400, response.text
    error = object_value(response.json())["error"]
    assert isinstance(error, dict)
    assert isinstance(error.get("message"), str) and error["message"]


def test_chat_completion_bad_model_with_spend_logs(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: str = _restricted_key(gateway, scenario)
        response = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": "non-existent-model", "messages": [{"role": "user", "content": "Hello!"}]},
            key=key,
        )
        assert response.status_code == 400, response.text
        call_id: str | None = response.headers.get("x-litellm-call-id")
        assert call_id is not None
        logs_response = eventually(
            lambda: gateway.request("GET", "/spend/logs", params={"request_id": call_id}),
            lambda candidate: candidate.status_code == 200 and bool(candidate.json()),
            seconds=70,
        )
        logs: list[JsonValue] = list_value(logs_response.json())
        assert len(logs) > 0
        log: dict[str, JsonValue] = object_value(logs[0])
        assert log["request_id"] == call_id
        assert log["model"] == "unknown-model"
        assert log["model_group"] in ("", "non-existent-model")
        assert log["spend"] == 0.0
        assert log["total_tokens"] == 0
        assert log["prompt_tokens"] == 0
        assert log["completion_tokens"] == 0

        metadata: dict[str, JsonValue] = object_value(log["metadata"])
        assert metadata["status"] == "failure"
        assert "user_api_key" in metadata
        error_info: dict[str, JsonValue] = object_value(metadata["error_information"])
        assert "traceback" in error_info
        assert error_info["error_code"] == "400"
        assert error_info["error_class"] in ("ProxyModelNotFoundError", "BadRequestError")
        assert "non-existent-model" not in string_value(error_info["error_message"])
        assert "/chat/completions: Invalid model name passed in" in string_value(error_info["error_message"])
        assert log["cache_hit"] == "False"
        assert log["response"] == {}
