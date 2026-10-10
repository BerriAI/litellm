import asyncio
import uuid
from typing import Final

import openai
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value
from tests.integration._support.database import read_rows


_ENDPOINTS: Final = (
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "unknown model"}]}),
    ("/v1/completions", {"prompt": "unknown model"}),
    ("/v1/embeddings", {"input": "unknown model"}),
    ("/v1/images/generations", {"prompt": "unknown model"}),
)


@pytest.mark.parametrize(("path", "body"), _ENDPOINTS, ids=("chat", "completion", "embedding", "image"))
def test_unknown_model_is_rejected_before_provider_request(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    model: Final = f"migration-unknown-model-{uuid.uuid4().hex}"
    response: Final = gateway.request("POST", path, {"model": model, **body})

    assert response.status_code == 400, response.text
    assert model in response.text


@pytest.mark.parametrize(("path", "body"), _ENDPOINTS, ids=("chat", "completion", "embedding", "image"))
def test_missing_model_is_rejected_on_openai_endpoints(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    response: Final = gateway.request("POST", path, body)

    assert response.status_code == 400, response.text
    assert "model" in response.text.lower()


def test_unknown_model_is_a_bad_request_for_async_openai_client(gateway: Gateway) -> None:
    model: Final = f"migration-unknown-model-{uuid.uuid4().hex}"

    async def call() -> openai.BadRequestError:
        client: Final = openai.AsyncOpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=gateway.key)
        try:
            with pytest.raises(openai.BadRequestError) as raised:
                await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "unknown model"}],
                )
            return raised.value
        finally:
            await client.close()

    error: Final = asyncio.run(call())
    assert error.status_code == 400
    assert model in str(error)


def test_unknown_model_failure_is_written_to_spend_logs(gateway: Gateway) -> None:
    model: Final = f"migration-unknown-model-{uuid.uuid4().hex}"
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "unknown model"}]},
    )
    assert response.status_code == 400, response.text

    call_id: Final = response.headers["x-litellm-call-id"]
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status, model, metadata->\'error_information\' AS error_information '
            'FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (call_id,),
        ),
        lambda values: len(values) == 1,
        seconds=30,
    )
    row: Final = rows[0]
    assert row["status"] == "failure", row
    assert row["model"] == model, row
    assert object_value(row["error_information"])["error_code"] == "400"
