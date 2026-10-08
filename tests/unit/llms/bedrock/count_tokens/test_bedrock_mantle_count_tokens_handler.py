import json
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.count_tokens.mantle_handler import BedrockMantleCountTokensHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

MANTLE_COUNT_URL: Final = "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages/count_tokens"
LITELLM_PARAMS: Final = {
    "aws_access_key_id": "AKIATESTACCESSKEY",
    "aws_secret_access_key": "test-secret",
    "aws_region_name": "us-east-1",
}
REQUEST: Final = {
    "model": "global.anthropic.claude-opus-4-8",
    "messages": [{"role": "user", "content": "The quick brown fox"}],
    "system": "You are a terse assistant.",
    "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
}


def _client(response: httpx.Response) -> AsyncHTTPHandler:
    client: Final = AsyncMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(return_value=response)
    return client


def _response(status_code: int, body: dict[str, object]) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("POST", MANTLE_COUNT_URL))


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_counts_on_mantle_with_the_base_model_and_the_deployment_credentials() -> None:
    client: Final = _client(_response(200, {"input_tokens": 2177}))

    result: Final = await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=client,
    )

    assert result == {"input_tokens": 2177}
    assert client.post.call_args.args == (MANTLE_COUNT_URL,)
    posted_headers: Final = client.post.call_args.kwargs["headers"]
    assert posted_headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "us-east-1/bedrock/aws4_request" in posted_headers["Authorization"]
    assert posted_headers["anthropic-version"] == "2023-06-01"
    assert json.loads(client.post.call_args.kwargs["data"]) == {
        "model": "anthropic.claude-opus-4-8",
        "messages": REQUEST["messages"],
        "system": REQUEST["system"],
        "tools": REQUEST["tools"],
    }


@pytest.mark.asyncio
async def test_bedrock_mantle_api_base_env_names_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws")
    client: Final = _client(_response(200, {"input_tokens": 3}))

    await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=client,
    )

    assert client.post.call_args.args == (
        "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws/anthropic/v1/messages/count_tokens",
    )


@pytest.mark.asyncio
async def test_non_200_answers_raise_bedrock_error_with_mantle_status() -> None:
    client: Final = _client(
        _response(404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}})
    )

    with pytest.raises(BedrockError) as raised:
        await BedrockMantleCountTokensHandler().handle_count_tokens_request(
            request_data=dict(REQUEST),
            litellm_params=dict(LITELLM_PARAMS),
            resolved_model="anthropic.claude-sonnet-5-5",
            client=client,
        )

    assert raised.value.status_code == 404
    assert "does not exist" in raised.value.message
