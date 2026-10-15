import json
from collections.abc import Mapping
from typing import Final

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
REQUEST: Final[dict[str, object]] = {
    "model": "global.anthropic.claude-opus-4-8",
    "messages": [{"role": "user", "content": "The quick brown fox"}],
    "system": "You are a terse assistant.",
    "tools": [{"name": "get_weather", "input_schema": {"type": "object", "properties": {}}}],
}


class _MantleEndpoint:
    def __init__(self, status_code: int, body: Mapping[str, object]) -> None:
        self.status_code: Final = status_code
        self.body: Final = body
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(self.status_code, json=dict(self.body), request=request)

    def only_request(self) -> httpx.Request:
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


def _client(endpoint: _MantleEndpoint) -> AsyncHTTPHandler:
    return AsyncHTTPHandler(transport=httpx.MockTransport(endpoint))


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_counts_on_mantle_with_the_base_model_and_the_deployment_credentials() -> None:
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 2177})

    result: Final = await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=_client(mantle),
    )

    assert result == {"input_tokens": 2177}
    posted: Final = mantle.only_request()
    assert str(posted.url) == MANTLE_COUNT_URL
    assert posted.headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert "us-east-1/bedrock/aws4_request" in posted.headers["Authorization"]
    assert posted.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(posted.content) == {
        "model": "anthropic.claude-opus-4-8",
        "messages": REQUEST["messages"],
        "system": REQUEST["system"],
        "tools": REQUEST["tools"],
    }


@pytest.mark.asyncio
async def test_bedrock_mantle_api_base_env_names_the_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BEDROCK_MANTLE_API_BASE", "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws")
    mantle: Final = _MantleEndpoint(200, {"input_tokens": 3})

    await BedrockMantleCountTokensHandler().handle_count_tokens_request(
        request_data=dict(REQUEST),
        litellm_params=dict(LITELLM_PARAMS),
        resolved_model="anthropic.claude-opus-4-8",
        client=_client(mantle),
    )

    assert (
        str(mantle.only_request().url)
        == "https://vpce-abc.bedrock-mantle.us-east-1.vpce.api.aws/anthropic/v1/messages/count_tokens"
    )


@pytest.mark.asyncio
async def test_non_200_answers_raise_bedrock_error_with_mantle_status() -> None:
    mantle: Final = _MantleEndpoint(
        404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}
    )

    with pytest.raises(BedrockError) as raised:
        await BedrockMantleCountTokensHandler().handle_count_tokens_request(
            request_data=dict(REQUEST),
            litellm_params=dict(LITELLM_PARAMS),
            resolved_model="anthropic.claude-sonnet-5-5",
            client=_client(mantle),
        )

    assert raised.value.status_code == 404
    assert "does not exist" in raised.value.message
