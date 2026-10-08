import json
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest

from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

RUNTIME_URL_PART: Final = "bedrock-runtime.us-east-1.amazonaws.com"
MANTLE_URL_PART: Final = "bedrock-mantle.us-east-1.api.aws"
UNSUPPORTED: Final = {"message": "The provided model doesn't support counting tokens."}
DEPLOYMENT: Final = {
    "litellm_params": {
        "aws_access_key_id": "AKIATESTACCESSKEY",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
}
MESSAGES: Final = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]


def _client(runtime: tuple[int, dict[str, object]], mantle: tuple[int, dict[str, object]]) -> AsyncHTTPHandler:
    async def _post(url: str, **kwargs: object) -> httpx.Response:  # kwargs-ok: mirrors AsyncHTTPHandler.post
        status_code, body = runtime if RUNTIME_URL_PART in url else mantle
        return httpx.Response(status_code, json=body, request=httpx.Request("POST", url))

    client: Final = AsyncMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(side_effect=_post)
    return client


def _posted_hosts(client: AsyncHTTPHandler) -> tuple[str, ...]:
    return tuple(httpx.URL(call.args[0]).host for call in client.post.call_args_list)


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_claude_model_bedrock_runtime_cannot_count_is_counted_on_mantle() -> None:
    client: Final = _client(runtime=(400, UNSUPPORTED), mantle=(200, {"input_tokens": 2177}))

    result: Final = await BedrockTokenCounter(client=client).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        system="You are a terse assistant.",
    )

    assert result is not None
    assert result.error is False
    assert result.total_tokens == 2177
    assert result.tokenizer_type == "bedrock_mantle_api"
    assert result.original_response == {"input_tokens": 2177}
    assert _posted_hosts(client) == (RUNTIME_URL_PART, MANTLE_URL_PART)
    mantle_body: Final = json.loads(client.post.call_args_list[1].kwargs["data"])
    assert mantle_body["model"] == "anthropic.claude-opus-4-8"
    assert mantle_body["system"] == "You are a terse assistant."


@pytest.mark.asyncio
async def test_bedrock_runtime_count_is_kept_when_it_answers() -> None:
    client: Final = _client(runtime=(200, {"inputTokens": 1353}), mantle=(200, {"input_tokens": 1336}))

    result: Final = await BedrockTokenCounter(client=client).count_tokens(
        model_to_use="global.anthropic.claude-sonnet-4-6",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-sonnet-4-6",
    )

    assert result is not None
    assert result.total_tokens == 1353
    assert result.tokenizer_type == "bedrock_api"
    assert _posted_hosts(client) == (RUNTIME_URL_PART,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_to_use", "runtime"),
    (
        ("global.anthropic.claude-opus-4-8", (403, {"Message": "not authorized to perform: bedrock:CountTokens"})),
        ("amazon.nova-pro-v1:0", (400, UNSUPPORTED)),
    ),
)
async def test_other_bedrock_runtime_failures_are_not_retried_on_mantle(
    model_to_use: str, runtime: tuple[int, dict[str, object]]
) -> None:
    client: Final = _client(runtime=runtime, mantle=(200, {"input_tokens": 2177}))

    result: Final = await BedrockTokenCounter(client=client).count_tokens(
        model_to_use=model_to_use,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model=model_to_use,
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == runtime[0]
    assert result.tokenizer_type == "bedrock_api"
    assert _posted_hosts(client) == (RUNTIME_URL_PART,)


@pytest.mark.asyncio
async def test_mantle_failure_is_reported_with_its_status() -> None:
    client: Final = _client(
        runtime=(400, UNSUPPORTED),
        mantle=(404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}),
    )

    result: Final = await BedrockTokenCounter(client=client).count_tokens(
        model_to_use="global.anthropic.claude-sonnet-5-5",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-sonnet-5-5",
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 404
    assert result.tokenizer_type == "bedrock_mantle_api"
    assert "does not exist" in (result.error_message or "")
    assert _posted_hosts(client) == (RUNTIME_URL_PART, MANTLE_URL_PART)
