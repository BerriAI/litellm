from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

RUNTIME_HOST: Final = "bedrock-runtime.us-east-1.amazonaws.com"
MANTLE_HOST: Final = "bedrock-mantle.us-east-1.api.aws"
UNSUPPORTED: Final = {"message": "The provided model doesn't support counting tokens."}
DEPLOYMENT: Final = {
    "litellm_params": {
        "aws_access_key_id": "AKIATESTACCESSKEY",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
}
MESSAGES: Final = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]
_JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])


class _Bedrock:
    def __init__(self, runtime: tuple[int, Mapping[str, object]], mantle: tuple[int, Mapping[str, object]]) -> None:
        self.runtime: Final = runtime
        self.mantle: Final = mantle
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        status_code, body = self.runtime if request.url.host == RUNTIME_HOST else self.mantle
        return httpx.Response(status_code, json=dict(body), request=request)

    def posted_hosts(self) -> tuple[str, ...]:
        return tuple(request.url.host for request in self.requests)


def _counter(bedrock: _Bedrock) -> BedrockTokenCounter:
    return BedrockTokenCounter(client=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock)))


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_claude_model_bedrock_runtime_cannot_count_is_counted_on_mantle() -> None:
    bedrock: Final = _Bedrock(runtime=(400, UNSUPPORTED), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
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
    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)
    mantle_body: Final = _JSON_BODY.validate_json(bedrock.requests[1].content)
    assert mantle_body["model"] == "anthropic.claude-opus-4-8"
    assert mantle_body["system"] == "You are a terse assistant."


@pytest.mark.asyncio
async def test_bedrock_runtime_count_is_kept_when_it_answers() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 1353}), mantle=(200, {"input_tokens": 1336}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-sonnet-4-6",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-sonnet-4-6",
    )

    assert result is not None
    assert result.total_tokens == 1353
    assert result.tokenizer_type == "bedrock_api"
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_to_use", "runtime"),
    (
        ("global.anthropic.claude-opus-4-8", (403, {"Message": "not authorized to perform: bedrock:CountTokens"})),
        ("amazon.nova-pro-v1:0", (400, UNSUPPORTED)),
    ),
)
async def test_other_bedrock_runtime_failures_are_not_retried_on_mantle(
    model_to_use: str, runtime: tuple[int, Mapping[str, object]]
) -> None:
    bedrock: Final = _Bedrock(runtime=runtime, mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
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
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)


@pytest.mark.asyncio
async def test_mantle_failure_is_reported_with_its_status() -> None:
    bedrock: Final = _Bedrock(
        runtime=(400, UNSUPPORTED),
        mantle=(404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}),
    )

    result: Final = await _counter(bedrock).count_tokens(
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
    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)
