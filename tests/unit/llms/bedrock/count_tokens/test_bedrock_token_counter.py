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
    def __init__(
        self,
        runtime: tuple[int, Mapping[str, object]],
        mantle: tuple[int, Mapping[str, object]],
        mantle_models: Mapping[str, tuple[int, Mapping[str, object]]] | None = None,
    ) -> None:
        self.runtime: Final = runtime
        self.mantle: Final = mantle
        self.mantle_models: Final = mantle_models or {}
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        status_code, body = (
            self.runtime
            if request.url.host == RUNTIME_HOST
            else self.mantle_models.get(str(_JSON_BODY.validate_json(request.content).get("model")), self.mantle)
        )
        return httpx.Response(status_code, json=dict(body), request=request)

    def posted_hosts(self) -> tuple[str, ...]:
        return tuple(request.url.host for request in self.requests)


def _counter(bedrock: _Bedrock, model_catalog: Mapping[str, Mapping[str, object]] | None = None) -> BedrockTokenCounter:
    return BedrockTokenCounter(
        client=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock)), model_catalog=model_catalog or {}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("native_status", [200, 403, 404, 503])
async def test_configured_counter_alias_only_runs_when_native_mantle_model_is_missing(native_status: int) -> None:
    requested: Final = "anthropic.claude-requested"
    fallback: Final = "anthropic.claude-counter"
    native_count: Final = 251
    fallback_count: Final = 249
    native_reply: Final = {"input_tokens": native_count} if native_status == 200 else {"message": "Unavailable"}
    provider: Final = _Bedrock(
        runtime=(400, UNSUPPORTED),
        mantle=(native_status, native_reply),
        mantle_models={fallback: (200, {"input_tokens": fallback_count})},
    )
    counter: Final = _counter(provider, {requested: {"token_count_fallback_model": fallback}})
    result: Final = await counter.count_tokens(
        model_to_use="global." + requested,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        system="Synthetic policy",
        tools=[
            {
                "name": "lookup",
                "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
            }
        ],
    )
    assert result is not None
    if native_status == 404:
        assert result.total_tokens == fallback_count
        assert result.model_used == fallback
        assert result.tokenizer_type == "bedrock_mantle_api_fallback"
        assert len(provider.requests) == 3
        body: Final = _JSON_BODY.validate_json(provider.requests[-1].content)
        assert body["model"] == fallback
        assert body["system"] == "Synthetic policy"
        returned_tools: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_python(body["tools"])
        assert returned_tools[0]["name"] == "lookup"
    elif native_status == 200:
        assert result.total_tokens == native_count
        assert result.model_used == "global." + requested
        assert result.tokenizer_type == "bedrock_mantle_api"
        assert len(provider.requests) == 2
    else:
        assert result.error is True
        assert result.status_code == native_status
        assert len(provider.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [None, "", 123, "anthropic.claude-requested"])
async def test_invalid_or_self_counter_alias_does_not_trigger_an_extra_call(configured: str | int | None) -> None:
    requested: Final = "anthropic.claude-requested"
    provider: Final = _Bedrock(runtime=(400, UNSUPPORTED), mantle=(404, {"message": "Unavailable"}))
    counter: Final = _counter(provider, {requested: {"token_count_fallback_model": configured}})
    result: Final = await counter.count_tokens(
        model_to_use="global." + requested,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
    )
    assert result is not None
    assert result.error is True
    assert result.status_code == 404
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_configured_counter_alias_failure_does_not_chain_or_claim_success() -> None:
    requested: Final = "anthropic.claude-requested"
    fallback: Final = "anthropic.claude-counter"
    provider: Final = _Bedrock(runtime=(400, UNSUPPORTED), mantle=(404, {"message": "Unavailable"}))
    counter: Final = _counter(
        provider,
        {
            requested: {"token_count_fallback_model": fallback},
            fallback: {"token_count_fallback_model": requested},
        },
    )
    result: Final = await counter.count_tokens(
        model_to_use="global." + requested,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
    )
    assert result is not None
    assert result.error is True
    assert result.status_code == 404
    assert result.model_used == fallback
    assert len(provider.requests) == 3


@pytest.fixture(autouse=True)
def _sigv4_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("BEDROCK_MANTLE_API_BASE", raising=False)


@pytest.mark.asyncio
async def test_bedrock_token_counter_skips_when_framing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setitem(litellm.model_cost, "global.moonshotai.kimi-k3", {"token_count_framing": 100})
    bedrock: Final = _Bedrock(runtime=(400, UNSUPPORTED), mantle=(200, {"input_tokens": 2177}))
    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.moonshotai.kimi-k3",
        messages=MESSAGES,
        contents=None,
    )
    assert result is not None
    assert result.error is True
    assert result.status_code == 501
    assert result.error_message == "Use local framed token count"
    assert bedrock.posted_hosts() == ()


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
