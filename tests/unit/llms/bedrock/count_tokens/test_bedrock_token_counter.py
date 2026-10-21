import base64
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
        ("global.anthropic.claude-opus-4-8", (401, {"Message": "not authorized to perform: bedrock:CountTokens"})),
        ("global.anthropic.claude-opus-4-8", (403, {"Message": "not authorized to perform: bedrock:CountTokens"})),
        ("amazon.nova-pro-v1:0", (400, UNSUPPORTED)),
        ("amazon.nova-pro-v1:0", (404, UNSUPPORTED)),
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
        runtime=(404, UNSUPPORTED),
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


@pytest.mark.asyncio
async def test_claude_runtime_404_retries_with_system_and_tools_on_mantle() -> None:
    bedrock: Final = _Bedrock(runtime=(404, UNSUPPORTED), mantle=(200, {"input_tokens": 2177}))
    tools: Final = [
        {
            "name": "lookup",
            "description": "Find a record",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        system="You are a terse assistant.",
        tools=tools,
    )

    assert result is not None
    assert result.error is False
    assert result.total_tokens == 2177
    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)
    mantle_body: Final = _JSON_BODY.validate_json(bedrock.requests[1].content)
    assert mantle_body == {
        "model": "anthropic.claude-opus-4-8",
        "messages": MESSAGES,
        "system": "You are a terse assistant.",
        "tools": tools,
    }


def _invoke_body(request: httpx.Request) -> dict[str, JsonValue]:
    invoke_model: Final = _JSON_BODY.validate_json(request.content)["input"]
    assert isinstance(invoke_model, dict)
    encoded: Final = invoke_model["invokeModel"]
    assert isinstance(encoded, dict) and isinstance(encoded["body"], str)
    return _JSON_BODY.validate_json(base64.b64decode(encoded["body"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_params",
    [
        {"tool_choice": "required"},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": {"type": "object", "properties": {"city": {"type": "string"}}}},
            }
        },
        {"reasoning_effort": "low"},
    ],
)
async def test_runtime_count_leaves_out_the_options_only_mantle_counts(chat_params: dict[str, JsonValue]) -> None:
    runtime_counts: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))
    mantle_counts: Final = _Bedrock(runtime=(404, UNSUPPORTED), mantle=(200, {"input_tokens": 2177}))
    request: Final = {
        "model_to_use": "global.anthropic.claude-opus-4-8",
        "messages": [{"role": "user", "content": "Weather in SF?"}],
        "contents": None,
        "deployment": DEPLOYMENT,
        "request_model": "claude-opus-4-8",
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "messages_format": "openai",
        "chat_params": chat_params,
    }

    await _counter(runtime_counts).count_tokens(**request)
    await _counter(mantle_counts).count_tokens(**request)

    assert runtime_counts.posted_hosts() == (RUNTIME_HOST,)
    assert mantle_counts.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)
    options: Final = ("thinking", "tool_choice", "output_config")
    invoke_body: Final = _invoke_body(runtime_counts.requests[0])
    mantle_body: Final = _JSON_BODY.validate_json(mantle_counts.requests[1].content)
    assert [name for name in options if name in invoke_body] == []
    assert [name for name in options if name in mantle_body] != []
    assert {name: mantle_body[name] for name in ("messages", "tools")} == {
        name: invoke_body[name] for name in ("messages", "tools")
    }


@pytest.mark.asyncio
async def test_claude_chat_tool_use_is_counted_in_the_shape_claude_inference_sends() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))
    schema: Final = {"type": "object", "properties": {"city": {"type": "string"}}}

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=[
            {"role": "user", "content": "Weather in SF?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": schema}}],
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 412)
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    body: Final = _invoke_body(bedrock.requests[0])
    assert body["tools"] == [{"name": "get_weather", "input_schema": schema, "type": "custom"}]
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in SF?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}]},
    ]

@pytest.mark.asyncio
async def test_claude_chat_tool_use_counted_on_mantle_metadata_route_is_translated_too() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))
    schema: Final = {"type": "object", "properties": {"city": {"type": "string"}}}

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="bedrock_mantle/global.anthropic.claude-opus-4-8",
        messages=[
            {"role": "user", "content": "Weather in SF?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        tools=[{"type": "function", "function": {"name": "get_weather", "parameters": schema}}],
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 2177)
    assert bedrock.posted_hosts() == (MANTLE_HOST,)
    body: Final = _JSON_BODY.validate_json(bedrock.requests[0].content)
    assert body["tools"] == [{"name": "get_weather", "input_schema": schema, "type": "custom"}]
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in SF?"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "host", "tokenizer", "expected_tokens"),
    (
        ("bedrock_mantle/anthropic.claude-opus-5-5", MANTLE_HOST, "bedrock_mantle_api", 271),
        ("bedrock/anthropic.claude-3-haiku-20240307-v1:0", RUNTIME_HOST, "bedrock_api", 193),
    ),
)
async def test_qualified_counting_model_uses_its_endpoint_without_a_failed_probe(
    model: str, host: str, tokenizer: str, expected_tokens: int
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 193}), mantle=(200, {"input_tokens": 271}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use=model,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="served-model",
    )

    assert result is not None
    assert result.total_tokens == expected_tokens
    assert result.tokenizer_type == tokenizer
    assert bedrock.posted_hosts() == (host,)
    if host == MANTLE_HOST:
        assert _JSON_BODY.validate_json(bedrock.requests[0].content)["model"] == model.split("/", 1)[1]


@pytest.mark.asyncio
async def test_explicit_mantle_failure_does_not_probe_runtime() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 193}), mantle=(404, UNSUPPORTED))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="bedrock_mantle/anthropic.claude-opus-5-5",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 404
    assert result.tokenizer_type == "bedrock_mantle_api"
    assert bedrock.posted_hosts() == (MANTLE_HOST,)
