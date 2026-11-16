import base64
from collections.abc import Mapping
from copy import deepcopy
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import CountContextManagement

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
async def test_claude_model_message_without_content_is_rejected_without_calling_runtime_or_mantle() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 1353}), mantle=(200, {"input_tokens": 1336}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=[{"role": "user"}],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
    )

    assert result is not None
    assert (result.error, result.status_code, result.error_message, result.rejected_before_provider_call) == (
        True,
        400,
        "Message 0 must have a 'content' field",
        True,
    )
    assert bedrock.posted_hosts() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "problem"),
    [
        ({"role": "user"}, "Message 0 must have a 'content' field"),
        ({"content": "hi"}, "Message 0 must have a 'role' field"),
    ],
)
async def test_mantle_counted_model_rejects_a_malformed_message_without_calling_mantle(
    message: dict[str, str], problem: str
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 1353}), mantle=(200, {"input_tokens": 1336}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="bedrock_mantle/anthropic.claude-opus-5-5",
        messages=[message],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-5-5",
    )

    assert result is not None
    assert (result.error, result.status_code, result.error_message, result.rejected_before_provider_call) == (
        True,
        400,
        problem,
        True,
    )
    assert bedrock.posted_hosts() == ()


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
async def test_runtime_rejection_of_an_edit_stands_when_mantle_does_not_serve_the_model() -> None:
    rejection: Final = "'claude-haiku-4-5-20251001' does not support the 'compact_20260112' context management strategy"
    bedrock: Final = _Bedrock(
        runtime=(400, {"message": rejection}),
        mantle=(404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}),
    )

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-haiku-4-5",
        context_management=CountContextManagement(
            spec={"edits": [{"type": "compact_20260112"}]}, drop_invalid_edits=False
        ),
    )

    assert result is not None
    assert result.error is True
    assert result.status_code == 400
    assert result.tokenizer_type == "bedrock_api"
    assert rejection in (result.error_message or "")
    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)


@pytest.mark.asyncio
async def test_runtime_400_without_an_edit_still_reports_the_mantle_failure() -> None:
    bedrock: Final = _Bedrock(
        runtime=(400, {"message": "The provided model identifier is invalid."}),
        mantle=(404, {"type": "error", "error": {"type": "not_found_error", "message": "does not exist"}}),
    )

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-haiku-4-5",
    )

    assert result is not None
    assert result.status_code == 404
    assert result.tokenizer_type == "bedrock_mantle_api"
    assert "does not exist" in (result.error_message or "")
    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)


@pytest.mark.asyncio
async def test_plain_text_history_with_an_edit_is_counted_through_invoke_carrying_the_edit() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 57}), mantle=(200, {"input_tokens": 1}))
    spec: Final[dict[str, JsonValue]] = {"edits": [{"type": "clear_tool_uses_20250919"}]}

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        context_management=CountContextManagement(spec=spec, drop_invalid_edits=False),
    )

    assert result is not None
    assert result.total_tokens == 57
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    invoke_body: Final = _invoke_body(bedrock.requests[0])
    assert invoke_body["context_management"] == spec
    assert invoke_body["anthropic_beta"] == ["context-management-2025-06-27"]


@pytest.mark.asyncio
async def test_an_empty_string_turn_with_an_edit_is_sent_as_an_empty_string_not_an_empty_text_block() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 16}), mantle=(200, {"input_tokens": 1}))
    spec: Final[dict[str, JsonValue]] = {"edits": [{"type": "clear_tool_uses_20250919"}]}

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "third question here"},
        ],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        context_management=CountContextManagement(spec=spec, drop_invalid_edits=False),
    )

    assert result is not None
    assert result.total_tokens == 16
    assert _invoke_body(bedrock.requests[0])["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": [{"type": "text", "text": "third question here"}]},
    ]


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
@pytest.mark.parametrize(
    ("model_to_use", "runtime", "posted_hosts"),
    (
        ("bedrock_mantle/anthropic.claude-opus-5-5", (200, {"inputTokens": 1}), (MANTLE_HOST,)),
        ("us.anthropic.claude-opus-5-5", (400, UNSUPPORTED), (RUNTIME_HOST, MANTLE_HOST)),
    ),
)
async def test_messages_count_options_reach_the_mantle_count(
    model_to_use: str, runtime: tuple[int, Mapping[str, object]], posted_hosts: tuple[str, ...]
) -> None:
    bedrock: Final = _Bedrock(runtime=runtime, mantle=(200, {"input_tokens": 433}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use=model_to_use,
        messages=MESSAGES,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-5-5",
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        count_options={"tool_choice": {"type": "any"}},
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 433)
    assert bedrock.posted_hosts() == posted_hosts
    assert _JSON_BODY.validate_json(bedrock.requests[-1].content)["tool_choice"] == {"type": "any"}


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
@pytest.mark.parametrize(
    "tool",
    (
        {"type": "mcp", "server_label": "wiki", "server_url": "https://mcp.example/mcp"},
        {"type": "computer_use_preview", "display_width": 1024, "display_height": 768, "environment": "browser"},
    ),
    ids=("mcp", "computer_use_preview"),
)
async def test_claude_count_rejects_a_lone_responses_tool_without_calling_bedrock(tool: dict[str, JsonValue]) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="global.anthropic.claude-opus-4-8",
        messages=[{"role": "user", "content": "Ask the wiki"}],
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-opus-4-8",
        tools=[tool],
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
    assert bedrock.posted_hosts() == ()



def _deployment(model: str) -> dict[str, JsonValue]:
    return {"litellm_params": {**DEPLOYMENT["litellm_params"], "model": model}}


_CONTENTLESS_TURN: Final = ({"role": "user"}, {"role": "user", "content": None})


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment", (DEPLOYMENT, _deployment("bedrock/us.anthropic.claude-opus-5-5")))
async def test_claude_converse_count_rejects_a_chat_message_without_a_content_key_without_calling_bedrock(
    deployment: dict[str, JsonValue],
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-opus-5-5",
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, {"role": "user"}],
        contents=None,
        deployment=deployment,
        request_model="claude-opus-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
    assert result.error_message == "Message 2 must have a 'content' field"
    assert bedrock.posted_hosts() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", _CONTENTLESS_TURN)
async def test_claude_invoke_count_sends_a_chat_message_without_content_as_chat_inference_sends_it(
    message: dict[str, JsonValue],
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))
    messages: Final = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}, message]
    inference: Final = AmazonAnthropicClaudeConfig().transform_request(
        model="us.anthropic.claude-opus-5-5",
        messages=deepcopy(messages),
        optional_params={},
        litellm_params={},
        headers={},
    )

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-opus-5-5",
        messages=messages,
        contents=None,
        deployment=_deployment("bedrock/invoke/us.anthropic.claude-opus-5-5"),
        request_model="claude-opus-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 412)
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    assert _invoke_body(bedrock.requests[0])["messages"] == inference["messages"]


_TOOL_CALL: Final = {
    "role": "assistant",
    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
}


_WEATHER_TOOL: Final = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "tools"),
    (
        (
            [
                {"role": "user", "content": "Weather in SF?"},
                _TOOL_CALL,
                {"role": "tool", "tool_call_id": "call_1", "content": None},
            ],
            None,
        ),
        (
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "name": "bob", "content": None},
            ],
            None,
        ),
        ([{"role": "system", "content": None}, {"role": "user", "content": "Hi"}], None),
        ([{"role": "user", "content": None}, {"role": "user", "content": "Hi"}], None),
        ([{"role": "user"}, {"role": "user", "content": "Weather in SF?"}], [_WEATHER_TOOL]),
    ),
    ids=(
        "null-content-tool-result",
        "null-content-named-user-turn",
        "null-content-system-turn",
        "null-content-user-turn",
        "contentless-user-turn-with-chat-tools",
    ),
)
async def test_claude_converse_count_sends_a_null_content_message_converse_inference_serves(
    messages: list[dict[str, JsonValue]], tools: list[dict[str, JsonValue]] | None
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-opus-5-5",
        messages=messages,
        contents=None,
        deployment=_deployment("bedrock/us.anthropic.claude-opus-5-5"),
        request_model="claude-opus-5-5",
        tools=tools,
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 412)
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)


@pytest.mark.asyncio
async def test_claude_count_sends_a_chat_tool_call_turn_sent_without_a_content_key() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-opus-5-5",
        messages=[
            {"role": "user", "content": "Weather in SF?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        contents=None,
        deployment=_deployment("bedrock/us.anthropic.claude-opus-5-5"),
        request_model="claude-opus-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 412)
    assert bedrock.posted_hosts() == (RUNTIME_HOST,)

_CLAUDE_REPLY: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deployment_model",
    ("bedrock/invoke/us.anthropic.claude-sonnet-5-5", "bedrock_mantle/us.anthropic.claude-sonnet-5-5"),
)
async def test_claude_count_sends_the_tool_search_tool_the_deployments_inference_sends(deployment_model: str) -> None:
    credentials: Final = DEPLOYMENT["litellm_params"]
    inference: Final = _Bedrock(runtime=(200, _CLAUDE_REPLY), mantle=(200, _CLAUDE_REPLY))
    await litellm.acompletion(
        model=deployment_model,
        messages=MESSAGES,
        tools=[{"type": "tool_search"}],
        max_tokens=8,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="bedrock_mantle/us.anthropic.claude-opus-5-5",
        messages=MESSAGES,
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}},
        request_model="claude-sonnet-5-5",
        tools=[{"type": "tool_search"}],
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 2177)
    counted_tools: Final = _JSON_BODY.validate_json(bedrock.requests[0].content)["tools"]
    assert counted_tools == _JSON_BODY.validate_json(inference.requests[0].content)["tools"]


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
