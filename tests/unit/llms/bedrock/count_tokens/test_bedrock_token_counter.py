import base64
import json
from collections.abc import Callable, Mapping
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.anthropic import (
    ANTHROPIC_EFFORT_BETA_HEADER,
    ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER,
)

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


_CONTEXT_EDITS: Final[dict[str, JsonValue]] = {
    "edits": [{"type": "clear_tool_uses_20250919"}, {"type": "clear_thinking_20251015"}]
}
_THINKING_AND_TOOL_CHOICE: Final[dict[str, JsonValue]] = {
    "thinking": {"type": "enabled", "budget_tokens": 2000},
    "tool_choice": {"type": "auto"},
}
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {"name": "get_weather", "input_schema": {"type": "object"}}


_CITY_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_CITY_FORMAT: Final[dict[str, JsonValue]] = {"output_config": {"format": {"type": "json_schema", "schema": _CITY_SCHEMA}}}


_CONTEXT_EDITING_OPTIONS: Final[dict[str, JsonValue]] = {**_THINKING_AND_TOOL_CHOICE, "context_management": _CONTEXT_EDITS}


async def _messages_inference_and_count(
    deployment_model: str,
    model_to_use: str,
    request_options: Mapping[str, JsonValue] = _CONTEXT_EDITING_OPTIONS,
    tool: Mapping[str, JsonValue] = _WEATHER_TOOL,
    runtime: tuple[int, Mapping[str, object]] = (200, {"inputTokens": 412}),
    inference_reply: Mapping[str, object] = _CLAUDE_REPLY,
) -> tuple[_Bedrock, _Bedrock]:
    credentials: Final = DEPLOYMENT["litellm_params"]
    inference: Final = _Bedrock(runtime=(200, inference_reply), mantle=(200, inference_reply))
    await litellm.anthropic_messages(
        model=deployment_model,
        messages=MESSAGES,
        tools=[dict(tool)],
        max_tokens=4096,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **request_options,
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=runtime, mantle=(200, {"input_tokens": 2177}))
    result: Final = await _counter(bedrock).count_tokens(
        model_to_use=model_to_use,
        messages=MESSAGES,
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}},
        request_model="claude",
        tools=[dict(tool)],
        request_options=request_options,
    )
    assert result is not None
    assert result.error is False
    return inference, bedrock


def _betas_besides_effort(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    betas: Final = body.get("anthropic_beta") or []
    assert isinstance(betas, list)
    return tuple(beta for beta in betas if beta != ANTHROPIC_EFFORT_BETA_HEADER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_model", "model_to_use"),
    (
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-6"),
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ),
)
async def test_runtime_count_sends_the_options_and_context_edits_invoke_inference_sends(
    deployment_model: str, model_to_use: str
) -> None:
    inference, bedrock = await _messages_inference_and_count(deployment_model, model_to_use)

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    counted: Final = _invoke_body(bedrock.requests[0])
    sent: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    fields: Final = ("thinking", "tool_choice", "output_config", "context_management")
    assert {name: counted.get(name) for name in fields} == {name: sent.get(name) for name in fields}
    assert _betas_besides_effort(counted) == _betas_besides_effort(sent)
    thinking: Final = counted["thinking"]
    assert isinstance(thinking, dict)
    budget_tokens: Final = thinking.get("budget_tokens", 0)
    assert isinstance(budget_tokens, int) and isinstance(counted["max_tokens"], int)
    assert counted["max_tokens"] > budget_tokens


@pytest.mark.asyncio
async def test_mantle_count_sends_the_options_context_edits_and_beta_mantle_inference_sends() -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock_mantle/us.anthropic.claude-opus-5-5", "bedrock_mantle/us.anthropic.claude-opus-5-5"
    )

    assert bedrock.posted_hosts() == (MANTLE_HOST,)
    counted: Final = _JSON_BODY.validate_json(bedrock.requests[0].content)
    sent: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    fields: Final = ("thinking", "tool_choice", "output_config", "context_management")
    assert {name: counted.get(name) for name in fields} == {name: sent.get(name) for name in fields}
    assert "anthropic_beta" not in counted
    assert bedrock.requests[0].headers["anthropic-beta"] == inference.requests[0].headers["anthropic-beta"]


_DISPLAY_THINKING: Final[dict[str, JsonValue]] = {"thinking": {"type": "adaptive", "display": "updates"}}


def _sent_betas(request: httpx.Request) -> tuple[str, ...]:
    return tuple(sorted(beta for beta in request.headers.get("anthropic-beta", "").split(",") if beta))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_model", "request_options", "expected_beta"),
    (
        (
            "bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0",
            {"output_config": {"effort": "low"}},
            ANTHROPIC_EFFORT_BETA_HEADER,
        ),
        ("bedrock/invoke/us.anthropic.claude-opus-5-5", _DISPLAY_THINKING, ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER),
    ),
    ids=("effort", "thinking_display"),
)
async def test_runtime_count_sends_the_betas_invoke_inference_sends(
    deployment_model: str, request_options: dict[str, JsonValue], expected_beta: str
) -> None:
    inference, bedrock = await _messages_inference_and_count(
        deployment_model, deployment_model.removeprefix("bedrock/invoke/"), request_options
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    counted_betas: Final = _invoke_body(bedrock.requests[0]).get("anthropic_beta")
    assert counted_betas == _JSON_BODY.validate_json(inference.requests[0].content).get("anthropic_beta")
    assert isinstance(counted_betas, list) and expected_beta in counted_betas


_EFFORT: Final[dict[str, JsonValue]] = {"output_config": {"effort": "low"}}
_CLEAR_THINKING_EDITS: Final[dict[str, JsonValue]] = {
    "context_management": {"edits": [{"type": "clear_thinking_20251015"}]}
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_options", "expected_betas"),
    ((_DISPLAY_THINKING, (ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER,)), (_EFFORT, ())),
    ids=("thinking_display", "effort"),
)
async def test_invoke_count_retried_on_mantle_sends_the_betas_invoke_inference_sends(
    request_options: dict[str, JsonValue], expected_betas: tuple[str, ...]
) -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock/invoke/us.anthropic.claude-opus-5-5",
        "us.anthropic.claude-opus-5-5",
        request_options,
        runtime=(400, UNSUPPORTED),
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST, MANTLE_HOST)
    sent_betas: Final = _JSON_BODY.validate_json(inference.requests[0].content).get("anthropic_beta", [])
    assert isinstance(sent_betas, list)
    assert _sent_betas(bedrock.requests[1]) == tuple(sorted(sent_betas)) == expected_betas


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_model", "request_options"),
    (
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", _EFFORT),
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", _CLEAR_THINKING_EDITS),
        ("bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0", _EFFORT),
    ),
    ids=("effort", "clear_thinking_sets_effort", "inference_sends_the_effort_beta"),
)
async def test_runtime_count_of_an_effort_sends_the_effort_beta_once_beside_the_betas_invoke_inference_sends(
    deployment_model: str, request_options: dict[str, JsonValue]
) -> None:
    inference, bedrock = await _messages_inference_and_count(
        deployment_model, deployment_model.removeprefix("bedrock/invoke/"), request_options
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    counted: Final = _invoke_body(bedrock.requests[0])
    sent: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    counted_config: Final = counted.get("output_config")
    assert isinstance(counted_config, dict) and counted_config.get("effort") == "low"
    counted_betas: Final = counted.get("anthropic_beta")
    sent_betas: Final = sent.get("anthropic_beta", [])
    assert isinstance(counted_betas, list) and isinstance(sent_betas, list)
    assert counted_betas.count(ANTHROPIC_EFFORT_BETA_HEADER) == 1
    assert set(counted_betas) - {ANTHROPIC_EFFORT_BETA_HEADER} == set(sent_betas) - {ANTHROPIC_EFFORT_BETA_HEADER}


@pytest.mark.asyncio
async def test_runtime_count_of_an_output_format_sends_only_the_betas_invoke_inference_sends() -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock/invoke/us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-sonnet-4-6",
        {**_CITY_FORMAT, "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    counted: Final = _invoke_body(bedrock.requests[0])
    sent: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    assert counted.get("output_config") == sent.get("output_config")
    assert counted.get("anthropic_beta") == sent.get("anthropic_beta")


_CONVERSE_REPLY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
    "metrics": {"latencyMs": 1},
}
_NO_PARALLEL_TOOL_USE: Final[dict[str, JsonValue]] = {
    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_model", "inference_reply"),
    (
        ("bedrock/converse/us.anthropic.claude-sonnet-4-6", _CONVERSE_REPLY),
        ("bedrock/invoke/us.anthropic.claude-sonnet-4-6", _CLAUDE_REPLY),
    ),
    ids=("converse", "invoke"),
)
async def test_runtime_count_sends_disable_parallel_tool_use_only_when_messages_inference_sends_it(
    deployment_model: str, inference_reply: Mapping[str, object]
) -> None:
    inference, bedrock = await _messages_inference_and_count(
        deployment_model,
        deployment_model.split("/")[-1],
        _NO_PARALLEL_TOOL_USE,
        inference_reply=inference_reply,
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    counted_tool_choice: Final = _invoke_body(bedrock.requests[0]).get("tool_choice")
    assert isinstance(counted_tool_choice, dict) and counted_tool_choice.get("type") == "auto"
    assert ("disable_parallel_tool_use" in counted_tool_choice) == (
        b"disable_parallel_tool_use" in inference.requests[0].content
    )


@pytest.mark.asyncio
async def test_count_on_a_stand_in_model_counts_the_messages_and_output_config_the_deployment_model_sends() -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock/us.anthropic.claude-haiku-5-5", "bedrock_mantle/us.anthropic.claude-opus-5-5", _CITY_FORMAT
    )

    assert bedrock.posted_hosts() == (MANTLE_HOST,)
    counted: Final = _JSON_BODY.validate_json(bedrock.requests[0].content)
    sent: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    fields: Final = ("messages", "output_config")
    assert {name: counted.get(name) for name in fields} == {name: sent.get(name) for name in fields}


@pytest.mark.asyncio
async def test_mantle_count_sends_the_thinking_display_beta_mantle_inference_sends() -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock_mantle/us.anthropic.claude-opus-5-5", "bedrock_mantle/us.anthropic.claude-opus-5-5", _DISPLAY_THINKING
    )

    assert bedrock.posted_hosts() == (MANTLE_HOST,)
    assert _sent_betas(bedrock.requests[0]) == _sent_betas(inference.requests[0])
    assert ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER in _sent_betas(bedrock.requests[0])


def _mantle_answers(status_code: int, body: Mapping[str, JsonValue]) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status_code, json=dict(body), request=request)


def _mantle_unreachable(error: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise error("bedrock-mantle is unreachable", request=request)

    return unreachable


_MANTLE_REJECTION: Final[dict[str, JsonValue]] = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "thinking: unsupported"},
}
_MANTLE_FAILURE: Final[dict[str, JsonValue]] = {"type": "error", "error": {"type": "api_error", "message": "down"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_options", "mantle", "expected"),
    (
        (_DISPLAY_THINKING, _mantle_unreachable(httpx.ConnectError), (True, 400, "bedrock_api")),
        (_DISPLAY_THINKING, _mantle_unreachable(httpx.ReadTimeout), (True, 400, "bedrock_api")),
        (_DISPLAY_THINKING, _mantle_answers(503, _MANTLE_FAILURE), (True, 400, "bedrock_api")),
        (_DISPLAY_THINKING, _mantle_answers(429, _MANTLE_FAILURE), (True, 400, "bedrock_api")),
        (_DISPLAY_THINKING, _mantle_answers(403, _MANTLE_FAILURE), (True, 400, "bedrock_api")),
        (_DISPLAY_THINKING, _mantle_answers(400, _MANTLE_REJECTION), (True, 400, "bedrock_mantle_api")),
        (_DISPLAY_THINKING, _mantle_answers(422, _MANTLE_REJECTION), (True, 422, "bedrock_mantle_api")),
        (_DISPLAY_THINKING, _mantle_answers(200, {"input_tokens": 2177}), (False, None, "bedrock_mantle_api")),
        ({}, _mantle_answers(503, _MANTLE_FAILURE), (True, 503, "bedrock_mantle_api")),
    ),
    ids=(
        "mantle-unreachable",
        "mantle-timeout",
        "mantle-5xx",
        "mantle-rate-limit",
        "mantle-denied",
        "mantle-rejects-the-options",
        "mantle-rejects-the-body",
        "mantle-counts",
        "no-options-mantle-5xx",
    ),
)
async def test_runtime_rejecting_the_options_is_answered_unless_mantle_counts_or_rejects_the_request(
    request_options: dict[str, JsonValue],
    mantle: Callable[[httpx.Request], httpx.Response],
    expected: tuple[bool, int | None, str],
) -> None:
    def bedrock(request: httpx.Request) -> httpx.Response:
        if request.url.host == RUNTIME_HOST:
            return httpx.Response(400, json=UNSUPPORTED, request=request)
        return mantle(request)

    result: Final = await BedrockTokenCounter(
        client=AsyncHTTPHandler(transport=httpx.MockTransport(bedrock))
    ).count_tokens(
        model_to_use="us.anthropic.claude-opus-5-5",
        messages=MESSAGES,
        contents=None,
        deployment={
            "litellm_params": {**DEPLOYMENT["litellm_params"], "model": "bedrock/us.anthropic.claude-opus-5-5"}
        },
        request_model="claude-opus-5-5",
        request_options=request_options,
    )

    assert result is not None
    assert (result.error, result.status_code, result.tokenizer_type) == expected


@pytest.mark.asyncio
async def test_runtime_count_sends_the_tools_invoke_inference_sends() -> None:
    inference, bedrock = await _messages_inference_and_count(
        "bedrock/invoke/us.anthropic.claude-opus-5-5",
        "us.anthropic.claude-opus-5-5",
        _THINKING_AND_TOOL_CHOICE,
        tool={**_WEATHER_TOOL, "eager_input_streaming": True},
    )

    assert bedrock.posted_hosts() == (RUNTIME_HOST,)
    assert _invoke_body(bedrock.requests[0])["tools"] == _JSON_BODY.validate_json(inference.requests[0].content)["tools"]


def _texts(message: JsonValue) -> tuple[str, ...]:
    content: Final = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return ()
    return tuple(text for block in content if isinstance(block, dict) and isinstance(text := block.get("text"), str))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native_output", "request_options", "schema_inlined"),
    (
        (False, _CITY_FORMAT, True),
        (True, _CITY_FORMAT, False),
        (False, _THINKING_AND_TOOL_CHOICE, False),
    ),
    ids=("inlined_schema", "native_schema", "no_format"),
)
async def test_runtime_count_counts_the_messages_bedrock_invoke_inference_sends(
    native_output: bool, request_options: dict[str, JsonValue], schema_inlined: bool
) -> None:
    deployment_model: Final = "bedrock/invoke/us.anthropic.claude-count-probe-v1:0"
    model_info: Final = {"supports_bedrock_native_structured_output": native_output}
    credentials: Final = DEPLOYMENT["litellm_params"]
    inference: Final = _Bedrock(runtime=(200, _CLAUDE_REPLY), mantle=(200, _CLAUDE_REPLY))
    await litellm.anthropic_messages(
        model=deployment_model,
        messages=MESSAGES,
        max_tokens=4096,
        model_info=model_info,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **request_options,
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-count-probe-v1:0",
        messages=MESSAGES,
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}, "model_info": model_info},
        request_model="claude",
        request_options=request_options,
    )

    assert result is not None
    assert (result.error, result.total_tokens) == (False, 412)
    counted: Final = _invoke_body(bedrock.requests[0])["messages"]
    assert counted == _JSON_BODY.validate_json(inference.requests[0].content)["messages"]
    assert isinstance(counted, list)
    assert (json.dumps(_CITY_SCHEMA) in _texts(counted[-1])) is schema_inlined
    assert (counted == MESSAGES) is not schema_inlined


_THINKING: Final[dict[str, JsonValue]] = {"thinking": {"type": "enabled", "budget_tokens": 2000}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "tools", "request_options"),
    (
        (MESSAGES, None, {"output_config": {"effort": "fastest"}}),
        (MESSAGES, [{"type": 5, "name": "x", "input_schema": {"type": "object"}}], _THINKING),
        (MESSAGES, ["get_weather"], _THINKING),
        ([{"role": "user", "content": 5}], None, _THINKING),
        ([{"role": "user", "content": [{"text": "hi"}]}], None, _THINKING),
    ),
    ids=("option_refused", "tool_type_not_a_string", "tool_not_a_mapping", "content_not_iterable", "block_without_type"),
)
async def test_claude_count_refuses_options_bedrock_messages_inference_refuses_without_calling_bedrock(
    messages: list[dict[str, JsonValue]], tools: list[JsonValue] | None, request_options: dict[str, JsonValue]
) -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 412}), mantle=(200, {"input_tokens": 2177}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=messages,
        contents=None,
        deployment=DEPLOYMENT,
        request_model="claude-sonnet-4-5",
        tools=tools,
        request_options=request_options,
    )

    assert result is not None
    assert (result.error, result.status_code, result.rejected_before_provider_call) == (True, 400, True)
    assert bedrock.posted_hosts() == ()


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
