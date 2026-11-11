import base64
import copy
from collections.abc import Mapping
from itertools import chain
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
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


_CONVERSE_REPLY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": "6"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 27, "outputTokens": 1, "totalTokens": 28},
}


def _texts(content: JsonValue) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    blocks: Final = content if isinstance(content, list) else []
    return tuple(text for block in blocks if isinstance(block, dict) and isinstance(text := block.get("text"), str))


def _role_texts(message: JsonValue) -> tuple[tuple[JsonValue, str], ...]:
    if not isinstance(message, dict):
        return ()
    role: Final = message.get("role")
    return tuple((role, text) for text in _texts(message.get("content")))


def _counted_body(request: httpx.Request) -> Mapping[str, JsonValue]:
    counted_input: Final = _JSON_BODY.validate_json(request.content)["input"]
    assert isinstance(counted_input, dict)
    converse: Final = counted_input.get("converse")
    return converse if isinstance(converse, dict) else _invoke_body(request)


def _sent_texts(body: Mapping[str, JsonValue]) -> tuple[tuple[JsonValue, str], ...]:
    messages: Final = body["messages"]
    return tuple(chain.from_iterable(map(_role_texts, messages if isinstance(messages, list) else [])))


def _empty_turns(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    messages: Final = body["messages"]
    return tuple(
        message
        for message in (messages if isinstance(messages, list) else [])
        if isinstance(message, dict) and not message.get("content")
    )


def _sent_system_and_texts(body: Mapping[str, JsonValue]) -> tuple[tuple[JsonValue, str], ...]:
    return (*(("system prompt", text) for text in _texts(body.get("system"))), *_sent_texts(body))


_DROPPED: Final = (("user", "What is 2+2?"), ("user", "And 3+3?"))
_CONTINUED: Final = (("user", "What is 2+2?"), ("assistant", "Please continue."), ("user", "And 3+3?"))
_EMPTY_WITH_CACHE_CONTROL: Final = {"role": "assistant", "content": "", "cache_control": {"type": "ephemeral"}}
_ANSWER: Final = {"role": "assistant", "content": "Four."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "between_users, modify_params, expected_sent",
    (
        (({"role": "assistant", "content": ""},), False, _DROPPED),
        (({"role": "assistant", "content": None},), False, _DROPPED),
        (({"role": "assistant", "content": []},), False, _DROPPED),
        (({"role": "assistant", "content": None},), True, _CONTINUED),
        (({"role": "assistant", "content": []},), True, _CONTINUED),
        (({"role": "assistant", "content": [{"type": "text", "text": ""}]},), False, _DROPPED),
        (
            (
                {
                    "role": "assistant",
                    "content": "",
                    "thinking_blocks": [{"type": "thinking", "thinking": "Adding.", "signature": "c2lnbmF0dXJl"}],
                },
            ),
            False,
            _DROPPED,
        ),
        (({"role": "assistant", "content": ""},), True, _CONTINUED),
        (({"role": "assistant", "content": [{"type": "text", "text": " "}]},), True, _CONTINUED),
        (({"role": "assistant", "content": ""},), None, _CONTINUED),
        (
            ({"role": "assistant", "content": "", "function_call": {"name": "add", "arguments": "{}"}},),
            True,
            _CONTINUED,
        ),
        ((_EMPTY_WITH_CACHE_CONTROL,), None, _DROPPED),
        ((_EMPTY_WITH_CACHE_CONTROL,), True, _CONTINUED),
        (
            (_ANSWER, _EMPTY_WITH_CACHE_CONTROL),
            True,
            (("user", "What is 2+2?"), ("assistant", "Four."), ("user", "And 3+3?")),
        ),
    ),
    ids=(
        "empty_string",
        "null_content",
        "empty_content_list",
        "modify_params_null_content",
        "modify_params_empty_content_list",
        "empty_text_block",
        "thinking_and_empty_string",
        "modify_params_empty_string",
        "modify_params_blank_text_block",
        "modify_params_null_empty_string",
        "modify_params_function_call_and_empty_string",
        "modify_params_null_empty_string_and_cache_control",
        "modify_params_empty_string_and_cache_control",
        "modify_params_empty_string_and_cache_control_after_an_answer",
    ),
)
async def test_claude_converse_count_sends_the_texts_converse_inference_sends(
    between_users: tuple[dict[str, JsonValue], ...],
    modify_params: bool | None,
    expected_sent: tuple[tuple[str, str], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)
    credentials: Final = DEPLOYMENT["litellm_params"]
    deployment_model: Final = "bedrock/us.anthropic.claude-haiku-5-5"
    messages: Final = [
        {"role": "user", "content": "What is 2+2?"},
        *between_users,
        {"role": "user", "content": "And 3+3?"},
    ]
    inference: Final = _Bedrock(runtime=(200, _CONVERSE_REPLY), mantle=(200, _CONVERSE_REPLY))
    await litellm.acompletion(
        model=deployment_model,
        messages=messages,
        max_tokens=8,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=messages,
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}},
        request_model="claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert result.error is False
    converse_sent: Final = _sent_texts(_JSON_BODY.validate_json(inference.requests[0].content))
    assert _sent_texts(_counted_body(bedrock.requests[0])) == converse_sent
    assert not _empty_turns(_counted_body(bedrock.requests[0]))
    assert converse_sent == expected_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blank_content",
    ([], None, "", "   ", [{"type": "text", "text": ""}]),
    ids=("empty_list", "null", "empty_string", "spaces", "empty_text_block"),
)
@pytest.mark.parametrize(
    "modify_params, continue_settings",
    (
        (False, {}),
        (True, {}),
        (None, {}),
        (False, {"user_continue_message": {"role": "user", "content": "Go on."}}),
    ),
    ids=("no_continue_setting", "modify_params", "modify_params_null", "user_continue_message"),
)
async def test_claude_converse_count_sends_the_blank_last_user_turn_converse_inference_sends(
    blank_content: JsonValue,
    modify_params: bool | None,
    continue_settings: dict[str, JsonValue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)
    deployment_params: Final = {
        **DEPLOYMENT["litellm_params"],
        **continue_settings,
        "model": "bedrock/us.anthropic.claude-haiku-5-5",
    }
    messages: Final = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": blank_content},
    ]
    inference: Final = _Bedrock(runtime=(200, _CONVERSE_REPLY), mantle=(200, _CONVERSE_REPLY))
    await litellm.acompletion(
        messages=copy.deepcopy(messages),
        max_tokens=8,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **deployment_params,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=copy.deepcopy(messages),
        contents=None,
        deployment={"litellm_params": deployment_params},
        request_model="claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert result.error is False
    converse_sent: Final = _sent_texts(_JSON_BODY.validate_json(inference.requests[0].content))
    assert converse_sent[:2] == (("user", "What is 2+2?"), ("assistant", "Four."))
    assert len(converse_sent) == 3
    assert _sent_texts(_counted_body(bedrock.requests[0])) == converse_sent


@pytest.mark.asyncio
async def test_claude_converse_count_under_modify_params_answers_for_a_text_block_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", True)
    credentials: Final = DEPLOYMENT["litellm_params"]
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [{"type": "text"}]},
            {"role": "user", "content": "And 3+3?"},
        ],
        contents=None,
        deployment={"litellm_params": {**credentials, "model": "bedrock/us.anthropic.claude-haiku-5-5"}},
        request_model="claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert result.error is False
    assert _sent_texts(_counted_body(bedrock.requests[0])) == _CONTINUED


def _add_call(call_id: str) -> dict[str, JsonValue]:
    return {"id": call_id, "type": "function", "function": {"name": "add", "arguments": "{}"}}


_ADD_TOOL: Final = {"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}


async def _converse_sent_and_counted(
    messages: list[dict[str, JsonValue]],
    tools: list[dict[str, JsonValue]],
    model: str = "us.anthropic.claude-haiku-5-5",
    response_format: dict[str, JsonValue] | None = None,
    counted_model: str | None = None,
    continue_settings: dict[str, JsonValue] | None = None,
) -> tuple[tuple[tuple[JsonValue, str], ...], tuple[tuple[JsonValue, str], ...]]:
    credentials: Final = {**DEPLOYMENT["litellm_params"], **(continue_settings or {})}
    deployment_model: Final = f"bedrock/{model}"
    chat_params: Final = {"response_format": response_format} if response_format else {}
    inference: Final = _Bedrock(runtime=(200, _CONVERSE_REPLY), mantle=(200, _CONVERSE_REPLY))
    await litellm.acompletion(
        model=deployment_model,
        messages=copy.deepcopy(messages),
        max_tokens=8,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **({"tools": tools} if tools else {}),
        **chat_params,
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use=counted_model or model,
        messages=messages,
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}},
        request_model=model,
        tools=tools or None,
        messages_format="openai",
        chat_params=chat_params,
    )

    assert result is not None
    assert result.error is False
    assert not _empty_turns(_counted_body(bedrock.requests[0]))
    return (
        _sent_system_and_texts(_JSON_BODY.validate_json(inference.requests[0].content)),
        _sent_system_and_texts(_counted_body(bedrock.requests[0])),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_turns, tools, expected_sent",
    (
        (
            (
                {"role": "assistant", "content": None, "tool_calls": [_add_call("call_1")]},
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            ),
            [_ADD_TOOL],
            _CONTINUED,
        ),
        (
            (
                {"role": "assistant", "content": None, "tool_calls": [_add_call("call_1"), _add_call("call_2")]},
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
                {"role": "tool", "tool_call_id": "call_2", "content": "6"},
            ),
            [_ADD_TOOL],
            _CONTINUED,
        ),
        (
            (
                {"role": "assistant", "content": None, "tool_calls": [_add_call("call_1")]},
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            ),
            [],
            (
                ("user", "What is 2+2?"),
                ("assistant", "[tool call call_1: add({})]"),
                ("user", "[tool result for call_1: 4]"),
                ("user", "And 3+3?"),
            ),
        ),
    ),
    ids=("tool_result_then_user", "parallel_tool_results_then_user", "tool_history_without_tools"),
)
async def test_claude_converse_count_under_modify_params_sends_tool_history_the_way_converse_inference_sends_it(
    tool_turns: tuple[dict[str, JsonValue], ...],
    tools: list[dict[str, JsonValue]],
    expected_sent: tuple[tuple[str, str], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", True)

    converse_sent, counted = await _converse_sent_and_counted(
        [{"role": "user", "content": "What is 2+2?"}, *tool_turns, {"role": "user", "content": "And 3+3?"}], tools
    )

    assert counted == converse_sent
    assert converse_sent == expected_sent


_QUESTION: Final = {"role": "user", "content": "What is 2+2?"}
_BE_BRIEF: Final = {"role": "system", "content": "Be brief."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages, modify_params, expected_sent",
    (
        (
            [_QUESTION, _EMPTY_WITH_CACHE_CONTROL],
            True,
            (("user", "What is 2+2?"), ("assistant", "Please continue."), ("user", "Please continue.")),
        ),
        (
            [_EMPTY_WITH_CACHE_CONTROL, _QUESTION],
            True,
            (("user", "Please continue."), ("assistant", "Please continue."), ("user", "What is 2+2?")),
        ),
        (
            [_QUESTION, _ANSWER],
            True,
            (("user", "What is 2+2?"), ("assistant", "Four."), ("user", "Please continue.")),
        ),
        ([_QUESTION, _ANSWER], None, (("user", "What is 2+2?"), ("assistant", "Four."))),
        (
            [_BE_BRIEF, _ANSWER, _QUESTION],
            True,
            (
                ("system prompt", "Be brief."),
                ("user", "Please continue."),
                ("assistant", "Four."),
                ("user", "What is 2+2?"),
            ),
        ),
        ([_ANSWER], True, (("user", "Please continue."), ("assistant", "Four."), ("user", "Please continue."))),
        ([_BE_BRIEF], True, (("system prompt", "Be brief."), ("user", "Please continue."))),
        (
            [_QUESTION, {"role": "assistant", "content": "The answer is", "prefix": True}],
            True,
            (("user", "What is 2+2?"), ("assistant", "The answer is")),
        ),
    ),
    ids=(
        "final_empty_string_and_cache_control",
        "leading_empty_string_and_cache_control",
        "final_answer",
        "final_answer_under_null",
        "leading_answer_after_a_system_message",
        "only_an_answer",
        "only_a_system_message",
        "final_prefix",
    ),
)
async def test_claude_converse_count_sends_the_edge_user_turns_converse_inference_adds(
    messages: list[dict[str, JsonValue]],
    modify_params: bool | None,
    expected_sent: tuple[tuple[str, str], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)

    converse_sent, counted = await _converse_sent_and_counted(messages, [])

    assert counted == converse_sent
    assert converse_sent == expected_sent


_TOOL_ROUND: Final = (
    {"role": "assistant", "content": None, "tool_calls": [_add_call("call_1")]},
    {"role": "tool", "tool_call_id": "call_1", "content": "4"},
)
_FOLLOW_UP: Final = {"role": "user", "content": "And 3+3?"}


@pytest.mark.asyncio
@pytest.mark.parametrize("modify_params", (False, True))
@pytest.mark.parametrize(
    "messages",
    (
        [_BE_BRIEF, _QUESTION, *_TOOL_ROUND, _FOLLOW_UP],
        [{"role": "developer", "content": "Be brief."}, _QUESTION, *_TOOL_ROUND, _FOLLOW_UP],
        [_QUESTION, *_TOOL_ROUND, _BE_BRIEF, _FOLLOW_UP],
        [_QUESTION, *_TOOL_ROUND, _FOLLOW_UP, _BE_BRIEF],
        [_QUESTION, _BE_BRIEF, _EMPTY_WITH_CACHE_CONTROL],
        [_BE_BRIEF, _QUESTION],
    ),
    ids=(
        "leading_system_and_tool_history",
        "leading_developer_and_tool_history",
        "system_after_tool_history",
        "trailing_system_after_tool_history",
        "system_before_an_empty_assistant_turn",
        "leading_system",
    ),
)
async def test_claude_converse_count_sends_system_messages_where_converse_inference_sends_them(
    messages: list[dict[str, JsonValue]], modify_params: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)

    converse_sent, counted = await _converse_sent_and_counted(messages, [])

    assert counted == converse_sent
    assert "Be brief." in (text for _, text in converse_sent)


@pytest.mark.asyncio
async def test_claude_converse_count_keeps_tool_history_when_neutralizing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "modify_params", True)
    monkeypatch.setattr(litellm, "bedrock_neutralize_orphaned_tool_blocks", False)

    converse_sent, counted = await _converse_sent_and_counted([_QUESTION, *_TOOL_ROUND, _FOLLOW_UP], [])

    assert counted == converse_sent
    assert not any(text.startswith("[tool") for _, text in counted)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("us.anthropic.claude-haiku-5-5", "us.anthropic.claude-opus-4-7"))
async def test_claude_converse_count_keeps_tool_history_the_response_format_tool_keeps(model: str) -> None:
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }

    converse_sent, counted = await _converse_sent_and_counted(
        [_QUESTION, *_TOOL_ROUND, _FOLLOW_UP], [], model=model, response_format=response_format
    )

    assert counted == converse_sent


@pytest.mark.asyncio
async def test_claude_converse_count_decides_the_response_format_tool_by_the_deployment_model() -> None:
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }

    converse_sent, counted = await _converse_sent_and_counted(
        [_QUESTION, *_TOOL_ROUND, _FOLLOW_UP],
        [],
        model="us.anthropic.claude-opus-4-8",
        response_format=response_format,
        counted_model="anthropic.claude-opus-4-8",
    )

    assert counted == converse_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("modify_params", (False, True), ids=("modify_params_off", "modify_params"))
@pytest.mark.parametrize(
    "continue_content", ("Go on.", [{"type": "text", "text": "Go on."}]), ids=("string_content", "list_content")
)
@pytest.mark.parametrize(
    "messages, expected_sent",
    (
        ([_QUESTION, _ANSWER], (("user", "What is 2+2?"), ("assistant", "Four."), ("user", "Go on."))),
        ([_ANSWER, _QUESTION], (("user", "Go on."), ("assistant", "Four."), ("user", "What is 2+2?"))),
        (
            [_QUESTION, _ANSWER, {"role": "user", "content": ""}],
            (("user", "What is 2+2?"), ("assistant", "Four."), ("user", "Go on.")),
        ),
    ),
    ids=("ends_on_an_assistant_turn", "starts_on_an_assistant_turn", "ends_on_a_blank_user_turn"),
)
async def test_claude_converse_count_adds_the_continue_message_the_deployment_sets(
    messages: list[dict[str, JsonValue]],
    expected_sent: tuple[tuple[str, str], ...],
    continue_content: JsonValue,
    modify_params: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)

    converse_sent, counted = await _converse_sent_and_counted(
        messages, [], continue_settings={"user_continue_message": {"role": "user", "content": continue_content}}
    )

    assert converse_sent == expected_sent
    assert counted == converse_sent


@pytest.mark.asyncio
async def test_claude_converse_count_follows_inference_for_an_empty_continue_message() -> None:
    converse_sent, counted = await _converse_sent_and_counted(
        [_QUESTION, _ANSWER, {"role": "user", "content": ""}], [], continue_settings={"user_continue_message": {}}
    )

    assert counted == converse_sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blank_content",
    ([], None, "", [{"type": "text", "text": ""}]),
    ids=("empty_list", "null", "empty_string", "empty_text_block"),
)
@pytest.mark.parametrize(
    "modify_params, sent_for_blank",
    ((False, ()), (True, (("user", "Please continue."),))),
    ids=("dropped", "continued"),
)
async def test_claude_converse_count_matches_inference_for_an_earlier_empty_user_turn(
    blank_content: JsonValue,
    modify_params: bool,
    sent_for_blank: tuple[tuple[str, str], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "modify_params", modify_params)

    converse_sent, counted = await _converse_sent_and_counted(
        [_QUESTION, _ANSWER, {"role": "user", "content": blank_content}, {"role": "user", "content": "And 3+3?"}], []
    )

    assert converse_sent == (
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        *sent_for_blank,
        ("user", "And 3+3?"),
    )
    assert counted == converse_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("block", ({"type": "text"}, {"type": "text", "text": None}), ids=("no_text", "null_text"))
async def test_claude_converse_count_answers_400_for_a_malformed_user_text_block(
    block: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "modify_params", True)
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=[_QUESTION, _ANSWER, {"role": "user", "content": [block]}],
        contents=None,
        deployment={
            "litellm_params": {**DEPLOYMENT["litellm_params"], "model": "bedrock/us.anthropic.claude-haiku-5-5"}
        },
        request_model="us.anthropic.claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.status_code) == (True, 400)
    assert not bedrock.requests


@pytest.mark.asyncio
async def test_claude_converse_count_keeps_string_user_content_when_it_lifts_a_system_message() -> None:
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="bedrock_mantle/us.anthropic.claude-haiku-5-5",
        messages=[{"role": "system", "content": "Be brief."}, _QUESTION],
        contents=None,
        deployment={
            "litellm_params": {**DEPLOYMENT["litellm_params"], "model": "bedrock/us.anthropic.claude-haiku-5-5"}
        },
        request_model="us.anthropic.claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert result.error is False
    counted: Final = _JSON_BODY.validate_json(bedrock.requests[0].content)
    assert counted["messages"] == [_QUESTION]
    assert _texts(counted["system"]) == ("Be brief.",)


_IMAGE_PART: Final = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


def _system_blocks(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    system: Final = body.get("system")
    blocks: Final = system if isinstance(system, list) else []
    return tuple(block.get("text") if isinstance(block, dict) else block for block in blocks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "system_content",
    (
        [{"type": "text", "text": "Be brief."}, _IMAGE_PART],
        [_IMAGE_PART, {"type": "text", "text": "Be brief."}],
        [_IMAGE_PART],
    ),
    ids=("text_then_image", "image_then_text", "only_an_image"),
)
async def test_claude_converse_count_sends_only_the_system_text_converse_inference_sends(
    system_content: JsonValue,
) -> None:
    credentials: Final = DEPLOYMENT["litellm_params"]
    deployment_model: Final = "bedrock/us.anthropic.claude-haiku-5-5"
    messages: Final = [{"role": "system", "content": system_content}, _QUESTION]
    inference: Final = _Bedrock(runtime=(200, _CONVERSE_REPLY), mantle=(200, _CONVERSE_REPLY))
    await litellm.acompletion(
        model=deployment_model,
        messages=copy.deepcopy(messages),
        max_tokens=8,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(inference)),
        **credentials,
    )
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=copy.deepcopy(messages),
        contents=None,
        deployment={"litellm_params": {**credentials, "model": deployment_model}},
        request_model="claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert result.error is False
    converse_body: Final = _JSON_BODY.validate_json(inference.requests[0].content)
    assert _system_blocks(_counted_body(bedrock.requests[0])) == _system_blocks(converse_body)
    assert _sent_texts(_counted_body(bedrock.requests[0])) == _sent_texts(converse_body)


@pytest.mark.asyncio
async def test_claude_converse_count_of_a_message_without_a_role_answers_bedrocks_400() -> None:
    bedrock: Final = _Bedrock(
        runtime=(400, {"message": "messages.0.role: Field required"}),
        mantle=(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "role"}}),
    )

    result: Final = await _counter(bedrock).count_tokens(
        model_to_use="us.anthropic.claude-haiku-5-5",
        messages=[{"content": "What is 2+2?"}],
        contents=None,
        deployment={"litellm_params": {**DEPLOYMENT["litellm_params"], "model": "bedrock/us.anthropic.claude-haiku-5-5"}},
        request_model="claude-haiku-5-5",
        messages_format="openai",
    )

    assert result is not None
    assert (result.error, result.status_code) == (True, 400)


@pytest.mark.asyncio
async def test_claude_converse_count_of_tool_history_without_tools_logs_no_neutralizing_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    credentials: Final = DEPLOYMENT["litellm_params"]
    bedrock: Final = _Bedrock(runtime=(200, {"inputTokens": 27}), mantle=(200, {"input_tokens": 27}))

    with caplog.at_level("WARNING"):
        result: Final = await _counter(bedrock).count_tokens(
            model_to_use="us.anthropic.claude-haiku-5-5",
            messages=[_QUESTION, *_TOOL_ROUND, _FOLLOW_UP],
            contents=None,
            deployment={"litellm_params": {**credentials, "model": "bedrock/us.anthropic.claude-haiku-5-5"}},
            request_model="claude-haiku-5-5",
            messages_format="openai",
        )

    assert result is not None
    assert result.error is False
    assert ("user", "[tool result for call_1: 4]") in _sent_texts(_counted_body(bedrock.requests[0]))
    assert not any("neutralizing orphaned tool blocks" in record.getMessage() for record in caplog.records)


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
