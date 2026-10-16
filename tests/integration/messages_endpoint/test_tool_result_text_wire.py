import json
import uuid
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import anthropic
import pytest
from integration._support.client import Gateway, Scenario, string_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_TOOL_USE_ID: Final = "toolu_01"
_WEATHER: Final = "sunny, 21C"
_FIELD_ERROR: Final = f"tool_result {_TOOL_USE_ID} content.0.text: Input should be a valid string"
_CLAUDE_CODE: Final = MappingProxyType({"user-agent": "claude-cli/2.1.0 (external, cli)"})
_OPENAI_COUNT: Final = 4242
_GOOGLE_COUNT: Final = 777
_ANTHROPIC_COUNT: Final = 58
_BEDROCK_COUNT: Final = 1345
_GEMINI: Final = "gemini-4-argon"
_VERTEX_PROJECT: Final = "scripted-project"
_CITY: Final = "Paris"
_REDACTED_CITY: Final = "[CITY]"
_TOOLS: Final[list[JsonValue]] = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _messages(*blocks: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": _TOOL_USE_ID, "name": "get_weather", "input": {"city": "Paris"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": _TOOL_USE_ID, "content": list(blocks)}]},
    ]


_NULL_TEXT: Final = _messages({"type": "text", "text": None})
_NUMBER_THEN_WEATHER: Final = _messages({"type": "text", "text": 5}, {"type": "text", "text": _WEATHER})
_WEATHER_ONLY: Final = _messages({"type": "text", "text": _WEATHER})
_WEATHER_THEN_MISSING: Final = _messages({"type": "text", "text": _WEATHER}, {"type": "text"})


def _body(model: str, messages: list[JsonValue], **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": messages, "tools": _TOOLS, **extra}


def _responses_reply() -> bytes:
    return json.dumps(
        {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": 1789788253,
            "status": "completed",
            "model": "gpt-5.4-mini",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex}",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "It is sunny.", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 41, "output_tokens": 5, "total_tokens": 46},
        }
    ).encode()


def _chat_reply() -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "mistral-small-latest",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "It is sunny."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        }
    ).encode()


def _anthropic_reply() -> bytes:
    return json.dumps(
        {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "It is sunny."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 9, "output_tokens": 4},
        }
    ).encode()


def _openai_peer(request: Request) -> Reply:
    if request.target.endswith("/responses/input_tokens"):
        return Reply(body=json.dumps({"input_tokens": _OPENAI_COUNT}).encode())
    if request.target.endswith("/responses"):
        return Reply(body=_responses_reply())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _chat_peer(request: Request) -> Reply:
    if request.target.endswith("/chat/completions"):
        return Reply(body=_chat_reply())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _google_peer(request: Request) -> Reply:
    if request.target.endswith(":countTokens"):
        return Reply(body=json.dumps({"totalTokens": _GOOGLE_COUNT}).encode())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _anthropic_peer(request: Request) -> Reply:
    if request.target.endswith("/v1/messages/count_tokens"):
        return Reply(body=json.dumps({"input_tokens": _ANTHROPIC_COUNT}).encode())
    if request.target.endswith("/v1/messages"):
        return Reply(body=_anthropic_reply())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _bedrock_peer(request: Request) -> Reply:
    if request.target.endswith("/count-tokens"):
        return Reply(body=json.dumps({"inputTokens": _BEDROCK_COUNT}).encode())
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _openai_model(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(model="openai/gpt-5.4-mini", api_base=wire.url, api_key="synthetic-openai-key", **parameters)


def _mistral_model(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(
        model="mistral/mistral-small-latest", api_base=wire.url, api_key="synthetic-mistral-key", **parameters
    )


def _anthropic_model(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(
        model="anthropic/claude-sonnet-4-6", api_base=wire.url, api_key="synthetic-anthropic-key", **parameters
    )


def _google_model(gateway: Gateway, scenario: Scenario, wire: Wire, provider: str, **parameters: JsonValue) -> str:
    if provider == "gemini":
        return scenario.model(model=f"gemini/{_GEMINI}", api_base=wire.url, api_key="scripted-gemini-key", **parameters)
    return scenario.model(
        model=f"vertex_ai/{_GEMINI}",
        api_base=wire.url,
        api_key=None,
        vertex_project=_VERTEX_PROJECT,
        vertex_location="us-east5",
        vertex_credentials=service_account_json(_VERTEX_PROJECT, gateway.upstream_url.rstrip("/")),
        **parameters,
    )


def _json(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def _only(requests: Sequence[Request], suffix: str) -> dict[str, JsonValue]:
    matching: Final = tuple(request for request in requests if request.target.endswith(suffix))
    assert len(matching) == 1, [request.target for request in requests]
    return _json(matching[0])


def _function_call_output(body: Mapping[str, JsonValue]) -> JsonValue:
    outputs: Final = tuple(
        item["output"]
        for item in _JSON_OBJECTS.validate_python(body["input"])
        if item.get("type") == "function_call_output"
    )
    assert len(outputs) == 1, body
    return outputs[0]


def _tool_message_text(body: Mapping[str, JsonValue]) -> str:
    contents: Final = tuple(
        message.get("content")
        for message in _JSON_OBJECTS.validate_python(body["messages"])
        if message.get("role") == "tool"
    )
    assert len(contents) == 1, body
    content: Final = contents[0]
    if isinstance(content, str):
        return content
    return "".join(str(part["text"]) for part in _JSON_OBJECTS.validate_python(content))


def _anthropic_error_message(status: int, text: str, payload: bytes) -> str:
    assert status == 400, text
    error: Final = _JSON_OBJECT.validate_json(payload)
    assert error["type"] == "error", text
    detail: Final = _JSON_OBJECT.validate_python(error["error"])
    assert detail["type"] == "invalid_request_error", text
    return str(detail["message"])


def _assert_field_rejection(gateway: Gateway, path: str, body: Mapping[str, JsonValue], **headers: str) -> None:
    response: Final = gateway.request("POST", path, body, headers=headers)
    assert _FIELD_ERROR in _anthropic_error_message(response.status_code, response.text, response.content), (
        response.text
    )


def _sdk(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=gateway.key, base_url=str(gateway.client.base_url), max_retries=0, http_client=gateway.client
    )


def _assert_sdk_bad_request(raised: pytest.ExceptionInfo[anthropic.BadRequestError]) -> None:
    error: Final = raised.value
    assert error.status_code == 400, error
    assert _FIELD_ERROR in _anthropic_error_message(error.status_code, error.message, error.response.content), error


def test_openai_rejects_a_non_string_tool_result_text_on_both_routes(gateway: Gateway) -> None:
    client: Final = _sdk(gateway)
    with wire_server(_openai_peer) as wire, gateway.scenario() as scenario:
        model: Final = _openai_model(scenario, wire)
        with pytest.raises(anthropic.BadRequestError) as created:
            client.messages.create(model=model, max_tokens=64, messages=_NULL_TEXT, tools=_TOOLS)
        _assert_sdk_bad_request(created)
        with pytest.raises(anthropic.BadRequestError) as streamed:
            client.messages.create(model=model, max_tokens=64, messages=_NULL_TEXT, tools=_TOOLS, stream=True)
        _assert_sdk_bad_request(streamed)
        with pytest.raises(anthropic.BadRequestError) as counted:
            client.messages.count_tokens(model=model, messages=_NULL_TEXT, tools=_TOOLS)
        _assert_sdk_bad_request(counted)
        assert wire.drain() == ()


_DROP_SOURCES: Final = MappingProxyType(
    {
        "deployment": (MappingProxyType({"drop_params": True}), MappingProxyType({}), MappingProxyType({})),
        "body": (MappingProxyType({}), MappingProxyType({"drop_params": True}), MappingProxyType({})),
        "claude_code": (MappingProxyType({}), MappingProxyType({}), _CLAUDE_CODE),
    }
)


@pytest.mark.parametrize("source", sorted(_DROP_SOURCES))
def test_openai_drop_params_drops_the_malformed_text(gateway: Gateway, source: str) -> None:
    deployment, extra, headers = _DROP_SOURCES[source]
    with wire_server(_openai_peer) as wire, gateway.scenario() as scenario:
        model: Final = _openai_model(scenario, wire, **deployment)
        body: Final = _body(model, _NUMBER_THEN_WEATHER, **extra)
        created: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 64}, headers=headers)
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body, headers=headers)
        assert created.status_code == 200, created.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _OPENAI_COUNT, counted.text
        requests: Final = wire.drain()
        assert _function_call_output(_only(requests, "/responses")) == _WEATHER
        assert _function_call_output(_only(requests, "/responses/input_tokens")) == _WEATHER


def test_explicit_drop_params_false_wins_on_both_routes(gateway: Gateway) -> None:
    with wire_server(_openai_peer) as wire, gateway.scenario() as scenario:
        model: Final = _openai_model(scenario, wire, drop_params=True)
        body: Final = _body(model, _NULL_TEXT, drop_params=False)
        _assert_field_rejection(gateway, "/v1/messages", {**body, "max_tokens": 64}, **_CLAUDE_CODE)
        _assert_field_rejection(gateway, "/v1/messages/count_tokens", body, **_CLAUDE_CODE)
        assert wire.drain() == ()


def test_chat_bridge_and_local_count_reject_a_non_string_tool_result_text(gateway: Gateway) -> None:
    with wire_server(_chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = _mistral_model(scenario, wire)
        body: Final = _body(model, _NULL_TEXT)
        _assert_field_rejection(gateway, "/v1/messages", {**body, "max_tokens": 64})
        _assert_field_rejection(gateway, "/v1/messages", {**body, "max_tokens": 64, "stream": True})
        _assert_field_rejection(gateway, "/v1/messages/count_tokens", body)
        assert wire.drain() == ()


@pytest.mark.parametrize("source", ("deployment", "body"))
def test_chat_bridge_drop_params_drops_the_malformed_text(gateway: Gateway, source: str) -> None:
    deployment, extra, _ = _DROP_SOURCES[source]
    with wire_server(_chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = _mistral_model(scenario, wire, **deployment)
        body: Final = _body(model, _NUMBER_THEN_WEATHER, **extra)
        created: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 64})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert created.status_code == 200, created.text
        assert counted.status_code == 200, counted.text
        assert _tool_message_text(_only(wire.drain(), "/chat/completions")) == _WEATHER


def test_count_without_a_deployment_rejects_and_honors_body_drop_params(gateway: Gateway) -> None:
    model: Final = f"integration-unrouted-{uuid.uuid4().hex}"
    _assert_field_rejection(gateway, "/v1/messages/count_tokens", _body(model, _NULL_TEXT))
    dropped: Final = gateway.request("POST", "/v1/messages/count_tokens", _body(model, _NULL_TEXT, drop_params=True))
    assert dropped.status_code == 200, dropped.text
    tokens: Final = _JSON_OBJECT.validate_json(dropped.content)["input_tokens"]
    assert isinstance(tokens, int) and tokens > 0, dropped.text


def _counted_contents(body: Mapping[str, JsonValue]) -> JsonValue:
    if "contents" in body:
        return body["contents"]
    return _JSON_OBJECT.validate_python(body["generateContentRequest"])["contents"]


@pytest.mark.parametrize("provider", ("gemini", "vertex_ai"))
def test_google_count_rejects_and_drops_a_non_string_tool_result_text(gateway: Gateway, provider: str) -> None:
    with wire_server(_google_peer) as wire, gateway.scenario() as scenario:
        strict: Final = _google_model(gateway, scenario, wire, provider)
        dropping: Final = _google_model(gateway, scenario, wire, provider, drop_params=True)
        _assert_field_rejection(gateway, "/v1/messages/count_tokens", _body(strict, _NULL_TEXT))
        assert tuple(request for request in wire.drain() if request.target.endswith(":countTokens")) == ()
        dropped: Final = gateway.request("POST", "/v1/messages/count_tokens", _body(dropping, _NUMBER_THEN_WEATHER))
        assert dropped.status_code == 200, dropped.text
        dropped_body: Final = _only(wire.drain(), ":countTokens")
        clean: Final = gateway.request("POST", "/v1/messages/count_tokens", _body(dropping, _WEATHER_ONLY))
        assert clean.status_code == 200, clean.text
        assert _JSON_OBJECT.validate_json(dropped.content)["input_tokens"] == _GOOGLE_COUNT, dropped.text
        assert _counted_contents(dropped_body) == _counted_contents(_only(wire.drain(), ":countTokens"))


def _forwarded_tool_result(body: Mapping[str, JsonValue]) -> JsonValue:
    last: Final = _JSON_OBJECTS.validate_python(body["messages"])[-1]
    return _JSON_OBJECTS.validate_python(last["content"])[0]["content"]


def test_native_anthropic_forwards_a_null_tool_result_text_unchanged(gateway: Gateway) -> None:
    with wire_server(_anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_model(scenario, wire)
        body: Final = _body(model, _NULL_TEXT)
        created: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 64}, headers=_CLAUDE_CODE)
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body, headers=_CLAUDE_CODE)
        assert created.status_code == 200, created.text
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _ANTHROPIC_COUNT, counted.text
        requests: Final = wire.drain()
        assert _forwarded_tool_result(_only(requests, "/v1/messages")) == [{"type": "text", "text": None}]
        assert _forwarded_tool_result(_only(requests, "/v1/messages/count_tokens")) == [{"type": "text", "text": None}]


def test_bedrock_count_with_a_claude_code_user_agent_still_counts(gateway: Gateway) -> None:
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/anthropic.claude-sonnet-4-6",
            api_base=wire.url,
            api_key=None,
            aws_access_key_id="AKIAINTEGRATIONTOOLRESULT",
            aws_secret_access_key="integration-tool-result-secret",
            aws_region_name="us-east-1",
        )
        counted: Final = gateway.request(
            "POST", "/v1/messages/count_tokens", _body(model, _WEATHER_ONLY), headers=_CLAUDE_CODE
        )
        assert counted.status_code == 200, counted.text
        assert _JSON_OBJECT.validate_json(counted.content)["input_tokens"] == _BEDROCK_COUNT, counted.text
        assert len(tuple(request for request in wire.drain() if request.target.endswith("/count-tokens"))) == 1


@pytest.mark.parametrize("provider", ("openai", "mistral"))
def test_valid_tool_result_text_reaches_the_upstream(gateway: Gateway, provider: str) -> None:
    peer: Final = _openai_peer if provider == "openai" else _chat_peer
    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = (_openai_model if provider == "openai" else _mistral_model)(scenario, wire)
        body: Final = _body(model, _WEATHER_THEN_MISSING)
        created: Final = gateway.request("POST", "/v1/messages", {**body, "max_tokens": 64})
        counted: Final = gateway.request("POST", "/v1/messages/count_tokens", body)
        assert created.status_code == 200, created.text
        assert counted.status_code == 200, counted.text
        requests: Final = wire.drain()
        if provider == "openai":
            assert _function_call_output(_only(requests, "/responses")) == f"{_WEATHER}\n"
            assert _function_call_output(_only(requests, "/responses/input_tokens")) == f"{_WEATHER}\n"
        else:
            assert _tool_message_text(_only(requests, "/chat/completions")) == _WEATHER


def _redacted(content: JsonValue) -> JsonValue:
    if isinstance(content, str):
        return content.replace(_CITY, _REDACTED_CITY)
    if isinstance(content, list):
        return [_redacted(part) for part in content]
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return {**content, "text": _redacted(content["text"])}
    return content


def _redacted_row(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**row, "content": _redacted(row["content"])} if "content" in row else dict(row)


def _guardrail_peer(request: Request) -> Reply:
    if request.target.startswith("/none/"):
        return Reply(body=json.dumps({"action": "NONE"}).encode())
    rows: Final = _JSON_OBJECTS.validate_python(_json(request)["structured_messages"])
    verdict: Final = {"action": "GUARDRAIL_INTERVENED", "structured_messages": [_redacted_row(row) for row in rows]}
    return Reply(body=json.dumps(verdict).encode())


def _delete_guardrail(gateway: Gateway, identity: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{identity}")
    assert deleted.status_code == 200, deleted.text


def _guardrail(gateway: Gateway, scenario: Scenario, wire: Wire, mode: str) -> str:
    name: Final = f"tool-result-{mode}-{uuid.uuid4().hex}"
    params: Final[dict[str, JsonValue]] = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": False,
        "api_base": f"{wire.url}/{mode}",
        "api_key": "synthetic-guardrail-key",
    }
    created: Final = gateway.post("/guardrails", {"guardrail": {"guardrail_name": name, "litellm_params": params}})
    scenario.cleanups.callback(_delete_guardrail, gateway, string_value(created["guardrail_id"]))
    return name


def _forwarded_tool_result_texts(body: Mapping[str, JsonValue]) -> tuple[JsonValue, ...]:
    content: Final = _forwarded_tool_result(body)
    if isinstance(content, str):
        return (content,)
    return tuple(block.get("text") for block in _JSON_OBJECTS.validate_python(content) if block.get("type") == "text")


def test_rewriting_guardrail_rejects_a_non_string_tool_result_text(gateway: Gateway) -> None:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        model: Final = _anthropic_model(scenario, provider)
        body: Final = _body(model, _NULL_TEXT, max_tokens=64, guardrails=[_guardrail(gateway, scenario, guard, "rows")])
        _assert_field_rejection(gateway, "/v1/messages", body)
        _assert_field_rejection(gateway, "/v1/messages", {**body, "stream": True})
        assert provider.drain() == ()


@pytest.mark.parametrize("source", ("body", "claude_code"))
def test_rewriting_guardrail_drops_the_malformed_text_under_drop_params(gateway: Gateway, source: str) -> None:
    deployment, extra, headers = _DROP_SOURCES[source]
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        model: Final = _anthropic_model(scenario, provider, **deployment)
        guardrail: Final = _guardrail(gateway, scenario, guard, "rows")
        body: Final = _body(model, _NUMBER_THEN_WEATHER, max_tokens=64, guardrails=[guardrail], **extra)
        created: Final = gateway.request("POST", "/v1/messages", body, headers=headers)
        assert created.status_code == 200, created.text
        forwarded: Final = _only(provider.drain(), "/v1/messages")
        first_turn: Final = json.dumps(_JSON_OBJECTS.validate_python(forwarded["messages"])[0])
        assert _REDACTED_CITY in first_turn and _CITY not in first_turn, first_turn
        assert _forwarded_tool_result_texts(forwarded) == (_WEATHER,), forwarded


def test_guardrail_that_rewrites_nothing_forwards_a_null_tool_result_text_unchanged(gateway: Gateway) -> None:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        model: Final = _anthropic_model(scenario, provider)
        guardrail: Final = _guardrail(gateway, scenario, guard, "none")
        created: Final = gateway.request(
            "POST", "/v1/messages", _body(model, _NULL_TEXT, max_tokens=64, guardrails=[guardrail])
        )
        assert created.status_code == 200, created.text
        assert len(guard.drain()) == 1
        assert _forwarded_tool_result(_only(provider.drain(), "/v1/messages")) == [{"type": "text", "text": None}]


def _forwarded_chat_turns(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return _JSON_OBJECTS.validate_python(_only(wire.drain(), "/chat/completions")["messages"])


def test_mistral_receives_the_tool_reply_whose_only_text_drop_params_dropped(gateway: Gateway) -> None:
    with wire_server(_chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = _mistral_model(scenario, wire)
        created: Final = gateway.request(
            "POST", "/v1/messages", _body(model, _NULL_TEXT, max_tokens=64, drop_params=True)
        )
        assert created.status_code == 200, created.text
        turns: Final = _forwarded_chat_turns(wire)
        assert [turn["role"] for turn in turns] == ["user", "assistant", "tool"], turns
        calls: Final = _JSON_OBJECTS.validate_python(turns[1]["tool_calls"])
        assert [call["id"] for call in calls] == [turns[2]["tool_call_id"]], turns
        assert _tool_message_text({"messages": list(turns)}) == "", turns


def test_mistral_chat_completions_drops_only_empty_assistant_messages(gateway: Gateway) -> None:
    ask: Final = f"What is the weather in {_CITY}? {uuid.uuid4().hex}"
    call: Final[dict[str, JsonValue]] = {
        "id": "call_weather",
        "type": "function",
        "function": {"name": "get_weather", "arguments": json.dumps({"city": _CITY})},
    }
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ask},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_weather", "content": ""},
        {"role": "user", "content": ""},
    ]
    with wire_server(_chat_peer) as wire, gateway.scenario() as scenario:
        model: Final = _mistral_model(scenario, wire)
        response: Final = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": messages})
        assert response.status_code == 200, response.text
        turns: Final = _forwarded_chat_turns(wire)
        assert [(turn["role"], turn.get("content")) for turn in turns] == [
            ("system", ""),
            ("user", ask),
            ("assistant", None),
            ("tool", ""),
            ("user", ""),
        ], turns
        assert turns[3]["tool_call_id"] == "call_weather", turns
