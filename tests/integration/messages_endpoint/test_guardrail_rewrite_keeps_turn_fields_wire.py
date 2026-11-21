import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
import pytest
from integration._support.anthropic_sse import message_json, message_stream, stream_reply
from integration._support.client import Gateway, Scenario, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
_CLAUDE: Final = "claude-opus-5-5"
_BETAS: Final = "per-turn-control-2026-07-01,mid-conversation-system-clear-at-2026-08-21"
_ERRORED: Final = "cat: a.json: No such file or directory"
_SYSTEM_TEXT: Final = "# Environment ..."
_REDACTED: Final = "[REDACTED]"
_KEPT_SYSTEM: Final = "keep this reminder"
_DROPPED_SYSTEM: Final = "drop this reminder"
_ABSENT: Final = "<absent>"
_MEDIUM: Final[dict[str, JsonValue]] = {"effort": "medium"}
_LOW: Final[dict[str, JsonValue]] = {"effort": "low"}
_HIGH: Final[dict[str, JsonValue]] = {"effort": "high"}
_NO_FIELDS: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_OUTPUT_CONFIG: Final[Mapping[str, JsonValue]] = MappingProxyType({"output_config": _MEDIUM})
_SENT_FALSE: Final[Mapping[str, JsonValue]] = MappingProxyType({"is_error": False})
_TOOLS: Final[list[JsonValue]] = [
    {
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }
]
_Rows: TypeAlias = tuple[dict[str, JsonValue], ...]


def _tool_use(tool_use_id: str, text: str, command: str) -> dict[str, JsonValue]:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": command}},
        ],
    }


def _tool_result(tool_use_id: str, content: str, flags: Mapping[str, JsonValue] = _NO_FIELDS) -> dict[str, JsonValue]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, **flags}


def _system(text: str, fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {"role": "system", "content": text, **fields}


def _repro(
    system_fields: Mapping[str, JsonValue] = _OUTPUT_CONFIG, second_flags: Mapping[str, JsonValue] = _NO_FIELDS
) -> list[JsonValue]:
    return [
        {"role": "user", "content": "inspect files"},
        _system(_SYSTEM_TEXT, system_fields),
        _tool_use("toolu_01", "Running.", "cat a.json"),
        {"role": "user", "content": [_tool_result("toolu_01", _ERRORED, {"is_error": True})]},
        _tool_use("toolu_02", "Next.", "echo ok"),
        {
            "role": "user",
            "content": [_tool_result("toolu_02", "ok", second_flags), {"type": "text", "text": "what failed?"}],
        },
    ]


def _not_found(request: Request) -> Reply:
    return Reply(status=404, body=json.dumps({"error": request.target}).encode())


def _anthropic_peer(request: Request) -> Reply:
    if not request.target.endswith("/v1/messages"):
        return _not_found(request)
    message_id: Final = f"msg_{uuid.uuid4().hex}"
    if _JSON_OBJECT.validate_json(request.body).get("stream") is True:
        return stream_reply(message_stream(message_id, _CLAUDE, "Checked."))
    return Reply(body=message_json(message_id, _CLAUDE, "Checked."))


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
                    "content": [{"type": "output_text", "text": "Checked.", "annotations": []}],
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
            "model": "gpt-5.4-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Checked."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        }
    ).encode()


def _openai_peer(request: Request) -> Reply:
    if request.target.endswith("/responses"):
        return Reply(body=_responses_reply())
    if request.target.endswith("/chat/completions"):
        return Reply(body=_chat_reply())
    return _not_found(request)


def _calls_tool(row: Mapping[str, JsonValue], tool_call_id: str) -> bool:
    calls: Final = row.get("tool_calls")
    return isinstance(calls, list) and any(
        call.get("id") == tool_call_id for call in _JSON_OBJECTS.validate_python(calls)
    )


def _answers_tool(row: Mapping[str, JsonValue], tool_call_id: str) -> bool:
    return row.get("role") == "tool" and row.get("tool_call_id") == tool_call_id


def _redacted_row(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if row.get("role") == "system" or _answers_tool(row, "toolu_01"):
        return {**row, "content": _REDACTED}
    return dict(row)


def _without_errored_exchange(rows: _Rows) -> list[dict[str, JsonValue]]:
    return [row for row in rows if not _calls_tool(row, "toolu_01") and not _answers_tool(row, "toolu_01")]


_REWRITES: Final[Mapping[str, Callable[[_Rows], list[dict[str, JsonValue]]]]] = MappingProxyType(
    {
        "echo": lambda rows: [dict(row) for row in rows],
        "redact": lambda rows: [_redacted_row(row) for row in rows],
        "delete-exchange": _without_errored_exchange,
        "delete-system": lambda rows: [row for row in rows if row.get("content") != _DROPPED_SYSTEM],
    }
)


def _guardrail_peer(request: Request) -> Reply:
    mode: Final = request.target.split("/")[1]
    if mode == "none":
        return Reply(body=json.dumps({"action": "NONE"}).encode())
    rows: Final = _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(request.body).get("structured_messages"))
    verdict: Final = {"action": "GUARDRAIL_INTERVENED", "structured_messages": _REWRITES[mode](rows)}
    return Reply(body=json.dumps(verdict).encode())


def _delete_guardrail(gateway: Gateway, identity: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{identity}")
    assert deleted.status_code == 200, deleted.text


def _guardrail(gateway: Gateway, scenario: Scenario, guard: Wire, mode: str, **params: JsonValue) -> str:
    name: Final = f"turn-fields-{mode}-{uuid.uuid4().hex}"
    litellm_params: Final[dict[str, JsonValue]] = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": False,
        "api_base": f"{guard.url}/{mode}",
        "api_key": "synthetic-guardrail-key",
        **params,
    }
    created: Final = gateway.post(
        "/guardrails", {"guardrail": {"guardrail_name": name, "litellm_params": litellm_params}}
    )
    scenario.cleanups.callback(_delete_guardrail, gateway, string_value(created["guardrail_id"]))
    return name


def _claude(scenario: Scenario, provider: Wire) -> str:
    return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=provider.url, api_key="synthetic-anthropic-key")


def _send(
    gateway: Gateway, model: str, messages: list[JsonValue], guardrail: str | None, stream: bool = False
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 64,
        "tools": _TOOLS,
        "messages": messages,
        **({"guardrails": [guardrail]} if guardrail is not None else {}),
        **({"stream": True} if stream else {}),
    }
    response: Final = gateway.request("POST", "/v1/messages", body, headers={"anthropic-beta": _BETAS})
    assert response.status_code == 200, response.text
    return response


def _only(requests: Sequence[Request], *suffixes: str) -> Request:
    matching: Final = tuple(request for request in requests if request.target.endswith(suffixes))
    assert len(matching) == 1, [request.target for request in requests]
    return matching[0]


def _forwarded(provider: Wire) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(_only(provider.drain(), "/v1/messages").body)


def _messages(body: Mapping[str, JsonValue]) -> _Rows:
    return _JSON_OBJECTS.validate_python(body["messages"])


def _blocks(message: Mapping[str, JsonValue]) -> _Rows:
    content: Final = message.get("content")
    return _JSON_OBJECTS.validate_python(content) if isinstance(content, list) else ()


def _text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in _JSON_OBJECTS.validate_python(content))


def _tool_results(body: Mapping[str, JsonValue]) -> tuple[tuple[JsonValue, str, JsonValue], ...]:
    blocks: Final = chain.from_iterable(_blocks(message) for message in _messages(body))
    return tuple(
        (block["tool_use_id"], _text(block.get("content", "")), block.get("is_error", _ABSENT))
        for block in blocks
        if block.get("type") == "tool_result"
    )


def _system_rows(body: Mapping[str, JsonValue], key: str) -> tuple[tuple[str, JsonValue], ...]:
    return tuple(
        (_text(row["content"]), row.get(key, _ABSENT)) for row in _messages(body) if row.get("role") == "system"
    )


def _rewritten(
    gateway: Gateway, mode: str, messages: list[JsonValue], *, stream: bool = False, **params: JsonValue
) -> dict[str, JsonValue]:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        guardrail: Final = _guardrail(gateway, scenario, guard, mode, **params)
        _send(gateway, _claude(scenario, provider), messages, guardrail, stream)
        assert len(guard.drain()) == 1
        return _forwarded(provider)


def test_echo_keeps_is_error_true_on_the_errored_tool_result_regression(gateway: Gateway) -> None:
    results: Final = _tool_results(_rewritten(gateway, "echo", _repro()))
    assert results[0] == ("toolu_01", _ERRORED, True), results


def test_echo_keeps_is_error_false_when_the_client_sent_false_regression(gateway: Gateway) -> None:
    results: Final = _tool_results(_rewritten(gateway, "echo", _repro(second_flags=_SENT_FALSE)))
    assert results == (("toolu_01", _ERRORED, True), ("toolu_02", "ok", False)), results


def test_echo_leaves_a_tool_result_sent_without_is_error_without_it_guard(gateway: Gateway) -> None:
    results: Final = _tool_results(_rewritten(gateway, "echo", _repro()))
    assert results[1] == ("toolu_02", "ok", _ABSENT), results


@pytest.mark.parametrize(
    ("key", "value"), (("output_config", _MEDIUM), ("clear_at", "next_user_message")), ids=("output_config", "clear_at")
)
def test_echo_keeps_the_extra_key_on_the_system_row_regression(gateway: Gateway, key: str, value: JsonValue) -> None:
    forwarded: Final = _rewritten(gateway, "echo", _repro(system_fields=MappingProxyType({key: value})))
    assert _system_rows(forwarded, key) == ((_SYSTEM_TEXT, value),), forwarded


def test_streamed_echo_keeps_is_error_and_output_config_regression(gateway: Gateway) -> None:
    forwarded: Final = _rewritten(gateway, "echo", _repro(second_flags=_SENT_FALSE), stream=True)
    assert _tool_results(forwarded) == (("toolu_01", _ERRORED, True), ("toolu_02", "ok", False)), forwarded
    assert _system_rows(forwarded, "output_config") == ((_SYSTEM_TEXT, _MEDIUM),), forwarded


def test_redacting_the_errored_tool_result_keeps_is_error_regression(gateway: Gateway) -> None:
    results: Final = _tool_results(_rewritten(gateway, "redact", _repro()))
    assert results == (("toolu_01", _REDACTED, True), ("toolu_02", "ok", _ABSENT)), results


def test_redacting_the_system_row_keeps_output_config_regression(gateway: Gateway) -> None:
    forwarded: Final = _rewritten(gateway, "redact", _repro())
    assert _system_rows(forwarded, "output_config") == ((_REDACTED, _MEDIUM),), forwarded


def test_deleting_the_errored_exchange_keeps_is_error_false_on_the_survivor_regression(gateway: Gateway) -> None:
    results: Final = _tool_results(_rewritten(gateway, "delete-exchange", _repro(second_flags=_SENT_FALSE)))
    assert results == (("toolu_02", "ok", False),), results


def test_echo_keeps_is_error_order_for_a_repeated_tool_use_id_regression(gateway: Gateway) -> None:
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": "read both files"},
        _tool_use("toolu_dup", "Running.", "cat a.json"),
        {"role": "user", "content": [_tool_result("toolu_dup", _ERRORED, {"is_error": True})]},
        _tool_use("toolu_dup", "Again.", "cat b.json"),
        {
            "role": "user",
            "content": [_tool_result("toolu_dup", "{}", _SENT_FALSE), {"type": "text", "text": "what failed?"}],
        },
    ]
    results: Final = _tool_results(_rewritten(gateway, "echo", messages))
    assert results == (("toolu_dup", _ERRORED, True), ("toolu_dup", "{}", False)), results


def test_echo_keeps_output_config_on_a_system_row_moved_out_of_a_tool_exchange_regression(gateway: Gateway) -> None:
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": "inspect files"},
        _tool_use("toolu_01", "Running.", "cat a.json"),
        _system(_SYSTEM_TEXT, _OUTPUT_CONFIG),
        {
            "role": "user",
            "content": [
                _tool_result("toolu_01", _ERRORED, {"is_error": True}),
                {"type": "text", "text": "what failed?"},
            ],
        },
    ]
    forwarded: Final = _rewritten(gateway, "echo", messages)
    assert _system_rows(forwarded, "output_config") == ((_SYSTEM_TEXT, _MEDIUM),), forwarded


def test_deleted_system_row_output_config_is_not_carried_to_another_row_guard(gateway: Gateway) -> None:
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": "inspect files"},
        _system(_KEPT_SYSTEM, {"output_config": _LOW}),
        {"role": "assistant", "content": "Looking."},
        _system(_DROPPED_SYSTEM, {"output_config": _HIGH}),
        {"role": "user", "content": "what failed?"},
    ]
    rows: Final = _system_rows(_rewritten(gateway, "delete-system", messages), "output_config")
    assert [text for text, _ in rows] == [_KEPT_SYSTEM], rows
    assert all(output_config != _HIGH for _, output_config in rows), rows


def test_guardrail_peer_receives_neither_is_error_nor_output_config_guard(gateway: Gateway) -> None:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        guardrail: Final = _guardrail(gateway, scenario, guard, "echo")
        _send(gateway, _claude(scenario, provider), _repro(second_flags=_SENT_FALSE), guardrail)
        checked: Final = _only(guard.drain(), "/beta/litellm_basic_guardrail_api").body
        forwarded: Final = _forwarded(provider)
    assert _ERRORED.encode() in checked and _SYSTEM_TEXT.encode() in checked, checked
    assert b'"is_error"' not in checked and b'"output_config"' not in checked, checked
    assert [tool_use_id for tool_use_id, _, _ in _tool_results(forwarded)] == ["toolu_01", "toolu_02"], forwarded


@pytest.mark.parametrize("mode", (None, "none"), ids=("no-guardrail", "guardrail-returns-none"))
def test_untouched_request_forwards_is_error_and_output_config_as_sent_guard(
    gateway: Gateway, mode: str | None
) -> None:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        guardrail: Final = None if mode is None else _guardrail(gateway, scenario, guard, mode)
        _send(gateway, _claude(scenario, provider), _repro(second_flags=_SENT_FALSE), guardrail)
        checks: Final = len(guard.drain())
        forwarded: Final = _forwarded(provider)
    assert checks == (0 if mode is None else 1)
    assert _tool_results(forwarded) == (("toolu_01", _ERRORED, True), ("toolu_02", "ok", False)), forwarded
    assert _system_rows(forwarded, "output_config") == ((_SYSTEM_TEXT, _MEDIUM),), forwarded


def test_openai_deployment_behind_an_echo_guardrail_receives_no_is_error_key_guard(gateway: Gateway) -> None:
    with (
        wire_server(_openai_peer) as provider,
        wire_server(_guardrail_peer) as guard,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="openai/gpt-5.4-mini", api_base=provider.url, api_key="synthetic-openai-key"
        )
        _send(gateway, model, _repro(second_flags=_SENT_FALSE), _guardrail(gateway, scenario, guard, "echo"))
        forwarded: Final = _only(provider.drain(), "/responses", "/chat/completions").body
    assert _ERRORED.encode() in forwarded, forwarded
    assert b'"is_error"' not in forwarded, forwarded


def test_tool_results_only_guardrail_redaction_keeps_is_error_and_output_config_regression(gateway: Gateway) -> None:
    forwarded: Final = _rewritten(gateway, "redact", _repro(), scan_only_tool_results=True)
    assert _tool_results(forwarded) == (("toolu_01", _REDACTED, True), ("toolu_02", "ok", _ABSENT)), forwarded
    assert _system_rows(forwarded, "output_config") == ((_SYSTEM_TEXT, _MEDIUM),), forwarded
