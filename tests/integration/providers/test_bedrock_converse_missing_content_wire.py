import asyncio
import base64
import json
import os
import re
import signal
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_if_encrypted_with
from pydantic import JsonValue, TypeAdapter

_MODEL_ID: Final = "anthropic.claude-3-haiku-20240307-v1:0"
_CONVERSE_MODEL: Final = f"bedrock/converse/{_MODEL_ID}"
_INVOKE_MODEL: Final = f"bedrock/invoke/{_MODEL_ID}"
_CONVERSE_TARGET: Final = f"/model/{_MODEL_ID}/converse"
_STREAM_TARGET: Final = f"/model/{_MODEL_ID}/converse-stream"
_INVOKE_TARGET: Final = f"/model/{_MODEL_ID}/invoke"
_ANSWER: Final = "bedrock missing content control"
_RESPONSE: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": _ANSWER}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }
).encode()
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_STREAM_EVENTS: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"delta": {"text": _ANSWER}, "contentBlockIndex": 0}),
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}}),
)
_STREAM_BYTES: Final = b"".join(_aws_event_frame(kind, payload, "sc", "u") for kind, payload in _STREAM_EVENTS)
_DEFAULT_CONTINUE: Final = "Please continue."
_DEPLOYMENT_CONTINUE: Final = "Deployment says continue."
_DEPLOYMENT_CONTINUE_MESSAGE: Final[dict[str, JsonValue]] = {"role": "user", "content": _DEPLOYMENT_CONTINUE}
_NO_NON_SYSTEM_MESSAGE: Final = "bedrock requires at least one non-system message"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_CALL_INDEX: Final = re.compile(r"call-[0-9a-f]{32}-(\d+)")
_QUESTION: Final = "What is the capital of France?"
_ANSWERED: Final = "Paris."
_FOLLOW_UP: Final = "And the capital of Spain?"
_QUESTION_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": _QUESTION}
_ANSWERED_TURN: Final[dict[str, JsonValue]] = {"role": "assistant", "content": _ANSWERED}
_FOLLOW_UP_TURN: Final[dict[str, JsonValue]] = {"role": "user", "content": _FOLLOW_UP}
_NO_CONTENT_USER: Final[dict[str, JsonValue]] = {"role": "user"}
_NULL_CONTENT_USER: Final[dict[str, JsonValue]] = {"role": "user", "content": None}
_EMPTY_CONTENT_USER: Final[dict[str, JsonValue]] = {"role": "user", "content": ""}
_NO_CONTENT_SYSTEM: Final[dict[str, JsonValue]] = {"role": "system"}
_NULL_CONTENT_SYSTEM: Final[dict[str, JsonValue]] = {"role": "system", "content": None}
_NO_CONTENT_ASSISTANT: Final[dict[str, JsonValue]] = {"role": "assistant"}
_TOOL_CALL_TURN: Final[dict[str, JsonValue]] = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'}}
    ],
}
_NO_CONTENT_TOOL: Final[dict[str, JsonValue]] = {"role": "tool", "tool_call_id": "call_1"}
_TOOLS: Final[tuple[dict[str, JsonValue], ...]] = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
)
_CONVERSE_QUESTION: Final[dict[str, JsonValue]] = {"role": "user", "content": [{"text": _QUESTION}]}
_CONVERSE_ANSWERED: Final[dict[str, JsonValue]] = {"role": "assistant", "content": [{"text": _ANSWERED}]}
_CONVERSE_FOLLOW_UP: Final[dict[str, JsonValue]] = {"role": "user", "content": [{"text": _FOLLOW_UP}]}
_CONVERSE_TOOL_USE: Final[dict[str, JsonValue]] = {
    "role": "assistant",
    "content": [{"toolUse": {"toolUseId": "call_1", "name": "get_weather", "input": {"city": "Boston"}}}],
}
_CONVERSE_EMPTY_TOOL_RESULT: Final[dict[str, JsonValue]] = {
    "role": "user",
    "content": [{"toolResult": {"toolUseId": "call_1", "content": []}}],
}
_NEUTRALIZED_TOOL_CALL: Final[dict[str, JsonValue]] = {
    "role": "assistant",
    "content": [{"text": '[tool call call_1: get_weather({"city": "Boston"})]'}],
}
_NEUTRALIZED_TOOL_RESULT: Final[dict[str, JsonValue]] = {
    "role": "user",
    "content": [{"text": "[tool result for call_1: <non-text tool result omitted>]"}],
}
_EXTRA: Final[dict[str, JsonValue]] = {"num_retries": 0, "cache": {"no-cache": True}}
_SIGNING_KEY: Final = os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt")
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_PLAIN: Final = "bedrock-missing-content-plain"
_CONTINUE: Final = "bedrock-missing-content-continue"

Endpoint = Literal["chat", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    user: str
    index: int


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _ModifyParamsCell:
    row: str
    model: str
    messages: tuple[dict[str, JsonValue], ...]
    expected: tuple[dict[str, JsonValue], ...]
    extra: Mapping[str, JsonValue] = MappingProxyType({})


def _continue_turn(text: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": [{"text": text}]}


_MODIFY_PARAMS_CELLS: Final = (
    _ModifyParamsCell("r11", _PLAIN, (_NO_CONTENT_USER,), (_continue_turn(_DEFAULT_CONTINUE),)),
    _ModifyParamsCell(
        "r12",
        _PLAIN,
        (_QUESTION_TURN, _ANSWERED_TURN, _NULL_CONTENT_USER),
        (_CONVERSE_QUESTION, _CONVERSE_ANSWERED, _continue_turn(_DEFAULT_CONTINUE)),
    ),
    _ModifyParamsCell(
        "r13",
        _PLAIN,
        (_QUESTION_TURN, _TOOL_CALL_TURN, _NO_CONTENT_TOOL),
        (_CONVERSE_QUESTION, _CONVERSE_TOOL_USE, _CONVERSE_EMPTY_TOOL_RESULT),
        MappingProxyType({"tools": list(_TOOLS)}),
    ),
    _ModifyParamsCell("r14", _PLAIN, (_NO_CONTENT_SYSTEM, _QUESTION_TURN), (_CONVERSE_QUESTION,)),
    _ModifyParamsCell("r15", _CONTINUE, (_NO_CONTENT_USER,), (_continue_turn(_DEPLOYMENT_CONTINUE),)),
)


def _converse_peer(request: Request) -> Reply:
    if unquote(request.target) == _STREAM_TARGET:
        return Reply(body=_STREAM_BYTES, content_type=_EVENT_STREAM)
    return Reply(body=_RESPONSE)


def _scripted_error(status: int, message: str) -> Reply:
    return Reply(status=status, body=json.dumps({"message": message}).encode())


def _converse_deployment(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(model=_CONVERSE_MODEL, api_base=wire.url, **_AWS, **extra)


def _auth(gateway: Gateway) -> dict[str, str]:
    return {"Authorization": f"Bearer {gateway.key}"}


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _chat(
    model: str,
    messages: Sequence[Mapping[str, JsonValue]],
    *,
    stream: bool = False,
    cached: bool = False,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": 16,
        "stream": stream,
        "num_retries": 0,
        **({} if cached else {"cache": {"no-cache": True}}),
        **extra,
    }


def _post_chat(gateway: Gateway, body: Mapping[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", body)


def _chat_answer(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["choices"][0]["message"]["content"] == _ANSWER, response.text
    return body["id"]


def _only_received(wire: Wire) -> tuple[str, dict[str, JsonValue]]:
    (request,) = wire.drain()
    return unquote(request.target), json.loads(request.body)


def _sse_payloads(lines: Iterable[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(line[6:]) for line in lines if line.startswith("data: ") and line != "data: [DONE]")


def _stream_lines(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    with gateway.client.stream("POST", path, json=body, headers=_auth(gateway)) as response:
        lines: Final = tuple(line for line in response.iter_lines() if line)
        status_code: Final = response.status_code
    assert status_code == 200, "\n".join(lines)
    return lines


def _chat_stream_text(chunks: Iterable[dict[str, JsonValue]]) -> str:
    return "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks if chunk["choices"])


def _spend_row(request_id: str) -> dict[str, JsonValue]:
    (row,) = eventually(
        lambda: read_rows(
            'SELECT request_id, status, call_type, end_user FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (request_id,),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return row


def _success_rows(prefix: str, expected: int) -> tuple[dict[str, JsonValue], ...]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, call_type, end_user FROM "LiteLLM_SpendLogs" WHERE end_user LIKE %s AND status=%s',
            (f"{prefix}%", "success"),
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len(rows) == expected, rows
    assert len({row["request_id"] for row in rows}) == expected, rows
    return tuple(rows)


def _converse_cell(gateway: Gateway, wire: Wire, body: Mapping[str, JsonValue]) -> tuple[str, dict[str, JsonValue]]:
    identity: Final = _chat_answer(_post_chat(gateway, body))
    target, received = _only_received(wire)
    assert target == _CONVERSE_TARGET, target
    assert _spend_row(identity)["status"] == "success"
    return identity, received


def _owned_config(wire: Wire, directory: Path, *, modify_params: bool) -> Path:
    base: Final = _JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    deployment: Final[dict[str, JsonValue]] = {
        "model": _CONVERSE_MODEL,
        "api_base": wire.url,
        "api_key": "integration-provider-key",
        **_AWS,
    }
    config: Final[dict[str, JsonValue]] = {
        **base,
        "model_list": [
            {"model_name": _PLAIN, "litellm_params": deployment},
            {
                "model_name": _CONTINUE,
                "litellm_params": {**deployment, "user_continue_message": _DEPLOYMENT_CONTINUE_MESSAGE},
            },
        ],
        "litellm_settings": {**_JSON.validate_python(base["litellm_settings"]), "modify_params": modify_params},
        "router_settings": {**_JSON.validate_python(base["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / f"bedrock-missing-content-{'modify-params' if modify_params else 'plain'}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def modify_params_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire]]:
    directory: Final = tmp_path_factory.mktemp("bedrock-modify-params")
    with gateway_from_environment() as gateway, wire_server(_converse_peer) as wire:
        config: Final = _owned_config(wire, directory, modify_params=True)
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield owned.gateway, wire


def test_r01_openai_sync_lone_user_without_content_sends_no_converse_block(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        with openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            completion: Final = client.chat.completions.create(
                model=model, messages=[_NO_CONTENT_USER], max_tokens=16, extra_body=_EXTRA
            )
        assert completion.choices[0].message.content == _ANSWER, completion
        target, received = _only_received(wire)
        assert target == _CONVERSE_TARGET, target
        assert received["messages"] == [] and "system" not in received, received
        assert _spend_row(completion.id)["status"] == "success"


async def test_r02_openai_async_user_with_null_content_after_an_assistant_turn_keeps_the_earlier_turns(
    gateway: Gateway,
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        async with openai.AsyncOpenAI(
            base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            completion: Final = await client.chat.completions.create(
                model=model,
                messages=[_QUESTION_TURN, _ANSWERED_TURN, _NULL_CONTENT_USER],
                max_tokens=16,
                extra_body=_EXTRA,
            )
        assert completion.choices[0].message.content == _ANSWER, completion
        target, received = _only_received(wire)
        assert target == _CONVERSE_TARGET, target
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_ANSWERED], received
        assert _spend_row(completion.id)["status"] == "success"


def test_r03_httpx_raw_sse_user_without_content_after_an_assistant_turn_streams_to_done(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        lines: Final = _stream_lines(
            gateway,
            "/v1/chat/completions",
            _chat(model, (_QUESTION_TURN, _ANSWERED_TURN, _NO_CONTENT_USER), stream=True),
        )
        assert lines[-1] == "data: [DONE]", lines
        chunks: Final = _sse_payloads(lines)
        assert _chat_stream_text(chunks) == _ANSWER, lines
        (identity,) = {chunk["id"] for chunk in chunks}
        target, received = _only_received(wire)
        assert target == _STREAM_TARGET, target
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_ANSWERED], received
        assert _spend_row(identity)["status"] == "success"


async def test_r04_openai_async_stream_tool_turn_without_content_sends_an_empty_tool_result(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        async with openai.AsyncOpenAI(
            base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0
        ) as client:
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[_QUESTION_TURN, _TOOL_CALL_TURN, _NO_CONTENT_TOOL],
                tools=list(_TOOLS),
                max_tokens=16,
                stream=True,
                extra_body=_EXTRA,
            )
            chunks: Final = [chunk async for chunk in stream]
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == _ANSWER, chunks
        (identity,) = {chunk.id for chunk in chunks}
        target, received = _only_received(wire)
        assert target == _STREAM_TARGET, target
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_TOOL_USE, _CONVERSE_EMPTY_TOOL_RESULT], received
        assert received["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather", received
        assert _spend_row(identity)["status"] == "success"


@pytest.mark.parametrize("system_turn", (_NO_CONTENT_SYSTEM, _NULL_CONTENT_SYSTEM), ids=("r05", "r06"))
def test_r05_r06_leading_system_without_content_is_dropped(gateway: Gateway, system_turn: dict[str, JsonValue]) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(gateway, wire, _chat(model, (system_turn, _QUESTION_TURN)))
        assert "system" not in received, received
        assert received["messages"] == [_CONVERSE_QUESTION], received


def test_r07_mid_conversation_system_without_content_is_dropped(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(
            gateway, wire, _chat(model, (_QUESTION_TURN, _ANSWERED_TURN, _NO_CONTENT_SYSTEM, _FOLLOW_UP_TURN))
        )
        assert "system" not in received, received
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_ANSWERED, _CONVERSE_FOLLOW_UP], received


def test_r08_assistant_without_content_between_two_user_turns_merges_them(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(
            gateway, wire, _chat(model, (_QUESTION_TURN, _NO_CONTENT_ASSISTANT, _FOLLOW_UP_TURN))
        )
        assert received["messages"] == [{"role": "user", "content": [{"text": _QUESTION}, {"text": _FOLLOW_UP}]}], (
            received
        )


def test_r09_lone_user_with_empty_string_content_sends_no_converse_block(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(gateway, wire, _chat(model, (_EMPTY_CONTENT_USER,)))
        assert received["messages"] == [], received


def test_r10_empty_null_and_missing_content_produce_byte_identical_converse_bodies(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        identities: Final = tuple(
            _chat_answer(_post_chat(gateway, _chat(model, (turn,))))
            for turn in (_EMPTY_CONTENT_USER, _NULL_CONTENT_USER, _NO_CONTENT_USER)
        )
        assert len(set(identities)) == 3, identities
        received: Final = wire.drain()
        assert [unquote(request.target) for request in received] == [_CONVERSE_TARGET] * 3, received
        assert len({request.body for request in received}) == 1, received
        assert json.loads(received[0].body)["messages"] == [], received
        for identity in identities:
            assert _spend_row(identity)["status"] == "success"


@pytest.mark.parametrize("cell", _MODIFY_PARAMS_CELLS, ids=lambda cell: cell.row)
def test_r11_to_r15_modify_params_fills_the_missing_user_content(
    cell: _ModifyParamsCell, modify_params_proxy: tuple[Gateway, Wire]
) -> None:
    gateway, wire = modify_params_proxy
    _, received = _converse_cell(gateway, wire, _chat(cell.model, cell.messages, **cell.extra))
    assert received["messages"] == list(cell.expected), received
    assert "system" not in received, received


def test_r16_deployment_user_continue_message_fills_the_missing_content_without_modify_params(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire, user_continue_message=_DEPLOYMENT_CONTINUE_MESSAGE)
        _, received = _converse_cell(gateway, wire, _chat(model, (_NO_CONTENT_USER,)))
        assert received["messages"] == [_continue_turn(_DEPLOYMENT_CONTINUE)], received


def test_r17_anthropic_sync_lone_user_without_content_is_rejected_before_any_peer_call(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        with anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(anthropic.BadRequestError, match=_NO_NON_SYSTEM_MESSAGE):
                client.messages.create(model=model, max_tokens=16, messages=[_NO_CONTENT_USER], extra_body=_EXTRA)
        assert wire.drain() == ()


async def test_r18_anthropic_async_stream_user_without_content_after_an_assistant_turn_keeps_the_earlier_turns(
    gateway: Gateway,
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        async with anthropic.AsyncAnthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            async with client.messages.stream(
                model=model,
                max_tokens=16,
                messages=[_QUESTION_TURN, _ANSWERED_TURN, _NO_CONTENT_USER],
                extra_body=_EXTRA,
            ) as stream:
                events: Final = [event async for event in stream]
                final: Final = await stream.get_final_message()
        (started,) = tuple(event for event in events if event.type == "message_start")
        assert final.content[0].text == _ANSWER, final
        assert final.id == started.message.id, (final.id, started.message.id)
        target, received = _only_received(wire)
        assert target == _STREAM_TARGET, target
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_ANSWERED], received
        assert _spend_row(final.id)["call_type"] == "anthropic_messages"


def test_r19_anthropic_native_invoke_forwards_the_turn_verbatim_and_relays_the_scripted_400(gateway: Gateway) -> None:
    with (
        wire_server(lambda _: _scripted_error(400, "scripted invoke validation")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=_INVOKE_MODEL, api_key=None, aws_bedrock_runtime_endpoint=wire.url, api_base=wire.url, **_AWS
        )
        with anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(anthropic.BadRequestError, match="scripted invoke validation"):
                client.messages.create(model=model, max_tokens=16, messages=[_NO_CONTENT_USER], extra_body=_EXTRA)
        target, received = _only_received(wire)
        assert target == _INVOKE_TARGET, target
        assert received["messages"] == [_NO_CONTENT_USER], received


def test_r20_responses_lone_input_item_without_content_is_rejected_before_any_peer_call(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": [_NO_CONTENT_USER], **_EXTRA}
        )
        assert response.status_code == 400, response.text
        assert _NO_NON_SYSTEM_MESSAGE in response.text, response.text
        assert wire.drain() == ()


def test_r21_responses_stream_input_item_without_content_after_an_assistant_item_keeps_the_earlier_turns(
    gateway: Gateway,
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        lines: Final = _stream_lines(
            gateway,
            "/v1/responses",
            {
                "model": model,
                "input": [_QUESTION_TURN, _ANSWERED_TURN, _NO_CONTENT_USER],
                "stream": True,
                "user": marker,
                **_EXTRA,
            },
        )
        events: Final = _sse_payloads(lines)
        (completed,) = tuple(event for event in events if event["type"] == "response.completed")
        assert completed["response"]["output"][0]["content"][0]["text"] == _ANSWER, lines
        target, received = _only_received(wire)
        assert target == _STREAM_TARGET, target
        assert received["messages"] == [_CONVERSE_QUESTION, _CONVERSE_ANSWERED], received
        (row,) = _success_rows(marker, 1)
        assert row["call_type"] == "aresponses" and row["end_user"] == marker, row
        assert row["request_id"] == _inner_response_id(str(completed["response"]["id"])), (row, completed)


def test_r22_passthrough_converse_forwards_a_message_without_content_verbatim(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(
            model=f"bedrock/{_MODEL_ID}", api_base=wire.url, aws_bedrock_runtime_endpoint=wire.url, **_AWS
        )
        response: Final = gateway.request(
            "POST", f"/bedrock/model/{deployment}/converse", {"messages": [_NO_CONTENT_USER]}
        )
        assert response.status_code == 200, response.text
        assert response.content == _RESPONSE, response.text
        (request,) = wire.drain()
        assert unquote(request.target) == _CONVERSE_TARGET, request.target
        assert json.loads(request.body)["messages"] == [_NO_CONTENT_USER], request.body


def test_r23_tool_turn_without_content_and_without_tools_is_neutralized_to_text(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(gateway, wire, _chat(model, (_QUESTION_TURN, _TOOL_CALL_TURN, _NO_CONTENT_TOOL)))
        assert received["messages"] == [_CONVERSE_QUESTION, _NEUTRALIZED_TOOL_CALL, _NEUTRALIZED_TOOL_RESULT], received
        assert "toolConfig" not in received, received


def test_s01_lone_user_with_integer_content_errors_before_any_peer_call(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = _post_chat(gateway, _chat(model, ({"role": "user", "content": 42},)))
        assert response.status_code >= 400, response.text
        assert "error" in response.json(), response.text
        assert wire.drain() == ()


def test_s02_lone_user_with_list_content_sends_the_text_block(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(
            gateway, wire, _chat(model, ({"role": "user", "content": [{"type": "text", "text": _QUESTION}]},))
        )
        assert received["messages"] == [_CONVERSE_QUESTION], received


def test_s03_lone_user_with_a_five_kilobyte_string_reaches_the_peer_whole(gateway: Gateway) -> None:
    text: Final = "k" * 5120
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        _, received = _converse_cell(gateway, wire, _chat(model, ({"role": "user", "content": text},)))
        assert received["messages"] == [{"role": "user", "content": [{"text": text}]}], received


def test_s04_duplicate_content_keys_in_the_raw_body_let_the_last_value_win(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        raw: Final = (
            f'{{"model": "{model}", "messages": [{{"role": "user", "content": "first", "content": "second"}}],'
            ' "max_tokens": 16, "num_retries": 0, "cache": {"no-cache": true}}'
        )
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            content=raw.encode(),
            headers={**_auth(gateway), "content-type": "application/json"},
        )
        identity: Final = _chat_answer(response)
        target, received = _only_received(wire)
        assert target == _CONVERSE_TARGET, target
        assert received["messages"] == [{"role": "user", "content": [{"text": "second"}]}], received
        assert _spend_row(identity)["status"] == "success"


def test_s05_unauthenticated_content_less_request_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat(model, (_NO_CONTENT_USER,)), key="sk-integration-bogus"
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == (), response.text


@pytest.mark.parametrize(
    ("peer_status", "message", "expected"),
    (
        (400, "ValidationException: scripted validation", 400),
        (429, "ThrottlingException: scripted throttle", 429),
        (500, "scripted outage", 503),
    ),
    ids=("s06", "s07", "s08"),
)
def test_s06_to_s08_peer_errors_on_a_content_less_turn_reach_the_caller_after_one_attempt(
    gateway: Gateway, peer_status: int, message: str, expected: int
) -> None:
    with wire_server(lambda _: _scripted_error(peer_status, message)) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        response: Final = _post_chat(gateway, _chat(model, (_NO_CONTENT_USER,)))
        assert response.status_code == expected, response.text
        assert message in response.text, response.text
        assert len(wire.drain()) == 1, response.text


def test_s09_unknown_model_with_a_content_less_turn_never_reaches_the_peer(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire:
        response: Final = _post_chat(gateway, _chat(f"integration-missing-{uuid.uuid4().hex}", (_NO_CONTENT_USER,)))
        assert response.status_code in (400, 404), response.text
        assert wire.drain() == (), response.text


def test_s10_a_deployment_continue_message_without_content_adds_no_converse_block(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire, user_continue_message={"role": "user"})
        _, received = _converse_cell(gateway, wire, _chat(model, (_NO_CONTENT_USER,)))
        assert received["messages"] == [], received


def test_s11_a_deployment_continue_message_given_as_a_string_errors_in_the_body_and_leaves_the_proxy_serving(
    gateway: Gateway,
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        broken: Final = _converse_deployment(scenario, wire, user_continue_message=_DEPLOYMENT_CONTINUE)
        healthy: Final = _converse_deployment(scenario, wire)
        response: Final = _post_chat(gateway, _chat(broken, (_NO_CONTENT_USER,)))
        assert response.status_code >= 400, response.text
        assert "error" in response.json(), response.text
        assert wire.drain() == ()
        _, received = _converse_cell(gateway, wire, _chat(healthy, (_QUESTION_TURN,)))
        assert received["messages"] == [_CONVERSE_QUESTION], received


def test_e01_the_same_content_less_request_twice_with_no_cache_hits_the_peer_twice(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        first: Final = _chat_answer(_post_chat(gateway, _chat(model, (_NO_CONTENT_USER,))))
        second: Final = _chat_answer(_post_chat(gateway, _chat(model, (_NO_CONTENT_USER,))))
        assert first != second
        assert len(wire.drain()) == 2
        assert _spend_row(first)["status"] == "success"
        assert _spend_row(second)["status"] == "success"


def test_e02_the_same_content_less_request_twice_is_served_from_the_response_cache(gateway: Gateway) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        body: Final = _chat(model, (_NO_CONTENT_USER,), cached=True)
        first: Final = _chat_answer(_post_chat(gateway, body))
        second: Final = _chat_answer(_post_chat(gateway, body))
        assert first == second
        assert len(wire.drain()) == 1
        assert _spend_row(first)["status"] == "success"
        cache_hits: Final = eventually(
            lambda: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{first}_cache_hit%",)
            ),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        assert len(cache_hits) == 1, cache_hits


def _model_id(gateway: Gateway, name: str) -> str:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    (identity,) = (
        string_value(object_value(object_value(entry)["model_info"])["id"])
        for entry in entries
        if object_value(entry)["model_name"] == name
    )
    return identity


def _content_less_bodies(gateway: Gateway, wire: Wire, model: str, count: int) -> tuple[JsonValue, ...]:
    identities: Final = tuple(
        _chat_answer(_post_chat(gateway, _chat(model, (_NO_CONTENT_USER,)))) for _ in range(count)
    )
    received: Final = wire.drain()
    assert len(received) == len(identities), (identities, received)
    bodies: Final = tuple(_JSON.validate_python(json.loads(request.body))["messages"] for request in received)
    assert all(body in ([], [_continue_turn(_DEPLOYMENT_CONTINUE)]) for body in bodies), bodies
    return bodies


@pytest.mark.timeout(180)
def test_e03_updating_the_deployment_continue_message_under_traffic_never_breaks_a_content_less_turn(
    gateway: Gateway,
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        assert _content_less_bodies(gateway, wire, model, 4) == ([],) * 4
        gateway.post(
            "/model/update",
            {
                "model_info": {"id": _model_id(gateway, model)},
                "litellm_params": {"user_continue_message": _DEPLOYMENT_CONTINUE_MESSAGE},
            },
        )
        settled: Final = eventually(
            lambda: _content_less_bodies(gateway, wire, model, 8),
            lambda bodies: all(body == [_continue_turn(_DEPLOYMENT_CONTINUE)] for body in bodies),
            seconds=90,
        )
        assert len(settled) == 8, settled


@pytest.mark.parametrize(
    ("continue_message", "expected"),
    ((None, []), ({}, [_continue_turn(_DEFAULT_CONTINUE)])),
    ids=("e04", "e05"),
)
def test_e04_e05_a_null_continue_message_means_absent_and_an_empty_one_means_the_default(
    gateway: Gateway, continue_message: JsonValue, expected: list[JsonValue]
) -> None:
    with wire_server(_converse_peer) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire, user_continue_message=continue_message)
        _, received = _converse_cell(gateway, wire, _chat(model, (_NO_CONTENT_USER,)))
        assert received["messages"] == expected, received


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"


def _calls(marker: str, endpoint: Endpoint, stream: bool, indexes: range) -> tuple[_Call, ...]:
    return tuple(_Call(endpoint, stream, f"{marker}-{index}", index) for index in indexes)


def _burst_body(model: str, call: _Call) -> dict[str, JsonValue]:
    turns: Final = ({"role": "user", "content": f"{_QUESTION} {call.user}"}, _ANSWERED_TURN, _NO_CONTENT_USER)
    if call.endpoint == "chat":
        return {**_chat(model, turns, stream=call.stream), "user": call.user}
    return {"model": model, "input": [dict(turn) for turn in turns], "stream": call.stream, "user": call.user, **_EXTRA}


def _call_index(request: Request) -> int:
    found: Final = _CALL_INDEX.search(request.body.decode())
    assert found is not None, request.body
    return int(found.group(1))


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST", _path(call.endpoint), json=_burst_body(model, call), headers={"Authorization": f"Bearer {key}"}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _inner_response_id(identity: str) -> str:
    managed: Final = decrypt_if_encrypted_with(identity.removeprefix("resp_"), _SIGNING_KEY)
    assert managed is not None, identity
    issued: Final = managed.split(";", 1)[0].rsplit("response_id:", 1)[1]
    decoded: Final = base64.b64decode(issued.removeprefix("resp_")).decode()
    return decoded.rsplit("response_id:", 1)[1]


def _served_id(served: _Served) -> str:
    if served.call.stream:
        (identity,) = {chunk["id"] for chunk in _sse_payloads(served.text.splitlines())}
        return identity
    return json.loads(served.text)["id"]


def _spend_row_id(served: _Served) -> str:
    identity: Final = _served_id(served)
    return identity if served.call.endpoint == "chat" else _inner_response_id(identity)


def _answered(served: Iterable[_Served]) -> frozenset[str]:
    ids: Final = tuple(_spend_row_id(item) for item in served)
    assert len(set(ids)) == len(ids), ids
    return frozenset(ids)


def _open_peer_connections(pid: int, peer_url: str) -> int:
    port: Final = urlsplit(peer_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


async def test_c01_a_mixed_burst_of_content_less_calls_survives_a_peer_outage_window(gateway: Gateway) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = (
        *_calls(marker, "chat", False, range(0, 10)),
        *_calls(marker, "chat", True, range(10, 20)),
        *_calls(marker, "responses", False, range(20, 30)),
    )

    def respond(request: Request) -> Reply:
        if _call_index(request) % 3 == 1:
            return _scripted_error(500, "scripted outage")
        return _converse_peer(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse_deployment(scenario, wire)
        served: Final = await _burst(_proxy_url(gateway), gateway.key, model, calls)
        assert len(served) == 30
        failed: Final = tuple(item for item in served if item.call.index % 3 == 1)
        answered: Final = tuple(item for item in served if item.call.index % 3 != 1)
        assert len(failed) == 10 and len(answered) == 20
        for item in failed:
            assert item.status == 503 and "scripted outage" in item.text, (item.call, item.status, item.text)
        for item in answered:
            assert item.status == 200 and _ANSWER in item.text, (item.call, item.status, item.text)
        identities: Final = _answered(answered)
        assert len(identities) == 20
        assert {row["request_id"] for row in _success_rows(marker, 20)} == identities
        assert len(wire.drain()) == 30


@pytest.mark.timeout(180)
async def test_c02_worker_sigkill_mid_burst_leaves_the_sibling_serving_content_less_turns(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    again: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = _calls(marker, "chat", False, range(20))
    release: Final = threading.Event()
    held_indexes: Final[SimpleQueue[int]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_indexes.put(_call_index(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _converse_peer(request)

    with wire_server(held) as wire:
        config: Final = _owned_config(wire, tmp_path, modify_params=False)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(_proxy_url(candidate), candidate.key, _PLAIN, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_indexes.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_peer_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            psutil.Process(victim_pid).send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                assert item.status == 200 and _ANSWER in item.text, (item.call, item.status, item.text)
            eventually(
                lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda count: count == 3, seconds=60
            )
            follow_up: Final = await _burst(
                _proxy_url(candidate), candidate.key, _PLAIN, _calls(again, "chat", False, range(6))
            )
            assert len(follow_up) == 6
            for item in follow_up:
                assert item.status == 200 and _ANSWER in item.text, (item.call, item.status, item.text)
            assert len(wire.drain()) == 26
            assert {row["request_id"] for row in _success_rows(marker, len(served))} == _answered(served)
            assert {row["request_id"] for row in _success_rows(again, 6)} == _answered(follow_up)


@pytest.mark.timeout(180)
async def test_c03_proxy_terminated_mid_burst_lands_every_answered_content_less_call_at_most_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"call-{uuid.uuid4().hex}"
    calls: Final = _calls(marker, "chat", False, range(12))
    release: Final = threading.Event()
    held_indexes: Final[SimpleQueue[int]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_indexes.put(_call_index(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _converse_peer(request)

    with wire_server(held) as wire:
        config: Final = _owned_config(wire, tmp_path, modify_params=False)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=1) as owned:
            candidate: Final = owned.gateway
            burst: Final = asyncio.create_task(
                _burst(_proxy_url(candidate), candidate.key, _PLAIN, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held_indexes.qsize, lambda size: size == 12, 60)
            owned.process.terminate()
            release.set()
            served: Final = await burst
            eventually(owned.process.poll, lambda code: code is not None, seconds=60)
            answered: Final = _answered(item for item in served if item.status == 200)
            assert len(served) <= 12
            landed: Final = tuple(
                row["request_id"]
                for row in read_rows(
                    'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE end_user LIKE %s AND status=%s',
                    (f"{marker}%", "success"),
                )
            )
            assert len(landed) == len(set(landed)), landed
            stray: Final = set(landed) - answered
            assert len(stray) <= 12 - len(answered), (landed, answered)
            assert len(wire.drain()) == 12
