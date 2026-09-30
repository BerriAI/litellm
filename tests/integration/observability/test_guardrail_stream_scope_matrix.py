from __future__ import annotations

import json
import os
import uuid
from asyncio import run
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

import anthropic
import httpx
import openai
import pytest
import websockets
import yaml
from integration._support.client import Gateway, JsonValue, Scenario, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpCaller, echo_tool, register_mcp, scripted_peer
from integration._support.process import owned_proxy_process
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.cost_calculation.cost_tracking_case import RealtimeResponse
from pydantic import TypeAdapter

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]
Mode: TypeAlias = Literal["pre_call", "during_call", "post_call", "logging_only"]
Scope: TypeAlias = Literal["streaming", "non_streaming"]
MatrixScope: TypeAlias = Scope | None
Endpoints: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
Modes: Final[tuple[Mode, ...]] = ("pre_call", "during_call", "post_call", "logging_only")
Scopes: Final[tuple[MatrixScope, ...]] = ("streaming", "non_streaming", None)
PROVIDER_TEXT: Final = "provider stream_scope control"
GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
F1_BLOCKED_WORD: Final = "streamscope-mcp-blocked-word"
F3_STREAM_BLOCKED_WORD: Final = "matrix-f3-stream-block"
F3_NON_STREAM_BLOCKED_WORD: Final = "matrix-f3-non-stream-block"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _strings(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(chain.from_iterable(_strings(item) for item in value))
    if isinstance(value, dict):
        return tuple(chain.from_iterable(_strings(item) for item in value.values()))
    return ()


def _marker(body: Mapping[str, JsonValue]) -> str:
    return next((text for text in _strings(dict(body)) if text.startswith("audit-")), "audit-provider")


def _sse(events: Iterable[Mapping[str, JsonValue]]) -> tuple[bytes, ...]:
    return tuple(f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode() for event in events) + (
        b"data: [DONE]\n\n",
    )


def _messages_stream(message: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    content: Final = cast(list[JsonValue], message["content"])
    text: Final = cast(dict[str, JsonValue], content[0])["text"]
    return (
        f"event: message_start\ndata: {json.dumps({**message, 'content': [], 'stop_reason': None, 'usage': {'input_tokens': 11, 'output_tokens': 0}})}\n\n".encode(),
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n".encode(),
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 4}})}\n\n".encode(),
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )


def _responses_stream(
    response: Mapping[str, JsonValue], output: Mapping[str, JsonValue], marker: str
) -> tuple[bytes, ...]:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": []}},
        {"type": "response.output_item.added", "item": output, "output_index": 0},
        {
            "type": "response.content_part.added",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        },
        {
            "type": "response.output_text.delta",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": f"{PROVIDER_TEXT} {marker}",
        },
        {
            "type": "response.output_text.done",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "text": f"{PROVIDER_TEXT} {marker}",
        },
        {
            "type": "response.content_part.done",
            "item_id": f"msg-{marker}",
            "output_index": 0,
            "content_index": 0,
            "part": cast(list[JsonValue], output["content"])[0],
        },
        {"type": "response.output_item.done", "item": output, "output_index": 0},
        {"type": "response.completed", "response": response},
    )
    return tuple(
        f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index}, separators=(',', ':'))}\n\n".encode()
        for index, event in enumerate(events)
    )


def _provider_reply(target: str, body: Mapping[str, JsonValue]) -> Reply:
    marker: Final = _marker(body)
    streamed: Final = bool(body.get("stream"))
    if target == "/v1/chat/completions":
        if streamed:
            common: Final = {"id": f"chatcmpl-{marker}", "object": "chat.completion.chunk", "created": 1}
            return Reply(
                content_type="text/event-stream",
                chunks=_sse(
                    (
                        {
                            **common,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        },
                        {
                            **common,
                            "choices": [
                                {"index": 0, "delta": {"content": f"{PROVIDER_TEXT} {marker}"}, "finish_reason": None}
                            ],
                        },
                        {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    )
                ),
            )
        return Reply(
            body=_json(
                {
                    "id": f"chatcmpl-{marker}",
                    "object": "chat.completion",
                    "created": 1,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"{PROVIDER_TEXT} {marker}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            )
        )
    if target == "/v1/messages":
        message: Final = {
            "id": f"msg-{marker}",
            "type": "message",
            "role": "assistant",
            "model": "synthetic-anthropic-model",
            "content": [{"type": "text", "text": f"{PROVIDER_TEXT} {marker}"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 4},
        }
        return (
            Reply(content_type="text/event-stream", chunks=_messages_stream(message))
            if streamed
            else Reply(body=_json(message))
        )
    if target == "/v1/responses":
        output: Final = {
            "type": "message",
            "id": f"msg-{marker}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"{PROVIDER_TEXT} {marker}", "annotations": []}],
        }
        response: Final = {
            "id": f"resp-{marker}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [output],
            "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
        }
        return (
            Reply(content_type="text/event-stream", chunks=_responses_stream(response, output, marker))
            if streamed
            else Reply(body=_json(response))
        )
    if target.endswith(":generateContent"):
        return Reply(
            body=_json(
                {"candidates": [{"content": {"role": "model", "parts": [{"text": f"{PROVIDER_TEXT} {marker}"}]}}]}
            )
        )
    if target.endswith(":streamGenerateContent"):
        return Reply(
            content_type="text/event-stream",
            chunks=(
                f"data: {_json({'candidates': [{'content': {'role': 'model', 'parts': [{'text': f'{PROVIDER_TEXT} {marker}'}]}}]}).decode()}\n\n".encode(),
            ),
        )
    return Reply(status=404, body=_json({"error": {"message": f"unexpected target {target}"}}))


def _provider(request: Request) -> Reply:
    if not request.body:
        return Reply(status=400, body=_json({"error": "empty request body"}))
    body: Final = JSON_OBJECT.validate_json(request.body)
    if request.target.startswith("/passthrough"):
        marker: Final = _marker(body)
        if bool(body.get("stream")):
            return Reply(
                content_type="text/event-stream",
                chunks=_sse(({"text": f"{PROVIDER_TEXT} {marker}"},)),
            )
        return Reply(body=_json({"received": body}))
    if "audit-g3-401-" in request.body.decode():
        return Reply(status=401, body=_json({"error": {"message": "synthetic provider unauthorized"}}))
    target: Final = request.target.split("?", 1)[0]
    provider_target: Final = "/v1/messages" if target.endswith("/anthropic/v1/messages") else target
    return _provider_reply(provider_target, body)


def _sink(request: Request) -> Reply:
    assert request.target.endswith(GUARDRAIL_PATH), request.target
    JSON_OBJECT.validate_json(request.body)
    if "/g1-" in request.target:
        return Reply(status=500, body=_json({"error": "synthetic sink failure"}))
    if "/g2-" in request.target or "/f2-block-" in request.target:
        return Reply(body=_json({"action": "BLOCKED", "blocked_reason": "synthetic policy block"}))
    return Reply(body=_json({"action": "NONE"}))


def _rail(
    name: str,
    sink: Wire,
    *,
    mode: str | list[str],
    scope: str | Mapping[str, str] | None = None,
    default_on: bool = False,
) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": default_on,
            **({"stream_scope": dict(scope)} if isinstance(scope, Mapping) else {}),
            **({"stream_scope": scope} if isinstance(scope, str) else {}),
            "api_base": f"{sink.url}/{name}",
            "api_key": "synthetic-guardrail-key",
        },
    }


def _realtime_filter_rail(name: str, scope: Scope, keyword: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "realtime_input_transcription",
            "default_on": True,
            "stream_scope": scope,
            "blocked_words": [{"keyword": keyword, "action": "BLOCK"}],
        },
    }


def _mcp_filter_rail(name: str, scope: MatrixScope) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "pre_mcp_call",
            "default_on": False,
            **({"stream_scope": scope} if scope is not None else {}),
            "blocked_words": [{"keyword": F1_BLOCKED_WORD, "action": "BLOCK"}],
        },
    }


def _mode_scope_pairs() -> tuple[tuple[Mode, MatrixScope], ...]:
    return tuple(chain.from_iterable(tuple((mode, scope) for scope in Scopes) for mode in Modes))


def _rail_names() -> Mapping[str, str]:
    return MappingProxyType(
        {
            **{f"a_{mode}_{scope or 'unset'}": f"a_{mode}_{scope or 'unset'}" for mode, scope in _mode_scope_pairs()},
            "c1_default": "c1_default",
            "c2_key": "c2_key",
            "c3_team": "c3_team",
            "c4_modes": "c4_modes",
            "c5_omitted": "c5_omitted",
            "c6_both": "c6_both",
            "c6_unset": "c6_unset",
            "e_stream": "e_stream",
            "e_non_stream": "e_non_stream",
            "f1_stream": "f1_stream",
            "f1_non_stream": "f1_non_stream",
            "f1_unset": "f1_unset",
            "f2_stream": "f2_stream",
            "f2_non_stream": "f2-block-non-stream",
            "f3_stream": "f3_stream",
            "f3_non_stream": "f3_non_stream",
            "g1_failure": "g1-failure",
            "g2_block": "g2-block",
            "g3_provider": "g3-provider",
        }
    )


def _configured_rails(names: Mapping[str, str], sink: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _rail(names[f"a_{mode}_{scope or 'unset'}"], sink, mode=mode, scope=scope)
        for mode, scope in _mode_scope_pairs()
    ) + (
        _rail(names["c1_default"], sink, mode="post_call", scope="streaming"),
        _rail(names["c2_key"], sink, mode="post_call", scope="streaming"),
        _rail(names["c3_team"], sink, mode="post_call", scope="streaming"),
        _rail(
            names["c4_modes"],
            sink,
            mode=["pre_call", "post_call"],
            scope={"pre_call": "non_streaming", "post_call": "streaming"},
        ),
        _rail(
            names["c5_omitted"],
            sink,
            mode=["pre_call", "post_call"],
            scope={"post_call": "streaming"},
        ),
        _rail(names["c6_both"], sink, mode="post_call", scope="both"),
        _rail(names["c6_unset"], sink, mode="post_call"),
        _rail(names["e_stream"], sink, mode="pre_call", scope="streaming"),
        _rail(names["e_non_stream"], sink, mode="pre_call", scope="non_streaming"),
        _mcp_filter_rail(names["f1_stream"], "streaming"),
        _mcp_filter_rail(names["f1_non_stream"], "non_streaming"),
        _mcp_filter_rail(names["f1_unset"], None),
        _rail(names["f2_stream"], sink, mode="post_call", scope="streaming"),
        _rail(names["f2_non_stream"], sink, mode="post_call", scope="non_streaming"),
        _realtime_filter_rail(names["f3_stream"], "streaming", F3_STREAM_BLOCKED_WORD),
        _realtime_filter_rail(names["f3_non_stream"], "non_streaming", F3_NON_STREAM_BLOCKED_WORD),
        _rail(names["g1_failure"], sink, mode="pre_call", scope="streaming"),
        _rail(names["g2_block"], sink, mode="pre_call", scope="streaming"),
        _rail(names["g3_provider"], sink, mode="pre_call", scope="both"),
    )


def _realtime_transcription_response(transcript: str) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": "evt_$REQUEST_ID",
                "item_id": "item_$REQUEST_ID",
                "content_index": 0,
                "transcript": transcript,
            },
            {
                "type": "response.done",
                "event_id": "evt_$REQUEST_ID",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "total_tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_token_details": {
                            "text_tokens": 0,
                            "audio_tokens": 0,
                            "cached_tokens": 0,
                            "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 0},
                        },
                        "output_token_details": {"text_tokens": 0, "audio_tokens": 0},
                    },
                },
            },
        ),
    )


async def _collect_realtime_events(websocket: websockets.ClientConnection) -> tuple[dict[str, JsonValue], ...]:
    event: Final = JSON_OBJECT.validate_json(await websocket.recv())
    if event.get("type") == "response.done":
        return (event,)
    return (event, *await _collect_realtime_events(websocket))


async def _realtime_transcription_events(
    url: str,
    key: str,
    model: str,
) -> tuple[dict[str, JsonValue], ...]:
    websocket_url: Final = (
        f"{url.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/')}/v1/realtime?model={model}"
    )
    async with websockets.connect(websocket_url, additional_headers={"Authorization": f"Bearer {key}"}) as websocket:
        session: Final = JSON_OBJECT.validate_json(await websocket.recv())
        assert session.get("type") == "session.created", session
        await websocket.send(json.dumps({"type": "response.create"}))
        return await _collect_realtime_events(websocket)


@dataclass(frozen=True, slots=True)
class MatrixRig:
    candidate: Gateway
    scenario: Scenario
    models: Mapping[str, str]
    rails: Mapping[str, str]
    key: str
    provider: Wire
    sink: Wire


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MatrixRig]:
    with httpx.Client(
        base_url=os.environ["INTEGRATION_PROXY_URL"],
        timeout=30,
        trust_env=False,
    ) as root_client:
        root_gateway: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("guardrail-stream-scope-matrix")
        with wire_server(_provider) as provider, wire_server(_sink) as sink:
            names: Final = _rail_names()
            config_base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            pass_through_paths: Final = (
                {
                    "path": "/pt",
                    "target": f"{provider.url}/passthrough",
                    "include_subpath": True,
                    "guardrails": {names["e_stream"]: None, names["e_non_stream"]: None},
                },
                {
                    "path": "/anthropic/v1/messages",
                    "target": f"{provider.url}/v1/messages",
                    "guardrails": {names["e_stream"]: None, names["e_non_stream"]: None},
                },
                {
                    "path": "/gemini/v1beta/models",
                    "target": "",
                    "include_subpath": True,
                    "guardrails": {names["e_stream"]: None, names["e_non_stream"]: None},
                },
            )
            config: Final = {
                **config_base,
                "guardrails": _configured_rails(names, sink),
                "model_list": [
                    {
                        "model_name": "matrix-pipeline-model",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": f"{provider.url}/v1",
                            "api_key": "synthetic-provider-key",
                        },
                    }
                ],
                "environment_variables": {
                    **config_base.get("environment_variables", {}),
                    "ANTHROPIC_API_BASE": provider.url,
                    "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
                    "GEMINI_API_BASE": provider.url,
                    "GEMINI_API_KEY": "synthetic-gemini-key",
                },
                "general_settings": {
                    **config_base["general_settings"],
                    "pass_through_endpoints": pass_through_paths,
                },
                "policies": {
                    "matrix-f2": {
                        "guardrails": {"add": [names["f2_stream"], names["f2_non_stream"]]},
                        "pipeline": {
                            "mode": "post_call",
                            "steps": [
                                {"guardrail": names["f2_stream"], "on_pass": "next", "on_fail": "block"},
                                {"guardrail": names["f2_non_stream"], "on_pass": "next", "on_fail": "block"},
                            ],
                        },
                    }
                },
                "policy_attachments": [{"policy": "matrix-f2", "models": ["matrix-pipeline-model"]}],
            }
            config_path: Final = directory / "stream-scope-matrix.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with owned_proxy_process(root_gateway, directory, {}, config=config_path, workers=1) as owned:
                with owned.gateway.scenario() as scenario:
                    models: Final = MappingProxyType(
                        {
                            "chat": scenario.model(
                                model="openai/gpt-4o-mini",
                                api_base=f"{provider.url}/v1",
                                api_key="synthetic-provider-key",
                            ),
                            "messages": scenario.model(
                                model="anthropic/claude-sonnet-4-5-20250929",
                                api_base=provider.url,
                                api_key="synthetic-provider-key",
                            ),
                            "responses": scenario.model(
                                model="openai/gpt-4o-mini",
                                api_base=f"{provider.url}/v1",
                                api_key="synthetic-provider-key",
                            ),
                        }
                    )
                    key: Final = scenario.key(guardrails=[names["c2_key"]])
                    yield MatrixRig(owned.gateway, scenario, models, names, key, provider, sink)


def _cell_body(
    rig: MatrixRig,
    endpoint: Endpoint,
    marker: str,
    streamed: bool,
    rail: str | None,
) -> tuple[str, dict[str, JsonValue]]:
    selected: Final = [] if rail is None else [rail]
    if endpoint == "chat":
        body: Final = {
            "model": rig.models["chat"],
            "messages": [{"role": "user", "content": marker}],
            "guardrails": selected,
            **({"stream": True} if streamed else {}),
        }
        return "/v1/chat/completions", body
    if endpoint == "messages":
        body = {
            "model": rig.models["messages"],
            "max_tokens": 32,
            "messages": [{"role": "user", "content": marker}],
            "guardrails": selected,
            **({"stream": True} if streamed else {}),
        }
        return "/v1/messages", body
    body = {
        "model": rig.models["responses"],
        "input": marker,
        "guardrails": selected,
        **({"stream": True} if streamed else {}),
    }
    return "/v1/responses", body


def _matching_requests(wire: Wire, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if marker.encode() in request.body)


def _rail_scans(rows: Sequence[Request], rail_name: str, marker: str) -> tuple[Request, ...]:
    return tuple(
        request for request in rows if request.target.startswith(f"/{rail_name}/") and marker.encode() in request.body
    )


def _scope_scan_count(scope: MatrixScope, streamed: bool) -> int:
    if scope is None or scope == "both":
        return 1
    return int((scope == "streaming") == streamed)


def _spend_row_for_call(call_id: str, content: bytes) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            "SELECT request_id, litellm_call_id, spend, prompt_tokens, completion_tokens, metadata "
            'FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
            (call_id, call_id),
        ),
        lambda values: len(values) >= 1,
        seconds=70,
    )
    assert len(rows) == 1, (call_id, rows, content)
    assert call_id in (rows[0]["request_id"], rows[0]["litellm_call_id"]), content
    return rows[0]


def _cell_rows(
    rig: MatrixRig,
    marker: str,
    response: httpx.Response,
    mode: Mode,
    streamed: bool,
    scope: MatrixScope,
    call_id: str,
    rail: str,
) -> tuple[tuple[Request, ...], tuple[Request, ...]]:
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows, response.text)
    _spend_row_for_call(call_id, response.content)
    expected_scans: Final = _scope_scan_count(scope, streamed)
    sink_rows: Final = (
        eventually(
            lambda: _rail_scans(rig.sink.drain(), rail, marker),
            lambda values: len(values) >= expected_scans,
            seconds=70,
        )
        if mode == "logging_only" and expected_scans
        else _rail_scans(rig.sink.drain(), rail, marker)
    )
    return provider_rows, sink_rows


def _run_matrix_cell(
    rig: MatrixRig,
    endpoint: Endpoint,
    streamed: bool,
    mode: Mode,
    scope: MatrixScope,
) -> None:
    marker: Final = f"audit-a-{endpoint}-{int(streamed)}-{mode}-{scope or 'unset'}-{uuid.uuid4().hex}"
    call_id: Final = f"matrix-a-{uuid.uuid4().hex}"
    rail: Final = rig.rails[f"a_{mode}_{scope or 'unset'}"]
    path, body = _cell_body(rig, endpoint, marker, streamed, rail)
    response: Final = rig.candidate.request(
        "POST",
        path,
        body,
        headers={"x-litellm-call-id": call_id},
    )
    assert response.status_code == 200, response.text
    assert PROVIDER_TEXT in response.text and marker in response.text, response.text
    provider_rows, sink_rows = _cell_rows(rig, marker, response, mode, streamed, scope, call_id, rail)
    assert len(provider_rows) == 1, (marker, provider_rows, response.text)
    expected_scans: Final = _scope_scan_count(scope, streamed)
    if mode == "logging_only":
        assert not expected_scans or sink_rows, (marker, sink_rows, response.text)
    else:
        assert len(sink_rows) == expected_scans, (marker, sink_rows, response.text)


@pytest.mark.parametrize("endpoint", Endpoints)
@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
@pytest.mark.parametrize("mode", Modes)
@pytest.mark.parametrize("scope", Scopes, ids=("streaming", "non_streaming", "unset"))
def test_a_stream_scope_matrix(
    rig: MatrixRig,
    endpoint: Endpoint,
    streamed: bool,
    mode: Mode,
    scope: MatrixScope,
) -> None:
    _run_matrix_cell(rig, endpoint, streamed, mode, scope)


SDK_CASES: Final[tuple[str, ...]] = (
    "openai-chat-sync",
    "openai-chat-async",
    "openai-responses-sync",
    "openai-responses-async",
    "anthropic-messages-sync",
    "anthropic-messages-async",
)


def _openai_sdk_base_url(rig: MatrixRig) -> str:
    return f"{str(rig.candidate.client.base_url).rstrip('/')}/v1"


def _anthropic_sdk_base_url(rig: MatrixRig) -> str:
    return str(rig.candidate.client.base_url).rstrip("/")


def _verify_sdk_call(rig: MatrixRig, marker: str, call_id: str, text: str, streamed: bool) -> None:
    assert marker in text and PROVIDER_TEXT in text, text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    _spend_row_for_call(call_id, text.encode())
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails["c2_key"], marker)
    assert len(sink_rows) == int(streamed), (marker, streamed, sink_rows)


def _run_sync_sdk(rig: MatrixRig, sdk: str, marker: str, call_id: str, streamed: bool) -> str:
    if sdk == "openai-chat-sync":
        with openai.OpenAI(
            api_key=rig.key,
            base_url=_openai_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            response: Final = client.chat.completions.create(
                model=rig.models["chat"],
                messages=[{"role": "user", "content": marker}],
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                return "".join(
                    chunk.choices[0].delta.content or "" for chunk in response if chunk.choices[0].delta.content
                )
            return cast(str, response.choices[0].message.content)
    if sdk == "openai-responses-sync":
        with openai.OpenAI(
            api_key=rig.key,
            base_url=_openai_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            response = client.responses.create(
                model=rig.models["responses"],
                input=marker,
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                return "".join(event.delta for event in response if event.type == "response.output_text.delta")
            return response.output_text
    if sdk == "anthropic-messages-sync":
        with anthropic.Anthropic(
            api_key=rig.key,
            base_url=_anthropic_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            response = client.messages.create(
                model=rig.models["messages"],
                max_tokens=32,
                messages=[{"role": "user", "content": marker}],
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                return "".join(
                    event.delta.text
                    for event in response
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                )
            return "".join(block.text for block in response.content if block.type == "text")
    raise AssertionError(sdk)


async def _run_async_sdk(rig: MatrixRig, sdk: str, marker: str, call_id: str, streamed: bool) -> str:
    if sdk == "openai-chat-async":
        async with openai.AsyncOpenAI(
            api_key=rig.key,
            base_url=_openai_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response = await client.chat.completions.create(
                model=rig.models["chat"],
                messages=[{"role": "user", "content": marker}],
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                chunks: Final = [
                    chunk.choices[0].delta.content or "" async for chunk in response if chunk.choices[0].delta.content
                ]
                return "".join(chunks)
            return cast(str, response.choices[0].message.content)
    if sdk == "openai-responses-async":
        async with openai.AsyncOpenAI(
            api_key=rig.key,
            base_url=_openai_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response = await client.responses.create(
                model=rig.models["responses"],
                input=marker,
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                events: Final = [event.delta async for event in response if event.type == "response.output_text.delta"]
                return "".join(events)
            return response.output_text
    if sdk == "anthropic-messages-async":
        async with anthropic.AsyncAnthropic(
            api_key=rig.key,
            base_url=_anthropic_sdk_base_url(rig),
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response = await client.messages.create(
                model=rig.models["messages"],
                max_tokens=32,
                messages=[{"role": "user", "content": marker}],
                stream=streamed,
                extra_headers={"x-litellm-call-id": call_id},
            )
            if streamed:
                events: Final = [
                    event.delta.text
                    async for event in response
                    if event.type == "content_block_delta" and event.delta.type == "text_delta"
                ]
                return "".join(events)
            return "".join(block.text for block in response.content if block.type == "text")
    raise AssertionError(sdk)


@pytest.mark.parametrize("sdk", SDK_CASES)
@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
def test_b_streaming_scope_classifies_sdk_streams(rig: MatrixRig, sdk: str, streamed: bool) -> None:
    marker: Final = f"audit-b-{sdk}-{int(streamed)}-{uuid.uuid4().hex}"
    call_id: Final = f"matrix-b-{uuid.uuid4().hex}"
    if sdk.endswith("-async"):
        text: Final = run(_run_async_sdk(rig, sdk, marker, call_id, streamed))
    else:
        text = _run_sync_sdk(rig, sdk, marker, call_id, streamed)
    _verify_sdk_call(rig, marker, call_id, text, streamed)


def _yaml_proxy_config(
    rig: MatrixRig,
    guardrail_name: str,
    parameters: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    config_base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    return cast(
        dict[str, JsonValue],
        {
            **config_base,
            "guardrails": [{"guardrail_name": guardrail_name, "litellm_params": dict(parameters)}],
            "model_list": [
                {
                    "model_name": "scope-yaml-chat",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": f"{rig.provider.url}/v1",
                        "api_key": "synthetic-provider-key",
                    },
                }
            ],
            "environment_variables": {
                **config_base.get("environment_variables", {}),
                "OPENAI_API_BASE": rig.provider.url,
                "OPENAI_API_KEY": "synthetic-provider-key",
            },
        },
    )


def _management_params(
    name: str,
    rig: MatrixRig,
    *,
    mode: str | list[str] = "pre_call",
    scope: JsonValue = "streaming",
    default_on: bool = False,
) -> dict[str, JsonValue]:
    return {
        "guardrail": "generic_guardrail_api",
        "mode": mode,
        "default_on": default_on,
        "api_base": f"{rig.sink.url}/{name}",
        "api_key": "synthetic-guardrail-key",
        "stream_scope": scope,
    }


def _raw_chat(
    gateway: Gateway,
    marker: str,
    streamed: bool,
    model: str,
    *,
    rails: Sequence[str] = (),
    key: str | None = None,
    call_id: str | None = None,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": marker}],
            "guardrails": list(rails),
            **({"stream": True} if streamed else {}),
        },
        key=key,
        headers={} if call_id is None else {"x-litellm-call-id": call_id},
    )


def _assert_raw_call(
    rig: MatrixRig,
    marker: str,
    response: httpx.Response,
    streamed: bool,
    expected_scans: int,
    rail: str,
    call_id: str | None = None,
) -> tuple[Request, ...]:
    assert response.status_code == 200, response.text
    assert PROVIDER_TEXT in response.text and marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    if call_id is not None:
        _spend_row_for_call(call_id, response.content)
    sink_rows: Final = _rail_scans(rig.sink.drain(), rail, marker)
    assert len(sink_rows) == expected_scans, (marker, streamed, expected_scans, sink_rows)
    return sink_rows


def test_c1_default_on_rail_respects_stream_scope(rig: MatrixRig, tmp_path: Path) -> None:
    name: Final = f"c1-default-{uuid.uuid4().hex}"
    parameters: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": True,
        "stream_scope": "streaming",
        "api_base": f"{rig.sink.url}/{name}",
        "api_key": "synthetic-guardrail-key",
    }
    config_path: Final = tmp_path / "default-on-stream-scope.yaml"
    config_path.write_text(yaml.safe_dump(_yaml_proxy_config(rig, name, parameters)))
    with owned_proxy_process(rig.candidate, tmp_path, {}, config=config_path, workers=1) as owned:
        for streamed in (False, True):
            marker: Final = f"audit-c1-{int(streamed)}-{uuid.uuid4().hex}"
            response: Final = _raw_chat(owned.gateway, marker, streamed, "scope-yaml-chat")
            _assert_raw_call(rig, marker, response, streamed, int(streamed), name)


def test_c2_key_attached_rail_respects_stream_scope(rig: MatrixRig) -> None:
    marker0: Final = f"audit-c2-0-{uuid.uuid4().hex}"
    response0: Final = _raw_chat(rig.candidate, marker0, False, rig.models["chat"], key=rig.key)
    _assert_raw_call(rig, marker0, response0, False, 0, rig.rails["c2_key"])
    marker1: Final = f"audit-c2-1-{uuid.uuid4().hex}"
    response1: Final = _raw_chat(rig.candidate, marker1, True, rig.models["chat"], key=rig.key)
    _assert_raw_call(rig, marker1, response1, True, 1, rig.rails["c2_key"])


def test_c3_team_attached_rail_respects_stream_scope(rig: MatrixRig) -> None:
    team: Final = rig.scenario.team(guardrails=[rig.rails["c3_team"]])
    key: Final = rig.scenario.key(team_id=team)
    marker0: Final = f"audit-c3-0-{uuid.uuid4().hex}"
    response0: Final = _raw_chat(rig.candidate, marker0, False, rig.models["chat"], key=key)
    _assert_raw_call(rig, marker0, response0, False, 0, rig.rails["c3_team"])
    marker1: Final = f"audit-c3-1-{uuid.uuid4().hex}"
    response1: Final = _raw_chat(rig.candidate, marker1, True, rig.models["chat"], key=key)
    _assert_raw_call(rig, marker1, response1, True, 1, rig.rails["c3_team"])


@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
def test_c4_per_mode_scope_map_selects_each_mode(rig: MatrixRig, streamed: bool) -> None:
    marker: Final = f"audit-c4-{int(streamed)}-{uuid.uuid4().hex}"
    response: Final = _raw_chat(
        rig.candidate,
        marker,
        streamed,
        rig.models["chat"],
        rails=(rig.rails["c4_modes"],),
    )
    assert response.status_code == 200, response.text
    assert PROVIDER_TEXT in response.text and marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails["c4_modes"], marker)
    request_bodies: Final = tuple(JSON_OBJECT.validate_json(row.body) for row in sink_rows)
    request_texts: Final = tuple(chain.from_iterable(_strings(body.get("texts", [])) for body in request_bodies))
    assert len(sink_rows) == 1, (marker, streamed, sink_rows)
    assert any(marker in text for text in request_texts), (marker, request_texts)
    assert (any(PROVIDER_TEXT in text for text in request_texts)) == streamed, (
        marker,
        streamed,
        request_texts,
    )


@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
def test_c5_mode_omitted_from_scope_map_means_both(rig: MatrixRig, streamed: bool) -> None:
    marker: Final = f"audit-c5-{int(streamed)}-{uuid.uuid4().hex}"
    response: Final = _raw_chat(
        rig.candidate,
        marker,
        streamed,
        rig.models["chat"],
        rails=(rig.rails["c5_omitted"],),
    )
    sink_rows: Final = _assert_raw_call(
        rig,
        marker,
        response,
        streamed,
        1 + int(streamed),
        rig.rails["c5_omitted"],
    )
    request_bodies: Final = tuple(JSON_OBJECT.validate_json(row.body) for row in sink_rows)
    request_texts: Final = tuple(chain.from_iterable(_strings(body.get("texts", [])) for body in request_bodies))
    assert sum(marker in text and PROVIDER_TEXT not in text for text in request_texts) == 1, (marker, request_texts)
    assert sum(PROVIDER_TEXT in text for text in request_texts) == int(streamed), (marker, request_texts)


@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
def test_c6_explicit_both_matches_unset_scope(rig: MatrixRig, streamed: bool) -> None:
    marker: Final = f"audit-c6-{int(streamed)}-{uuid.uuid4().hex}"
    response: Final = _raw_chat(
        rig.candidate,
        marker,
        streamed,
        rig.models["chat"],
        rails=(rig.rails["c6_both"], rig.rails["c6_unset"]),
    )
    assert response.status_code == 200, response.text
    assert PROVIDER_TEXT in response.text and marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    sink_rows: Final = rig.sink.drain()
    both_rows: Final = _rail_scans(sink_rows, rig.rails["c6_both"], marker)
    unset_rows: Final = _rail_scans(sink_rows, rig.rails["c6_unset"], marker)
    assert (len(both_rows), len(unset_rows)) == (1, 1), (marker, streamed, both_rows, unset_rows)


@pytest.mark.parametrize(
    ("case", "scope"),
    (
        ("missing", None),
        ("null", None),
        ("empty", ""),
        ("invalid-scalar", "sometimes"),
        ("invalid-map", {"unknown_mode": "streaming"}),
    ),
    ids=("missing", "null", "empty", "invalid-scalar", "invalid-map"),
)
def test_c7_yaml_unset_and_invalid_scope_values_are_tolerated(
    rig: MatrixRig,
    tmp_path: Path,
    case: str,
    scope: JsonValue | None,
) -> None:
    name: Final = f"c7-yaml-{case}-{uuid.uuid4().hex}"
    parameters: Final = {
        "guardrail": "generic_guardrail_api",
        "mode": "pre_call",
        "default_on": True,
        "api_base": f"{rig.sink.url}/{name}",
        "api_key": "synthetic-guardrail-key",
        **({} if case == "missing" else {"stream_scope": scope}),
    }
    config_path: Final = tmp_path / f"{case}-stream-scope.yaml"
    config_path.write_text(yaml.safe_dump(_yaml_proxy_config(rig, name, parameters)))
    with owned_proxy_process(rig.candidate, tmp_path, {}, config=config_path, workers=1) as owned:
        for streamed in (False, True):
            marker: Final = f"audit-c7-{case}-{int(streamed)}-{uuid.uuid4().hex}"
            response: Final = _raw_chat(owned.gateway, marker, streamed, "scope-yaml-chat")
            _assert_raw_call(rig, marker, response, streamed, 1, name)
        if case == "empty":
            assert "Ignoring invalid stored stream_scope value of type str" in owned.log.read_text()


def test_c8_identical_requests_have_one_scan_and_spend_each(rig: MatrixRig) -> None:
    marker: Final = f"audit-c8-identical-{uuid.uuid4().hex}"
    call_ids: Final = tuple(f"matrix-c8-{uuid.uuid4().hex}" for _ in range(3))
    responses: Final = tuple(
        _raw_chat(
            rig.candidate,
            marker,
            True,
            rig.models["chat"],
            rails=(rig.rails["a_post_call_streaming"],),
            call_id=call_id,
        )
        for call_id in call_ids
    )
    assert tuple(response.status_code for response in responses) == (200, 200, 200), tuple(
        response.text for response in responses
    )
    for call_id, response in zip(call_ids, responses):
        _spend_row_for_call(call_id, response.content)
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails["a_post_call_streaming"], marker)
    assert len(sink_rows) == 3, (marker, sink_rows)
    guardrail_call_ids: Final = tuple(
        cast(str, JSON_OBJECT.validate_json(row.body).get("litellm_call_id")) for row in sink_rows
    )
    assert set(guardrail_call_ids) == set(call_ids), (call_ids, guardrail_call_ids)


def _create_management_rail(rig: MatrixRig, name: str, params: Mapping[str, JsonValue]) -> str:
    created: Final = rig.candidate.request(
        "POST",
        "/guardrails",
        {"guardrail": {"guardrail_name": name, "litellm_params": dict(params)}},
    )
    assert created.status_code == 200, created.text
    payload: Final = JSON_OBJECT.validate_json(created.content)
    identity: Final = payload.get("guardrail_id")
    assert isinstance(identity, str), payload
    return identity


def _management_observation(
    rig: MatrixRig,
    name: str,
    streamed: bool,
) -> tuple[int, int, int]:
    marker: Final = f"audit-d-management-{int(streamed)}-{uuid.uuid4().hex}"
    call_id: Final = f"matrix-d1-{uuid.uuid4().hex}"
    response: Final = _raw_chat(
        rig.candidate,
        marker,
        streamed,
        rig.models["chat"],
        rails=(name,),
        call_id=call_id,
    )
    provider_rows: Final = _matching_requests(rig.provider, marker)
    sink_rows: Final = _rail_scans(rig.sink.drain(), name, marker)
    if response.status_code == 200 and len(provider_rows) == 1:
        assert marker in response.text, response.text
        _spend_row_for_call(call_id, response.content)
    return response.status_code, len(provider_rows), len(sink_rows)


def _eventually_management_scope(
    rig: MatrixRig,
    name: str,
    expected: tuple[int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return eventually(
        lambda: (
            _management_observation(rig, name, False),
            _management_observation(rig, name, True),
        ),
        lambda values: (
            (values[0][0], values[0][2]) == (200, expected[0])
            and (values[1][0], values[1][2]) == (200, expected[1])
            and values[0][1] == values[1][1] == 1
        ),
        seconds=30,
    )


def test_d1_management_create_read_and_runtime_scope(rig: MatrixRig) -> None:
    name: Final = f"d1-management-{uuid.uuid4().hex}"
    params: Final = _management_params(name, rig)
    identity: Final = _create_management_rail(rig, name, params)
    try:
        info: Final = rig.candidate.request("GET", f"/guardrails/{identity}/info")
        listing: Final = rig.candidate.request("GET", "/v2/guardrails/list")
        info_payload: Final = JSON_OBJECT.validate_json(info.content)
        list_payload: Final = JSON_OBJECT.validate_json(listing.content)
        list_rows: Final = list_payload.get("guardrails")
        listed: Final = (
            tuple(row for row in list_rows if isinstance(row, dict) and row.get("guardrail_id") == identity)
            if isinstance(list_rows, list)
            else ()
        )
        info_params: Final = info_payload.get("litellm_params")
        list_params: Final = listed[0].get("litellm_params") if listed else None
        assert (
            info.status_code == 200
            and listing.status_code == 200
            and isinstance(info_params, dict)
            and info_params.get("stream_scope") == "streaming"
            and len(listed) == 1
            and isinstance(list_params, dict)
            and list_params.get("stream_scope") == "streaming"
        ), (info.status_code, info.text, listing.status_code, listing.text)
        observed: Final = _eventually_management_scope(rig, name, (0, 1))
        assert observed[0][2] == 0 and observed[1][2] == 1, observed
    finally:
        deleted: Final = rig.candidate.request("DELETE", f"/guardrails/{identity}")
        assert deleted.status_code == 200, deleted.text


def test_d2_patch_then_put_updates_runtime_scope(rig: MatrixRig) -> None:
    name: Final = f"d2-management-{uuid.uuid4().hex}"
    identity: Final = _create_management_rail(rig, name, _management_params(name, rig))
    try:
        patched: Final = rig.candidate.request(
            "PATCH",
            f"/guardrails/{identity}",
            {"litellm_params": {"stream_scope": "non_streaming"}},
        )
        assert patched.status_code == 200, patched.text
        after_patch: Final = _eventually_management_scope(rig, name, (1, 0))
        assert after_patch[0][2] == 1 and after_patch[1][2] == 0, after_patch
        params: Final = _management_params(name, rig)
        updated: Final = rig.candidate.request(
            "PUT",
            f"/guardrails/{identity}",
            {"guardrail": {"guardrail_name": name, "litellm_params": params}},
        )
        assert updated.status_code == 200, updated.text
        after_put: Final = _eventually_management_scope(rig, name, (0, 1))
        assert after_put[0][2] == 0 and after_put[1][2] == 1, after_put
    finally:
        deleted: Final = rig.candidate.request("DELETE", f"/guardrails/{identity}")
        assert deleted.status_code == 200, deleted.text


HOSTILE_SCOPE_VALUES: Final[tuple[JsonValue, ...]] = (
    1,
    [],
    "",
    "x" * 5000,
    {"unknown_mode": "streaming"},
    {"pre_call": 1},
)


@pytest.mark.parametrize("operation", ("POST", "PUT", "PATCH"))
@pytest.mark.parametrize(
    "scope", HOSTILE_SCOPE_VALUES, ids=("integer", "list", "empty", "long", "unknown-key", "wrong-value")
)
@pytest.mark.parametrize("repeat", (1, 2), ids=("first", "second"))
def test_d3_management_rejects_invalid_scope_values(
    rig: MatrixRig,
    operation: str,
    scope: JsonValue,
    repeat: int,
) -> None:
    name: Final = f"d3-management-{operation.lower()}-{repeat}-{uuid.uuid4().hex}"
    identity: Final = _create_management_rail(rig, name, _management_params(name, rig)) if operation != "POST" else None
    payload: Final = (
        {"litellm_params": {"stream_scope": scope}}
        if operation == "PATCH"
        else {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": _management_params(name, rig, scope=scope),
            }
        }
    )
    path: Final = "/guardrails" if operation == "POST" else f"/guardrails/{identity}"
    try:
        response: Final = rig.candidate.request(operation, path, payload)
        if operation == "POST" and response.status_code == 200:
            created_payload: Final = JSON_OBJECT.validate_json(response.content)
            created_identity: Final = created_payload.get("guardrail_id")
            if isinstance(created_identity, str):
                rig.candidate.request("DELETE", f"/guardrails/{created_identity}")
        assert response.status_code == 422 and "stream_scope" in response.text, (
            operation,
            scope,
            repeat,
            response.status_code,
            response.text,
        )
        if operation == "POST":
            stored: Final = read_rows(
                'SELECT guardrail_id FROM "LiteLLM_GuardrailsTable" WHERE guardrail_name=%s',
                (name,),
            )
            assert stored == [], (operation, scope, stored)
        else:
            existing: Final = rig.candidate.request("GET", f"/guardrails/{identity}/info")
            existing_payload: Final = JSON_OBJECT.validate_json(existing.content)
            existing_params: Final = existing_payload.get("litellm_params")
            assert (
                existing.status_code == 200
                and isinstance(existing_params, dict)
                and existing_params.get("stream_scope") == "streaming"
            ), (operation, scope, existing.status_code, existing.text)
    finally:
        if isinstance(identity, str):
            deleted: Final = rig.candidate.request("DELETE", f"/guardrails/{identity}")
            assert deleted.status_code == 200, deleted.text


def test_d3_management_normalizes_uppercase_scope(rig: MatrixRig) -> None:
    name: Final = f"d3-uppercase-{uuid.uuid4().hex}"
    identity: Final = _create_management_rail(rig, name, _management_params(name, rig, scope="STREAMING"))
    try:
        info: Final = rig.candidate.request("GET", f"/guardrails/{identity}/info")
        payload: Final = JSON_OBJECT.validate_json(info.content)
        parameters: Final = payload.get("litellm_params")
        assert (
            info.status_code == 200 and isinstance(parameters, dict) and parameters.get("stream_scope") == "streaming"
        ), (info.status_code, info.text)
    finally:
        deleted: Final = rig.candidate.request("DELETE", f"/guardrails/{identity}")
        assert deleted.status_code == 200, deleted.text


def test_d4_unauthenticated_management_create_is_rejected(rig: MatrixRig) -> None:
    name: Final = f"d4-unauthenticated-{uuid.uuid4().hex}"
    response: Final = rig.candidate.client.request(
        "POST",
        "/guardrails",
        json={"guardrail": {"guardrail_name": name, "litellm_params": _management_params(name, rig)}},
    )
    assert response.status_code == 401, response.text


def _scoped_rows(
    rows: Sequence[Request],
    marker: str,
    streaming_name: str,
    non_streaming_name: str,
) -> tuple[tuple[Request, ...], tuple[Request, ...]]:
    return (
        _rail_scans(rows, streaming_name, marker),
        _rail_scans(rows, non_streaming_name, marker),
    )


HOSTILE_CLASSIFICATION_VALUES: Final[tuple[JsonValue, ...]] = (
    True,
    1,
    [],
    "",
    "litellm-server-streaming",
    "x" * 5000,
)


@pytest.mark.parametrize(
    "value",
    HOSTILE_CLASSIFICATION_VALUES,
    ids=("true", "integer", "list", "empty", "marker", "long"),
)
def test_ea_is_streaming_request_body_does_not_change_chat_classification(
    rig: MatrixRig,
    value: JsonValue,
) -> None:
    marker: Final = f"audit-ea-{uuid.uuid4().hex}"
    response: Final = rig.candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": rig.models["chat"],
            "messages": [{"role": "user", "content": marker}],
            "guardrails": [rig.rails["e_stream"], rig.rails["e_non_stream"]],
            "is_streaming_request": value,
        },
    )
    assert response.status_code == 200, response.text
    assert PROVIDER_TEXT in response.text and marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    provider_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    assert provider_body.get("is_streaming_request") == value, provider_rows[0].body.decode()
    sink_rows: Final = rig.sink.drain()
    streaming_rows, non_streaming_rows = _scoped_rows(
        sink_rows,
        marker,
        rig.rails["e_stream"],
        rig.rails["e_non_stream"],
    )
    assert (len(streaming_rows), len(non_streaming_rows)) == (0, 1), (marker, streaming_rows, non_streaming_rows)


@pytest.mark.parametrize("endpoint", ("chat", "passthrough"), ids=("chat", "configured-pass-through"))
@pytest.mark.parametrize("value", (True, "litellm-server-streaming"), ids=("true", "marker"))
def test_eb_namespaced_caller_body_field_cannot_flip_classification(
    rig: MatrixRig,
    endpoint: str,
    value: JsonValue,
) -> None:
    marker: Final = f"audit-eb-{endpoint}-{uuid.uuid4().hex}"
    body: Final = (
        {
            "model": rig.models["chat"],
            "messages": [{"role": "user", "content": marker}],
            "guardrails": [rig.rails["e_stream"], rig.rails["e_non_stream"]],
            "litellm_server_streaming_classification": value,
        }
        if endpoint == "chat"
        else {
            "marker": marker,
            "litellm_server_streaming_classification": value,
        }
    )
    response: Final = rig.candidate.request(
        "POST",
        "/v1/chat/completions" if endpoint == "chat" else "/pt",
        body,
    )
    assert response.status_code == 200 and marker in response.text, response.text
    actual_streamed: Final = response.headers.get("content-type", "").lower().startswith("text/event-stream")
    assert not actual_streamed, (endpoint, response.headers, response.text)
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    provider_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    assert provider_body.get("litellm_server_streaming_classification") == value, provider_rows[0].body.decode()
    sink_rows: Final = rig.sink.drain()
    streaming_rows, non_streaming_rows = _scoped_rows(
        sink_rows,
        marker,
        rig.rails["e_stream"],
        rig.rails["e_non_stream"],
    )
    assert (len(streaming_rows), len(non_streaming_rows)) == (0, 1), (marker, streaming_rows, non_streaming_rows)


HOSTILE_STREAM_VALUES: Final[tuple[JsonValue, ...]] = ("true", 1, [], "")


@pytest.mark.parametrize("value", HOSTILE_STREAM_VALUES, ids=("string", "integer", "list", "empty"))
def test_ec_hostile_stream_values_follow_observed_response_shape(rig: MatrixRig, value: JsonValue) -> None:
    marker: Final = f"audit-ec-{uuid.uuid4().hex}"
    response: Final = rig.candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": rig.models["chat"],
            "messages": [{"role": "user", "content": marker}],
            "guardrails": [rig.rails["e_stream"], rig.rails["e_non_stream"]],
            "stream": value,
        },
    )
    assert response.status_code == 200, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    observed_stream: Final = response.headers.get("content-type", "").startswith("text/event-stream")
    assert marker in response.text, response.text
    sink_rows: Final = rig.sink.drain()
    streaming_rows, non_streaming_rows = _scoped_rows(
        sink_rows,
        marker,
        rig.rails["e_stream"],
        rig.rails["e_non_stream"],
    )
    assert (len(streaming_rows), len(non_streaming_rows)) == (
        int(observed_stream),
        int(not observed_stream),
    ), (marker, value, response.headers, streaming_rows, non_streaming_rows)


@pytest.mark.parametrize("streamed", (False, True), ids=("stream-absent", "stream-true"))
def test_ed_configured_passthrough_forwards_caller_flag_and_uses_route_body_stream(
    rig: MatrixRig,
    streamed: bool,
) -> None:
    marker: Final = f"audit-ed-{int(streamed)}-{uuid.uuid4().hex}"
    body: Final = {
        "marker": marker,
        "is_streaming_request": True,
        **({"stream": True} if streamed else {}),
    }
    response: Final = rig.candidate.request("POST", "/pt", body)
    assert response.status_code == 200 and marker in response.text, response.text
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    provider_body: Final = JSON_OBJECT.validate_json(provider_rows[0].body)
    assert provider_body == body, (body, provider_body)


@pytest.mark.parametrize(
    ("route", "body", "streamed"),
    (
        (
            "/anthropic/v1/messages",
            {
                "model": "synthetic-anthropic-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "MARKER"}],
            },
            False,
        ),
        (
            "/anthropic/v1/messages",
            {
                "model": "synthetic-anthropic-model",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "MARKER"}],
                "stream": True,
            },
            True,
        ),
        (
            "/gemini/v1beta/models/audit-model:generateContent",
            {"contents": [{"parts": [{"text": "MARKER"}]}]},
            False,
        ),
        (
            "/gemini/v1beta/models/audit-model:streamGenerateContent?alt=sse",
            {"contents": [{"parts": [{"text": "MARKER"}]}]},
            True,
        ),
    ),
    ids=("anthropic-S0", "anthropic-S1", "gemini-generate", "gemini-stream"),
)
def test_eh_provider_passthrough_routes_classify_effective_streaming(
    rig: MatrixRig,
    route: str,
    body: dict[str, JsonValue],
    streamed: bool,
) -> None:
    marker: Final = f"audit-eh-{uuid.uuid4().hex}"
    request_body: Final = JSON_OBJECT.validate_python(json.loads(json.dumps(body).replace("MARKER", marker)))
    team: Final = (
        rig.scenario.team(
            metadata={
                "allowed_passthrough_routes": [
                    "/gemini/v1beta/models/audit-model:generateContent",
                    "/gemini/v1beta/models/audit-model:streamGenerateContent",
                ]
            }
        )
        if route.startswith("/gemini/")
        else None
    )
    key: Final = (
        rig.scenario.key(team_id=team, guardrails=[rig.rails["e_stream"], rig.rails["e_non_stream"]])
        if team is not None
        else rig.scenario.key(guardrails=[rig.rails["e_stream"], rig.rails["e_non_stream"]])
    )
    headers: Final = {"x-goog-api-key": key} if route.startswith("/gemini/") else None
    response: Final = rig.candidate.request(
        "POST",
        route,
        request_body,
        headers=headers,
    )
    assert response.status_code == 200 and marker in response.text, response.text
    actual_streamed: Final = response.headers.get("content-type", "").lower().startswith("text/event-stream")
    assert actual_streamed is streamed, (route, streamed, response.headers, response.text)
    provider_rows: Final = _matching_requests(rig.provider, marker)
    assert len(provider_rows) == 1, (marker, provider_rows)
    sink_rows: Final = rig.sink.drain()
    streaming_rows, non_streaming_rows = _scoped_rows(
        sink_rows,
        marker,
        rig.rails["e_stream"],
        rig.rails["e_non_stream"],
    )
    assert (len(streaming_rows), len(non_streaming_rows)) == ((1, 0) if streamed else (0, 1)), (
        marker,
        route,
        streaming_rows,
        non_streaming_rows,
    )


def test_ei_unauthenticated_scoped_request_has_no_upstream_or_guardrail_call(rig: MatrixRig) -> None:
    marker: Final = f"audit-ei-{uuid.uuid4().hex}"
    response: Final = rig.candidate.client.request(
        "POST",
        "/v1/chat/completions",
        json={
            "model": rig.models["chat"],
            "messages": [{"role": "user", "content": marker}],
            "stream": True,
            "guardrails": [rig.rails["e_stream"]],
        },
    )
    assert response.status_code == 401, response.text
    assert _matching_requests(rig.provider, marker) == ()
    assert _rail_scans(rig.sink.drain(), rig.rails["e_stream"], marker) == ()


@pytest.mark.parametrize(
    ("rail_key", "blocked"),
    (("f1_stream", False), ("f1_non_stream", True), ("f1_unset", True)),
    ids=("streaming", "non_streaming", "unset"),
)
def test_f1_mcp_content_filter_respects_stream_scope(
    rig: MatrixRig,
    rail_key: Literal["f1_stream", "f1_non_stream", "f1_unset"],
    blocked: bool,
) -> None:
    with scripted_peer(echo_tool("echo")) as peer:
        alias: Final = f"streamscope{uuid.uuid4().hex}"
        identity: Final = register_mcp(rig.scenario, peer, alias)
        key: Final = rig.scenario.key(
            object_permission={"mcp_servers": [identity]},
            guardrails=[rig.rails[rail_key]],
        )
        marker: Final = f"audit-f1-{uuid.uuid4().hex}"
        caller: Final = McpCaller(rig.candidate, key, "rest", alias)
        outcome: Final = caller.call(
            f"{alias}-echo",
            {"text": f"{F1_BLOCKED_WORD} {marker}"},
            identity,
        )
        peer_requests: Final = peer.drain()
        marker_requests: Final = tuple(request for request in peer_requests if marker in json.dumps(request))
        if blocked:
            assert outcome.error is not None, outcome.raw
            assert "block" in outcome.raw.casefold() or F1_BLOCKED_WORD in outcome.raw.casefold(), outcome.raw
            assert marker_requests == (), (marker, marker_requests)
            return
        assert outcome.ok and marker in (outcome.text or ""), outcome.raw
        assert len(marker_requests) == 1, (marker, marker_requests)


@pytest.mark.parametrize("streamed", (False, True), ids=("S0", "S1"))
def test_f2_mismatched_policy_step_skips_matching_sibling_still_enforces(
    rig: MatrixRig,
    streamed: bool,
) -> None:
    marker: Final = f"audit-f2-{int(streamed)}-{uuid.uuid4().hex}"
    response: Final = _raw_chat(
        rig.candidate,
        marker,
        streamed,
        "matrix-pipeline-model",
    )
    provider_rows: Final = _matching_requests(rig.provider, marker)
    sink_rows: Final = rig.sink.drain()
    streaming_rows, non_streaming_rows = _scoped_rows(
        sink_rows,
        marker,
        rig.rails["f2_stream"],
        rig.rails["f2_non_stream"],
    )
    assert len(provider_rows) == 1, (marker, provider_rows)
    if streamed:
        assert response.status_code == 200 and marker in response.text, response.text
        assert (len(streaming_rows), len(non_streaming_rows)) == (1, 0), (marker, streaming_rows, non_streaming_rows)
        return
    assert response.status_code == 400 and "synthetic policy block" in response.text, response.text
    assert (len(streaming_rows), len(non_streaming_rows)) == (0, 1), (marker, streaming_rows, non_streaming_rows)


def test_f3_realtime_transcription_uses_streaming_scope(rig: MatrixRig) -> None:
    streaming_scenario_id: Final = f"matrix-f3-stream-{uuid.uuid4().hex}"
    streaming_upstream: Final = register_scenario(
        streaming_scenario_id,
        _realtime_transcription_response(F3_STREAM_BLOCKED_WORD),
    )
    rig.scenario.cleanups.callback(delete_scenario, streaming_upstream)
    streaming_model: Final = rig.scenario.model(
        model="openai/gpt-realtime-2",
        api_key=streaming_scenario_id,
        api_base=rig.candidate.upstream_url,
    )
    non_streaming_scenario_id: Final = f"matrix-f3-non-stream-{uuid.uuid4().hex}"
    non_streaming_upstream: Final = register_scenario(
        non_streaming_scenario_id,
        _realtime_transcription_response(F3_NON_STREAM_BLOCKED_WORD),
    )
    rig.scenario.cleanups.callback(delete_scenario, non_streaming_upstream)
    non_streaming_model: Final = rig.scenario.model(
        model="openai/gpt-realtime-2",
        api_key=non_streaming_scenario_id,
        api_base=rig.candidate.upstream_url,
    )
    key: Final = rig.scenario.key()
    proxy_url: Final = str(rig.candidate.client.base_url).rstrip("/")

    streaming_events: Final = run(_realtime_transcription_events(proxy_url, key, streaming_model))
    streaming_transcriptions: Final = tuple(
        event
        for event in streaming_events
        if event.get("type") == "conversation.item.input_audio_transcription.completed"
    )
    streaming_errors: Final = tuple(event for event in streaming_events if event.get("type") == "error")
    assert len(streaming_transcriptions) == 1, streaming_events
    assert streaming_transcriptions[0].get("transcript") == F3_STREAM_BLOCKED_WORD, streaming_events
    assert len(streaming_errors) == 1, streaming_events
    streaming_error: Final = streaming_errors[0].get("error")
    assert isinstance(streaming_error, dict) and streaming_error.get("type") == "guardrail_violation", streaming_events

    non_streaming_events: Final = run(_realtime_transcription_events(proxy_url, key, non_streaming_model))
    non_streaming_transcriptions: Final = tuple(
        event
        for event in non_streaming_events
        if event.get("type") == "conversation.item.input_audio_transcription.completed"
    )
    non_streaming_errors: Final = tuple(event for event in non_streaming_events if event.get("type") == "error")
    completions: Final = tuple(event for event in non_streaming_events if event.get("type") == "response.done")
    assert len(non_streaming_transcriptions) == 1, non_streaming_events
    assert non_streaming_transcriptions[0].get("transcript") == F3_NON_STREAM_BLOCKED_WORD, non_streaming_events
    assert non_streaming_errors == (), non_streaming_events
    assert len(completions) == 1, non_streaming_events


def test_g1_sink_failure_fails_closed_only_when_rail_is_in_scope(rig: MatrixRig) -> None:
    marker_s0: Final = f"audit-g1-S0-{uuid.uuid4().hex}"
    response_s0: Final = _raw_chat(
        rig.candidate,
        marker_s0,
        False,
        rig.models["chat"],
        rails=(rig.rails["g1_failure"],),
    )
    assert response_s0.status_code == 200 and marker_s0 in response_s0.text, response_s0.text
    provider_rows_s0: Final = _matching_requests(rig.provider, marker_s0)
    sink_rows_s0: Final = _rail_scans(rig.sink.drain(), rig.rails["g1_failure"], marker_s0)
    assert len(provider_rows_s0) == 1 and sink_rows_s0 == (), (marker_s0, provider_rows_s0, sink_rows_s0)
    marker_s1: Final = f"audit-g1-S1-{uuid.uuid4().hex}"
    response_s1: Final = _raw_chat(
        rig.candidate,
        marker_s1,
        True,
        rig.models["chat"],
        rails=(rig.rails["g1_failure"],),
    )
    assert response_s1.status_code == 500 and "Generic Guardrail API failed" in response_s1.text, response_s1.text
    assert _matching_requests(rig.provider, marker_s1) == ()
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails["g1_failure"], marker_s1)
    assert len(sink_rows) == 1, (marker_s1, sink_rows)


def test_g2_blocked_verdict_blocks_only_when_rail_is_in_scope(rig: MatrixRig) -> None:
    marker_s0: Final = f"audit-g2-S0-{uuid.uuid4().hex}"
    response_s0: Final = _raw_chat(
        rig.candidate,
        marker_s0,
        False,
        rig.models["chat"],
        rails=(rig.rails["g2_block"],),
    )
    assert response_s0.status_code == 200 and marker_s0 in response_s0.text, response_s0.text
    assert len(_matching_requests(rig.provider, marker_s0)) == 1
    assert _rail_scans(rig.sink.drain(), rig.rails["g2_block"], marker_s0) == ()
    marker_s1: Final = f"audit-g2-S1-{uuid.uuid4().hex}"
    response_s1: Final = _raw_chat(
        rig.candidate,
        marker_s1,
        True,
        rig.models["chat"],
        rails=(rig.rails["g2_block"],),
    )
    assert response_s1.status_code == 400 and "synthetic policy block" in response_s1.text, response_s1.text
    assert _matching_requests(rig.provider, marker_s1) == ()
    sink_rows: Final = _rail_scans(rig.sink.drain(), rig.rails["g2_block"], marker_s1)
    assert len(sink_rows) == 1, (marker_s1, sink_rows)


def test_g3_provider_errors_reach_caller_and_proxy_remains_usable(rig: MatrixRig) -> None:
    marker: Final = f"audit-g3-401-{uuid.uuid4().hex}"
    unauthorized: Final = _raw_chat(
        rig.candidate,
        marker,
        False,
        rig.models["chat"],
        rails=(rig.rails["g3_provider"],),
    )
    assert unauthorized.status_code == 401 and "synthetic provider unauthorized" in unauthorized.text, unauthorized.text
    assert len(_matching_requests(rig.provider, marker)) == 1
    assert len(_rail_scans(rig.sink.drain(), rig.rails["g3_provider"], marker)) == 1
    unknown_marker: Final = f"audit-g3-unknown-{uuid.uuid4().hex}"
    unknown: Final = _raw_chat(
        rig.candidate,
        unknown_marker,
        False,
        "audit-unknown-model",
        rails=(rig.rails["g3_provider"],),
    )
    assert unknown.status_code in (400, 404) and "model" in unknown.text.lower(), unknown.text
    assert _matching_requests(rig.provider, unknown_marker) == ()
    healthy_marker: Final = f"audit-g3-healthy-{uuid.uuid4().hex}"
    healthy: Final = _raw_chat(rig.candidate, healthy_marker, False, rig.models["chat"], key=rig.key)
    assert healthy.status_code == 200 and healthy_marker in healthy.text, healthy.text
    assert len(_matching_requests(rig.provider, healthy_marker)) == 1
