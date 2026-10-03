from __future__ import annotations

import json
import re
import signal
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import psutil
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy_process

_RESPONSES_JSON_SCENARIO: Final = "audit-chaos-responses"
_RESPONSES_STREAM_SCENARIO: Final = "audit-chaos-responses-stream"
_MESSAGES_JSON_SCENARIO: Final = "audit-chaos-messages"
_MESSAGES_STREAM_SCENARIO: Final = "audit-chaos-messages-stream"

_RESPONSES_BODY: Final = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1700000000,
    "status": "completed",
    "model": "openai/gpt-4o-mini",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
}

_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_$UNIQUE_ID",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "scripted reply"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 9, "output_tokens": 5},
}

_ANTHROPIC_EVENTS: Final = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_$UNIQUE_ID","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],"usage":{"input_tokens":9,"output_tokens":0}}}',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"scripted reply"}}',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
    'event: message_stop\ndata: {"type":"message_stop"}',
)

_RESPONSES_EVENTS: Final = (
    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_$UNIQUE_ID","object":"response","created_at":1700000000,"status":"completed","model":"openai/gpt-4o-mini","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}}',
)


def _register_scenarios(upstream_url: str) -> None:
    with httpx.Client(timeout=5, trust_env=False) as client:
        for scenario_id, response in (
            (
                _RESPONSES_JSON_SCENARIO,
                {
                    "content_type": "application/x-routed",
                    "routes": {"POST /responses": {"content_type": "application/json", "body": _RESPONSES_BODY}},
                },
            ),
            (_RESPONSES_STREAM_SCENARIO, {"content_type": "text/event-stream", "frames": _RESPONSES_EVENTS}),
            (
                _MESSAGES_JSON_SCENARIO,
                {
                    "content_type": "application/x-routed",
                    "routes": {"POST /v1/messages": {"content_type": "application/json", "body": _ANTHROPIC_MESSAGE}},
                },
            ),
            (_MESSAGES_STREAM_SCENARIO, {"content_type": "text/event-stream", "frames": _ANTHROPIC_EVENTS}),
        ):
            result: Final = client.post(
                f"{upstream_url}/__scenarios", json={"scenario_id": scenario_id, "response": response}
            )
            assert result.status_code == 200, result.text


def _stream_response_id(text: str) -> str | None:
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        value: Final = json.loads(line.removeprefix("data: "))
        if isinstance(value, dict):
            if isinstance(value.get("id"), str):
                return value["id"]
            for key_name in ("response", "message"):
                nested: Final = value.get(key_name)
                if isinstance(nested, dict) and isinstance(nested.get("id"), str):
                    return nested["id"]
    return None


def _fire(owned: Gateway, models: Mapping[str, str], key: str, index: int) -> tuple[int, str, int | None, str | None]:
    stream: Final = index % 2 == 0
    path: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
    kind: Final = (
        f"{'chat' if path == '/v1/chat/completions' else path.rsplit('/', 1)[-1]}{'-stream' if stream else ''}"
    )
    model: Final = models[f"{path.rsplit('/', 1)[-1]}{'-stream' if stream else ''}"]
    if path == "/v1/messages":
        body: Final = {
            "model": model,
            "messages": [{"role": "user", "content": f"burst-{index}"}],
            "max_tokens": 64,
            "stream": stream,
        }
    elif path == "/v1/responses":
        body = {"model": model, "input": f"burst-{index}", "stream": stream}
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": f"burst-{index}"}],
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }
    try:
        response: Final = owned.request("POST", path, body, key=key)
    except Exception:
        return index, kind, None, None
    if response.status_code != 200:
        return index, kind, response.status_code, None
    if stream:
        return index, kind, 200, _stream_response_id(response.text)
    response_id: Final = response.json().get("id")
    return index, kind, 200, response_id if isinstance(response_id, str) else None


_WORKER_PID: Final = re.compile(r"Started server process \[(\d+)\]")


def _worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(match) for match in _WORKER_PID.findall(log.read_text()))


def test_proxy_survives_worker_kill_mid_burst(tmp_path: Path) -> None:
    with gateway_from_environment() as upstream_gateway:
        directory: Final = tmp_path
        _register_scenarios(upstream_gateway.upstream_url)
        with owned_proxy_process(upstream_gateway, directory, {}, workers=2) as owned:
            with owned.gateway.scenario() as scenario:
                model: Final = scenario.model()
                responses_model: Final = scenario.model(
                    api_base=f"{owned.gateway.upstream_url}/{_RESPONSES_JSON_SCENARIO}"
                )
                responses_stream_model: Final = scenario.model(
                    api_base=f"{owned.gateway.upstream_url}/{_RESPONSES_STREAM_SCENARIO}"
                )
                messages_model: Final = scenario.model(
                    model="anthropic/claude-3-5-sonnet-20241022",
                    api_base=f"{owned.gateway.upstream_url}/{_MESSAGES_JSON_SCENARIO}",
                )
                messages_stream_model: Final = scenario.model(
                    model="anthropic/claude-3-5-sonnet-20241022",
                    api_base=f"{owned.gateway.upstream_url}/{_MESSAGES_STREAM_SCENARIO}",
                )
                key: Final = scenario.key(
                    models=[model, responses_model, responses_stream_model, messages_model, messages_stream_model]
                )
                models: Final = MappingProxyType(
                    {
                        "completions": model,
                        "completions-stream": model,
                        "messages": messages_model,
                        "messages-stream": messages_stream_model,
                        "responses": responses_model,
                        "responses-stream": responses_stream_model,
                    }
                )
                workers: Final = eventually(lambda: _worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=30)
                records: dict[int, tuple[str, int | None, str | None]] = {}
                with ThreadPoolExecutor(max_workers=30) as pool:
                    futures: Final = [pool.submit(_fire, owned.gateway, models, key, index) for index in range(30)]
                    psutil.Process(workers[0]).send_signal(signal.SIGKILL)
                    for future in futures:
                        index, kind, status, response_id = future.result()
                        records[index] = (kind, status, response_id)
                served_ids: Final = [record[2] for record in records.values() if record[2] is not None]
                malformed: Final = {i: r for i, r in records.items() if r[1] == 200 and r[2] is None}
                assert not malformed, f"200 responses without a well-formed body: {malformed}"
                assert len(served_ids) == len(set(served_ids)), records
                kinds_seen: Final = {record[0] for record in records.values()}
                for seen in sorted(kinds_seen):
                    assert any(record[0] == seen and record[2] is not None for record in records.values()), (
                        f"no successful {seen} request survived the worker kill: {records}"
                    )
                probe: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "post-kill probe"}]},
                    key=key,
                )
                assert probe.status_code == 200, probe.text
                assert served_ids, "no burst request completed"
