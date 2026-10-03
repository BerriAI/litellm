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

_RESPONSES_BODY: Final = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1700000000,
    "status": "completed",
    "model": "openai/gpt-4o-mini",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
    "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
}

_RESPONSES_EVENTS: Final = (
    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_$UNIQUE_ID","object":"response","created_at":1700000000,"status":"completed","model":"openai/gpt-4o-mini","output":[{"type":"message","content":[{"type":"output_text","text":"done"}]}],"usage":{"input_tokens":3,"output_tokens":4,"total_tokens":7}}',
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
        ):
            result: Final = client.post(
                f"{upstream_url}/__scenarios", json={"scenario_id": scenario_id, "response": response}
            )
            assert result.status_code == 200, result.text


def _fire(owned: Gateway, models: Mapping[str, str], key: str, index: int) -> tuple[int, str | None]:
    stream: Final = index % 2 == 0
    path: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
    model: Final = models["responses-stream" if path == "/v1/responses" and stream else path.rsplit("/", 1)[-1]]
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
        return index, None
    if response.status_code != 200:
        return index, None
    if stream:
        for line in response.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                value: Final = json.loads(line.removeprefix("data: "))
                if isinstance(value, dict) and isinstance(value.get("id"), str):
                    return index, value["id"]
        return index, None
    response_id: Final = response.json().get("id")
    return index, response_id if isinstance(response_id, str) else None


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
                    api_base=f"{owned.gateway.upstream_url}/{_RESPONSES_JSON_SCENARIO}/v1"
                )
                responses_stream_model: Final = scenario.model(
                    api_base=f"{owned.gateway.upstream_url}/{_RESPONSES_STREAM_SCENARIO}/v1"
                )
                key: Final = scenario.key(models=[model, responses_model, responses_stream_model])
                models: Final = MappingProxyType(
                    {
                        "completions": model,
                        "messages": model,
                        "responses": responses_model,
                        "responses-stream": responses_stream_model,
                    }
                )
                workers: Final = eventually(lambda: _worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=30)
                results: dict[int, str | None] = {}
                with ThreadPoolExecutor(max_workers=30) as pool:
                    futures: Final = [pool.submit(_fire, owned.gateway, models, key, index) for index in range(30)]
                    psutil.Process(workers[0]).send_signal(signal.SIGKILL)
                    for future in futures:
                        index, response_id = future.result()
                        results[index] = response_id
                served_ids: Final = [value for value in results.values() if value is not None]
                assert len(served_ids) == len(set(served_ids)), results
                probe: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "post-kill probe"}]},
                    key=key,
                )
                assert probe.status_code == 200, probe.text
                assert served_ids, "no burst request completed"
