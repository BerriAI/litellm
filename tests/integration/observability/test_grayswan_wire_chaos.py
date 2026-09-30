import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import Gateway
from integration._support.process import group_members, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from test_grayswan_wire import _PROVIDER_KEY, _REQUEST_MESSAGES, _VENDOR_KEY, _monitor_bodies, _serving_model_probe

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _chaos_config(tmp_path: Path, identity: str, vendor_url: str, *, fail_open: bool = True) -> Path:
    config: Final = {
        **yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()),
        "guardrails": [
            {
                "guardrail_name": identity,
                "litellm_params": {
                    "guardrail": "grayswan",
                    "mode": "post_call",
                    "default_on": True,
                    "api_base": vendor_url,
                    "api_key": _VENDOR_KEY,
                    "streaming_end_of_stream_only": True,
                    "optional_params": {
                        "on_flagged_action": "monitor",
                        "violation_threshold": 0.5,
                        "policy_id": "synthetic-policy",
                        "fail_open": fail_open,
                    },
                },
            }
        ],
    }
    path: Final = tmp_path / f"{identity}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _provider(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    marker: Final = next(
        (
            str(message.get("content"))
            for message in body.get("messages", [])
            if isinstance(message, dict) and str(message.get("content", "")).startswith("marker-")
        ),
        "none",
    )
    if body.get("stream"):
        frames: Final = (
            b'data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n',
            f'data: {{"id":"chatcmpl-c","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{{"index":0,"delta":{{"content":"echo {marker}"}}}}]}}\n\n'.encode(),
            b'data: {"id":"chatcmpl-c","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        )
        return Reply(content_type="text/event-stream", chunks=frames)
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-chaos",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo {marker}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()
    )


def _fire(candidate: Gateway, model: str, marker: str, stream: bool) -> int:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "max_tokens": 16,
            "stream": stream,
            "messages": [
                dict(_REQUEST_MESSAGES[0]),
                {"role": "user", "content": marker},
                *[dict(message) for message in _REQUEST_MESSAGES[2:]],
            ],
        },
    )
    response.read()
    return response.status_code


def _body_markers(body: dict[str, JsonValue]) -> tuple[str, ...]:
    messages: Final = body.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(
        str(message.get("content"))
        for message in messages
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and message["content"].startswith("marker-")
    )


def test_vendor_outage_mid_burst_no_duplicate_monitor_calls(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "grayswan" + uuid.uuid4().hex
    up: Final = threading.Event()
    up.set()

    def vendor(request: Request) -> Reply:
        assert request.target == "/cygnal/monitor", request.target
        assert request.headers["grayswan-api-key"] == _VENDOR_KEY
        if not up.is_set():
            return Reply(status=503, body=b'{"error":"sink down"}')
        return Reply(body=b'{"violation":0.0}')

    with wire_server(vendor) as vendor_wire, wire_server(_serving_model_probe(_provider)) as upstream:
        config_path: Final = _chaos_config(tmp_path, identity, vendor_wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config_path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url, api_key=_PROVIDER_KEY)
                with ThreadPoolExecutor(max_workers=10) as pool:
                    before: Final = tuple(
                        pool.map(lambda i: _fire(candidate, model, f"marker-up-{i}", i < 2), range(8))
                    )
                    assert all(status == 200 for status in before), before
                    first_bodies: Final = _monitor_bodies(vendor_wire, expected=8)
                    up.clear()
                    during: Final = tuple(
                        pool.map(lambda i: _fire(candidate, model, f"marker-down-{i}", i < 2), range(8))
                    )
                    assert all(status == 200 for status in during), during
                    up.set()
                    after: Final = tuple(
                        pool.map(lambda i: _fire(candidate, model, f"marker-post-{i}", i < 2), range(8))
                    )
                    assert all(status == 200 for status in after), after
                    rest_bodies: Final = _monitor_bodies(vendor_wire, expected=16, seconds=50)
                    bodies: Final = (*first_bodies, *rest_bodies)
                observed: Final = tuple(marker for body in bodies for marker in _body_markers(body))
                unique: Final = frozenset(observed)
                assert len(observed) == len(unique), observed
                for index in range(8):
                    assert f"marker-up-{index}" in unique, observed
                    assert f"marker-post-{index}" in unique, observed
                for body in bodies:
                    messages: Final = body["messages"]
                    assert isinstance(messages, list) and len(messages) >= 2, body
                    assert any(isinstance(message, dict) and message.get("role") == "tool" for message in messages), (
                        body
                    )


def test_slow_vendor_burst_completes_without_deadlock(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "grayswan" + uuid.uuid4().hex

    def slow_vendor(request: Request) -> Reply:
        assert request.target == "/cygnal/monitor", request.target
        time.sleep(2)
        return Reply(body=b'{"violation":0.0}')

    with wire_server(slow_vendor) as vendor, wire_server(_serving_model_probe(_provider)) as upstream:
        config_path: Final = _chaos_config(tmp_path, identity, vendor.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config_path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url, api_key=_PROVIDER_KEY)
                with ThreadPoolExecutor(max_workers=10) as pool:
                    statuses: Final = tuple(
                        pool.map(lambda i: _fire(candidate, model, f"marker-slow-{i}", False), range(10))
                    )
                assert all(status == 200 for status in statuses), statuses
                bodies: Final = _monitor_bodies(vendor, expected=10)
                assert len(bodies) == 10, bodies
                for body in bodies:
                    assert _body_markers(body), body


def test_worker_kill_mid_burst_survivor_keeps_serving(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "grayswan" + uuid.uuid4().hex

    def vendor(request: Request) -> Reply:
        return Reply(body=b'{"violation":0.0}')

    with wire_server(vendor) as vendor_wire, wire_server(_serving_model_probe(_provider)) as upstream:
        config_path: Final = _chaos_config(tmp_path, identity, vendor_wire.url)
        with owned_proxy_process(gateway, tmp_path, {}, config=config_path, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as scenario:
                model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=upstream.url, api_key=_PROVIDER_KEY)
                warm: Final = _fire(candidate, model, "marker-warm", False)
                assert warm == 200
                members: Final = group_members(owned.process.pid)
                children: Final = tuple(member for member in members if member.pid != owned.process.pid)
                assert len(children) >= 2, [member.pid for member in members]
                os.kill(children[0].pid, signal.SIGKILL)
                statuses: Final = tuple(_fire(candidate, model, f"marker-kill-{index}", False) for index in range(6))
                assert all(status == 200 for status in statuses), statuses
                bodies: Final = _monitor_bodies(vendor_wire, expected=7)
                kill_bodies: Final = [
                    body for body in bodies if any(m.startswith("marker-kill-") for m in _body_markers(body))
                ]
                assert len(kill_bodies) == 6, bodies
                for body in kill_bodies:
                    messages: Final = body["messages"]
                    assert isinstance(messages, list) and len(messages) >= 2, body
