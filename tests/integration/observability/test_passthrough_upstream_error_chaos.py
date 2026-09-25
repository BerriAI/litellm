import asyncio
import json
import re
import signal
import threading
from pathlib import Path
from typing import Final

import httpx
import psutil
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_GENERATE_CONTENT: Final[dict[str, JsonValue]] = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
_NOT_FOUND_BODY: Final = json.dumps(
    {
        "error": {
            "code": 404,
            "message": "models/nope-9 is not found for this scripted upstream",
            "status": "NOT_FOUND",
        }
    }
).encode()
_INTERNAL_BODY: Final = (
    '{"error":{"code":500,"message":"' + "chunked upstream failure body " * 200 + '","status":"INTERNAL"}}'
).encode()
_OK_CHUNKS: Final = tuple(f"data: ok-{index}\n\n".encode() for index in range(3))
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


def _chaos_reply(request: Request) -> Reply:
    if "streamGenerateContent" in request.target:
        return Reply(status=500, chunks=tuple(_INTERNAL_BODY[i : i + 512] for i in range(0, len(_INTERNAL_BODY), 512)))
    if "healthy-model" in request.target:
        return Reply(status=200, chunks=_OK_CHUNKS, content_type="text/event-stream")
    return Reply(status=404, body=_NOT_FOUND_BODY)


def _error_information(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    return object_value(parsed["error_information"])


def _single_spend_row(call_id: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert len(rows) == 1, call_id


async def _fire_burst(
    base_url: str, key: str, count: int, *, tolerate_transport_errors: bool = False
) -> tuple[httpx.Response, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> httpx.Response:
        if index % 3 == 0:
            path: Final = "/gemini/v1beta/models/nope-9:generateContent"
        elif index % 3 == 1:
            path = "/gemini/v1beta/models/nope-9:streamGenerateContent?alt=sse"
        else:
            path = "/gemini/v1beta/models/healthy-model:streamGenerateContent?alt=sse"
        return await client.post(
            path,
            json=_GENERATE_CONTENT,
            headers={"Authorization": f"Bearer {key}", "x-goog-api-key": key},
        )

    async with httpx.AsyncClient(base_url=base_url, timeout=30, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(one(client, index) for index in range(count)), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, httpx.Response))


async def test_passthrough_upstream_outage_mid_burst_still_logs_errors_once(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "chaos-outage.yaml"
    with wire_server(_chaos_reply) as wire:
        port: Final = int(wire.url.rsplit(":", 1)[1])
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            burst: Final = asyncio.create_task(_fire_burst(str(candidate.client.base_url), candidate.key, 30))
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 10, 30)
    with wire_server(_chaos_reply, port=port):
        responses: Final = await burst
        assert len(responses) == 30
        for response in responses:
            assert response.status_code in (200, 404, 500, 502), response.status_code
            assert "x-litellm-call-id" in response.headers, response.status_code
        assert len(_STARTED_WORKER.findall(owned.log.read_text())) >= 2
        for response in responses:
            _single_spend_row(response.headers["x-litellm-call-id"])
            if response.status_code == 404:
                error_information: Final = _error_information(response.headers["x-litellm-call-id"])
                assert "not found for this scripted upstream" in str(error_information["error_message"]), response.text
            elif response.status_code == 500:
                assert "chunked upstream failure body" in str(
                    _error_information(response.headers["x-litellm-call-id"])["error_message"]
                ), response.text


async def test_passthrough_worker_sigkill_leaves_sibling_serving_and_logging(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "chaos-kill.yaml"
    with wire_server(_chaos_reply) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _fire_burst(str(candidate.client.base_url), candidate.key, 20, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 5, 30)
            psutil.Process(workers[0]).send_signal(signal.SIGKILL)
            responses: Final = await burst
            for response in responses:
                assert response.status_code in (200, 404, 500, 502), response.status_code
            follow_up: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/nope-9:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": candidate.key},
            )
            assert follow_up.status_code == 404, follow_up.text
            assert follow_up.json() == json.loads(_NOT_FOUND_BODY), follow_up.text
            for response in responses:
                if "x-litellm-call-id" in response.headers:
                    _single_spend_row(response.headers["x-litellm-call-id"])
            error_information: Final = _error_information(follow_up.headers["x-litellm-call-id"])
            assert "not found for this scripted upstream" in str(error_information["error_message"]), follow_up.text


_RATE_LIMITED_FRAMES: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")


async def test_passthrough_disconnect_burst_logs_every_failure_once(gateway: Gateway, tmp_path: Path) -> None:
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=429, content_type="text/event-stream", chunks=_RATE_LIMITED_FRAMES, gate_after_first=gate
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "chaos-disconnect-burst.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            call_ids: Final = await asyncio.gather(
                *(_first_frame_then_close(str(candidate.client.base_url), candidate.key) for _ in range(20))
            )
            assert len(set(call_ids)) == 20, call_ids
            gate.set()
            for call_id in call_ids:
                _single_spend_row(call_id)
                error_information: Final = _error_information(call_id)
                assert error_information["error_code"] == "429", error_information
            eventually(
                lambda: tuple(line for line in owned.log.read_text().splitlines() if "returned 429" in line),
                lambda lines: len(lines) == 20,
                seconds=60,
            )
            follow_up: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/nope-9:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": candidate.key},
            )
            assert follow_up.status_code == 200, follow_up.text


_SLOW_FAILURE_HOOK: Final = """
import asyncio

from litellm.integrations.custom_logger import CustomLogger


class SlowFailureHook(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        await asyncio.sleep(15)


instance = SlowFailureHook()
"""


async def test_passthrough_sigterm_drains_reports_parked_on_a_slow_failure_hook(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=429, content_type="text/event-stream", chunks=_RATE_LIMITED_FRAMES, gate_after_first=gate
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["slow_hook.instance"]})
    (tmp_path / "slow_hook.py").write_text(_SLOW_FAILURE_HOOK)
    path: Final = tmp_path / "chaos-sigterm-drain.yaml"
    path.write_text(yaml.safe_dump(config))
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=1) as owned:
            candidate: Final = owned.gateway
            call_ids: Final = await asyncio.gather(
                *(_first_frame_then_close(str(candidate.client.base_url), candidate.key) for _ in range(20))
            )
            assert len(set(call_ids)) == 20, call_ids
            gate.set()
            owned.process.send_signal(signal.SIGTERM)
            owned.process.wait(timeout=90)
            for call_id in call_ids:
                _single_spend_row(call_id)


async def _first_frame_then_close(base_url: str, key: str) -> str:
    async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(5, connect=5), trust_env=False) as client:
        async with client.stream(
            "POST",
            "/gemini/v1beta/models/nope-9:streamGenerateContent",
            params={"alt": "sse"},
            json=_GENERATE_CONTENT,
            headers={"Authorization": f"Bearer {key}", "x-goog-api-key": key},
        ) as response:
            assert response.status_code == 429, response.status_code
            first: Final = await response.aiter_bytes().__anext__()
            assert first.startswith(b'data: {"error":"rate limited"}'), first
            return response.headers["x-litellm-call-id"]
