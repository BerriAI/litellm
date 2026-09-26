import asyncio
import json
import re
import signal
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
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


@dataclass(frozen=True, slots=True)
class _Served:
    response: httpx.Response
    client_port: int


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,))


def _single_spend_row(call_id: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert len(rows) == 1, call_id


async def _fire_burst(
    base_url: str, key: str, count: int, *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> _Served:
        if index % 3 == 0:
            path: Final = "/gemini/v1beta/models/nope-9:generateContent"
        elif index % 3 == 1:
            path = "/gemini/v1beta/models/nope-9:streamGenerateContent?alt=sse"
        else:
            path = "/gemini/v1beta/models/healthy-model:streamGenerateContent?alt=sse"
        async with client.stream(
            "POST",
            path,
            json=_GENERATE_CONTENT,
            headers={"Authorization": f"Bearer {key}", "x-goog-api-key": key},
        ) as response:
            client_port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
            await response.aread()
        return _Served(response=response, client_port=client_port)

    async with httpx.AsyncClient(base_url=base_url, timeout=30, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(one(client, index) for index in range(count)), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


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
        responses: Final = tuple(served.response for served in await burst)
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
    pytest.skip(
        "BUG: a worker SIGKILLed between the last body byte of a passthrough error and its BackgroundTask report loses that response's spend row"
    )
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
            victim: Final = psutil.Process(workers[0])
            victim.suspend()
            victim_ports: Final = frozenset(
                connection.raddr.port for connection in victim.net_connections(kind="tcp") if connection.raddr
            )
            victim.send_signal(signal.SIGKILL)
            served: Final = await burst
            for item in served:
                assert item.response.status_code in (200, 404, 500, 502), item.response.status_code
            follow_up: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/nope-9:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": candidate.key},
            )
            assert follow_up.status_code == 404, follow_up.text
            assert follow_up.json() == json.loads(_NOT_FOUND_BODY), follow_up.text
            logged: Final = tuple(item for item in served if "x-litellm-call-id" in item.response.headers)
            survivor_served: Final = tuple(item for item in logged if item.client_port not in victim_ports)
            assert survivor_served, [item.client_port for item in logged]
            for item in survivor_served:
                _single_spend_row(item.response.headers["x-litellm-call-id"])
            for item in logged:
                assert len(_spend_rows(item.response.headers["x-litellm-call-id"])) <= 1, item.response.headers
            error_information: Final = _error_information(follow_up.headers["x-litellm-call-id"])
            assert "not found for this scripted upstream" in str(error_information["error_message"]), follow_up.text


_RATE_LIMITED_FRAMES: Final = (b'data: {"error":"rate limited"}\n\n', b"data: [DONE]\n\n")


async def test_passthrough_disconnect_burst_logs_every_failure_once(gateway: Gateway, tmp_path: Path) -> None:
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=429,
                content_type="text/event-stream",
                chunks=_RATE_LIMITED_FRAMES,
                gate_after_first=gate,
                gate_timeout_seconds=300,
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "chaos-disconnect-burst.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            try:
                call_ids: Final = await asyncio.gather(
                    *(_first_frame_then_close(str(candidate.client.base_url), candidate.key) for _ in range(20))
                )
                assert len(set(call_ids)) == 20, call_ids
                for call_id in call_ids:
                    _single_spend_row(call_id)
                    error_information: Final = _error_information(call_id)
                    assert error_information["error_code"] == "429", error_information
            finally:
                gate.set()
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


_PARKING_FAILURE_HOOK: Final = """
import asyncio
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


class ParkingFailureHook(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        Path({marker!r}).touch()
        await asyncio.Event().wait()


instance = ParkingFailureHook()
"""


async def test_passthrough_sigterm_with_graceful_timeout_exits_and_flushes_spend(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A streamed upstream error whose failure report can never finish must not hold
    the proxy open: uvicorn cancels the stuck request task after the graceful
    window and the shutdown spend flush still lands rows buffered behind the
    3600 s batch interval."""
    gate: Final = threading.Event()

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=429, content_type="text/event-stream", chunks=_RATE_LIMITED_FRAMES, gate_after_first=gate
            )
        return Reply(
            status=200,
            body=json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
                }
            ).encode(),
        )

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["park_hook.instance"]})
    config["general_settings"]["proxy_batch_write_at"] = 3600
    (tmp_path / "park_hook.py").write_text(_PARKING_FAILURE_HOOK.format(marker=str(tmp_path / "hook_started")))
    path: Final = tmp_path / "chaos-graceful-sigterm.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, graceful_shutdown_seconds=3) as owned:
            candidate: Final = owned.gateway
            await _first_frame_then_close(str(candidate.client.base_url), candidate.key)
            gate.set()
            follow_up: Final = candidate.request(
                "POST",
                "/gemini/v1beta/models/claude-nope-9:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": candidate.key},
            )
            assert follow_up.status_code == 200, follow_up.text
            call_id: Final = follow_up.headers["x-litellm-call-id"]
            await asyncio.to_thread(eventually, lambda: (tmp_path / "hook_started").exists(), bool, 30)
            owned.process.send_signal(signal.SIGTERM)
            owned.process.wait(timeout=20)
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


_MARKER_SLEEP_FAILURE_HOOK: Final = """
import asyncio
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


class MarkerSleepFailureHook(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        Path({marker!r}).touch()
        await asyncio.sleep({sleep})
        Path({done!r}).touch()


instance = MarkerSleepFailureHook()
"""


async def test_passthrough_abort_after_budget_with_disconnect_during_hook_logs_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Post-budget abort: the report await must be shielded, or a client disconnect
    cancels the failure hook mid-flight and no spend row lands."""
    chunks: Final = tuple(b"x" * 512 for _ in range(10))

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(status=500, content_type="text/event-stream", chunks=chunks, abort_after=9)
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["marker_sleep_hook.instance"]})
    (tmp_path / "marker_sleep_hook.py").write_text(
        _MARKER_SLEEP_FAILURE_HOOK.format(
            marker=str(tmp_path / "hook_started"), sleep=3, done=str(tmp_path / "hook_done")
        )
    )
    path: Final = tmp_path / "chaos-abort-budget-disconnect.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=1) as owned:
            candidate: Final = owned.gateway
            received: Final = bytearray()
            async with httpx.AsyncClient(
                base_url=str(candidate.client.base_url),
                timeout=httpx.Timeout(10, read=1, connect=5),
                trust_env=False,
            ) as client:
                try:
                    async with client.stream(
                        "POST",
                        "/gemini/v1beta/models/nope-9:streamGenerateContent",
                        params={"alt": "sse"},
                        json=_GENERATE_CONTENT,
                        headers={"Authorization": f"Bearer {candidate.key}", "x-goog-api-key": candidate.key},
                    ) as response:
                        call_id: Final = response.headers["x-litellm-call-id"]
                        async for chunk in response.aiter_bytes():
                            received.extend(chunk)
                except (httpx.TransportError, httpx.TimeoutException):
                    pass
            assert bytes(received) == b"x" * 4608, len(received)
            await asyncio.to_thread(eventually, lambda: (tmp_path / "hook_started").exists(), bool, 30)
            await asyncio.to_thread(eventually, lambda: (tmp_path / "hook_done").exists(), bool, 30)
            _single_spend_row(call_id)


@pytest.mark.parametrize(
    ("asgi_server", "graceful_seconds", "hook_sleep"),
    [("hypercorn", 3, 4.5), ("uvicorn", 1, 2)],
    ids=["hypercorn", "uvicorn"],
)
async def test_passthrough_error_report_survives_sigterm_after_full_response(
    gateway: Gateway, tmp_path: Path, asgi_server: str, graceful_seconds: int, hook_sleep: float
) -> None:
    """The graceful window cancels the request task, but the report task is shielded
    and the lifespan shutdown waits for it before the spend flushes."""

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(status=429, content_type="text/event-stream", chunks=_RATE_LIMITED_FRAMES)
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["marker_sleep_hook.instance"]})
    (tmp_path / "marker_sleep_hook.py").write_text(
        _MARKER_SLEEP_FAILURE_HOOK.format(
            marker=str(tmp_path / "hook_started"), sleep=hook_sleep, done=str(tmp_path / "hook_done")
        )
    )
    path: Final = tmp_path / f"chaos-sigterm-{asgi_server}.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(
            gateway,
            tmp_path,
            {},
            config=path,
            graceful_shutdown_seconds=graceful_seconds,
            asgi_server=asgi_server,
        ) as owned:
            candidate: Final = owned.gateway
            async with httpx.AsyncClient(
                base_url=str(candidate.client.base_url), timeout=httpx.Timeout(15, connect=5), trust_env=False
            ) as client:
                async with client.stream(
                    "POST",
                    "/gemini/v1beta/models/nope-9:streamGenerateContent",
                    params={"alt": "sse"},
                    json=_GENERATE_CONTENT,
                    headers={"Authorization": f"Bearer {candidate.key}", "x-goog-api-key": candidate.key},
                ) as response:
                    call_id: Final = response.headers["x-litellm-call-id"]
                    body: Final = b"".join([chunk async for chunk in response.aiter_bytes()])
            assert body == b"".join(_RATE_LIMITED_FRAMES), body
            await asyncio.to_thread(eventually, lambda: (tmp_path / "hook_started").exists(), bool, 30)
            owned.process.send_signal(signal.SIGTERM)
            owned.process.wait(timeout=60)
    await asyncio.to_thread(eventually, lambda: (tmp_path / "hook_done").exists(), bool, 30)
    _single_spend_row(call_id)


_BUDGET_MARKER_FAILURE_HOOK: Final = """
import asyncio
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


class BudgetMarkerFailureHook(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        call_id = (request_data or {{}}).get("litellm_call_id") or "unknown"
        Path({directory!r}, f"started-{{call_id}}").touch()
        await asyncio.sleep(12)
        Path({directory!r}, f"done-{{call_id}}").touch()


instance = BudgetMarkerFailureHook()
"""

_BUDGET_CHUNK: Final = b"e" * 8000


async def test_passthrough_sigterm_graceful_window_reports_dispatched_at_preview_budget(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The upstream sends 8000 bytes then stalls for 60 s: the failure report must
    dispatch when the preview budget is crossed, so a graceful shutdown can drain
    the 12 s hooks it would otherwise find still buffered behind the gate."""
    gate: Final = threading.Event()
    markers: Final = tmp_path / "budget-markers"
    markers.mkdir()

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=500,
                content_type="text/event-stream",
                chunks=(_BUDGET_CHUNK, b"data: tail\n\n"),
                gate_after_first=gate,
                gate_timeout_seconds=120,
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["budget_hook.instance"]})
    (tmp_path / "budget_hook.py").write_text(_BUDGET_MARKER_FAILURE_HOOK.format(directory=str(markers)))
    path: Final = tmp_path / "chaos-budget-sigterm.yaml"
    path.write_text(yaml.safe_dump(config))
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, graceful_shutdown_seconds=20) as owned:

            async def hold_stream() -> str | None:
                async with httpx.AsyncClient(
                    base_url=str(owned.gateway.client.base_url), timeout=httpx.Timeout(30, connect=5), trust_env=False
                ) as client:
                    try:
                        async with client.stream(
                            "POST",
                            "/gemini/v1beta/models/nope-9:streamGenerateContent",
                            params={"alt": "sse"},
                            json=_GENERATE_CONTENT,
                            headers={
                                "Authorization": f"Bearer {owned.gateway.key}",
                                "x-goog-api-key": owned.gateway.key,
                            },
                        ) as response:
                            return response.headers["x-litellm-call-id"]
                    except (httpx.TransportError, httpx.TimeoutException):
                        return None

            try:
                streams: Final = asyncio.gather(*(hold_stream() for _ in range(20)), return_exceptions=True)
                await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 20, 60)
                owned.process.send_signal(signal.SIGTERM)
                owned.process.wait(timeout=90)
                await streams
            finally:
                gate.set()
    done_markers: Final = tuple(markers.glob("done-*"))
    assert len(done_markers) == 20, sorted(p.name for p in markers.iterdir())
    call_ids: Final = [call_id for call_id in await streams if isinstance(call_id, str)]
    assert len(call_ids) == 20, call_ids
    for call_id in call_ids:
        _single_spend_row(call_id)


_SLOW_HEADERS_HOOK: Final = """
import asyncio
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


class SlowHeadersHook(CustomLogger):
    async def async_post_call_response_headers_hook(
        self, data, user_api_key_dict, response, request_headers, **kwargs
    ):
        await asyncio.sleep(15)
        return {{}}

    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        call_id = (request_data or {{}}).get("litellm_call_id") or "unknown"
        Path({directory!r}, f"failure-reported-{{call_id}}").touch()


instance = SlowHeadersHook()
"""


async def test_passthrough_sigterm_during_headers_hook_still_dispatches_the_report(
    gateway: Gateway, tmp_path: Path
) -> None:
    """SIGTERM cancelling the request task mid-headers-hook is an exit between
    building the relay and returning its StreamingResponse: the relay must
    dispatch its report there instead of losing the failure."""
    markers: Final = tmp_path / "headers-markers"
    markers.mkdir()
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=500,
                content_type="text/event-stream",
                chunks=(f"data: upstream blew up {marker}\n\n".encode(), b"data: [DONE]\n\n"),
                delay_before_headers_seconds=1,
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["slow_headers_hook.instance"]})
    (tmp_path / "slow_headers_hook.py").write_text(_SLOW_HEADERS_HOOK.format(directory=str(markers)))
    path: Final = tmp_path / "chaos-headers-hook-sigterm.yaml"
    path.write_text(yaml.safe_dump(config))
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path, graceful_shutdown_seconds=3) as owned:

            async def stream_error() -> None:
                async with httpx.AsyncClient(
                    base_url=str(owned.gateway.client.base_url), timeout=httpx.Timeout(60, connect=5), trust_env=False
                ) as client:
                    try:
                        await client.post(
                            "/gemini/v1beta/models/nope-9:streamGenerateContent",
                            params={"alt": "sse"},
                            json=_GENERATE_CONTENT,
                            headers={
                                "Authorization": f"Bearer {owned.gateway.key}",
                                "x-goog-api-key": owned.gateway.key,
                            },
                        )
                    except (httpx.TransportError, httpx.TimeoutException):
                        pass

            request_task: Final = asyncio.create_task(stream_error())
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 1, 30)
            owned.process.send_signal(signal.SIGTERM)
            owned.process.wait(timeout=30)
            await request_task
            reported: Final = await asyncio.to_thread(
                eventually,
                lambda: tuple(markers.glob("failure-reported-*")),
                lambda paths: len(paths) == 1,
                30,
            )
            call_id: Final = reported[0].name.removeprefix("failure-reported-")
            _single_spend_row(call_id)
            error_information: Final = _error_information(call_id)
            assert error_information["error_code"] == "500", error_information


_PARKING_HEADERS_HOOK_FAILURE_ONLY: Final = """
import asyncio
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger


class NeverReturningFailureHook(CustomLogger):
    async def async_post_call_failure_hook(
        self, request_data, original_exception, user_api_key_dict, traceback_str=None
    ):
        Path({marker!r}).touch()
        await asyncio.Event().wait()


instance = NeverReturningFailureHook()
"""


async def test_passthrough_sigterm_exits_when_late_relay_background_never_returns(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Keepalive replays the late response's background task inside aclose: a
    failure hook that never returns must be abandoned after the bound so SIGTERM
    still gets the process out."""
    hook_started: Final = tmp_path / "late-relay-hook-started"

    def respond(request: Request) -> Reply:
        if "streamGenerateContent" in request.target:
            return Reply(
                status=500,
                content_type="text/event-stream",
                chunks=(b"data: err\n\n", b"data: [DONE]\n\n"),
                delay_before_headers_seconds=3,
            )
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["park_hook.instance"], "sse_keepalive_ping_interval_seconds": 0.2})
    (tmp_path / "park_hook.py").write_text(_PARKING_HEADERS_HOOK_FAILURE_ONLY.format(marker=str(hook_started)))
    path: Final = tmp_path / "chaos-late-relay-bound.yaml"
    path.write_text(yaml.safe_dump(config))
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path) as owned:
            async with httpx.AsyncClient(
                base_url=str(owned.gateway.client.base_url),
                timeout=httpx.Timeout(10, read=1.5, connect=5),
                trust_env=False,
            ) as client:
                try:
                    async with client.stream(
                        "POST",
                        "/gemini/v1beta/models/nope-9:streamGenerateContent",
                        params={"alt": "sse"},
                        json=_GENERATE_CONTENT,
                        headers={
                            "Authorization": f"Bearer {owned.gateway.key}",
                            "x-goog-api-key": owned.gateway.key,
                        },
                    ) as response:
                        await response.aread()
                except (httpx.TransportError, httpx.TimeoutException):
                    pass
            await asyncio.to_thread(eventually, lambda: hook_started.exists(), bool, 60)
            owned.process.send_signal(signal.SIGTERM)
            owned.process.wait(timeout=40)
    log_text: Final = owned.log.read_text()
    assert "relayed response background task still running after 10s" in log_text, log_text[-4000:]
    assert "upstream error reports still running after 10s shutdown wait" in log_text, log_text[-4000:]


_BIG_ERROR_BODY: Final = (
    '{"error":{"code":500,"message":"upstream blew up with key sk-leak0leak0leak0leak0 then '
    + "x" * (5 * 1024 * 1024)
    + '","status":"INTERNAL"}}'
).encode()


def test_passthrough_buffered_error_body_redaction_stays_bounded(gateway: Gateway, tmp_path: Path) -> None:
    """A 5 MiB buffered error body must not be fully decoded and redacted: the
    request returns fast and the spend row still carries the redacted prefix."""

    def respond(request: Request) -> Reply:
        if "generateContent" in request.target:
            return Reply(status=500, body=_BIG_ERROR_BODY)
        return Reply(status=200, body=json.dumps({"ok": True}).encode())

    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = tmp_path / "chaos-big-error-body.yaml"
    with wire_server(respond) as wire:
        config["environment_variables"] = {"GEMINI_API_BASE": wire.url, "GEMINI_API_KEY": "scripted"}
        path.write_text(yaml.safe_dump(config))
        with owned_proxy_process(gateway, tmp_path, {}, config=path) as owned:
            started: Final = time.perf_counter()
            response: Final = owned.gateway.request(
                "POST",
                "/gemini/v1beta/models/nope-9:generateContent",
                _GENERATE_CONTENT,
                headers={"x-goog-api-key": owned.gateway.key},
            )
            elapsed: Final = time.perf_counter() - started
            assert response.status_code == 500, response.status_code
            call_id: Final = response.headers["x-litellm-call-id"]
            _single_spend_row(call_id)
            error_information: Final = _error_information(call_id)
            assert "REDACTED" in str(error_information["error_message"]), error_information
    assert elapsed < 0.5, elapsed
