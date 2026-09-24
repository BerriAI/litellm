"""The proxy serves chat completions, messages and responses with grpcio, polars and google.cloud absent."""

import json
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, eventually, gateway_from_environment, object_value
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse

SITECUSTOMIZE: Final = """
import importlib.abc
import sys

_BLOCKED_TOP_LEVEL = ("grpc", "polars")
_BLOCKED_GOOGLE = ("google.cloud", "google.api_core")


class _BlockedImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in _BLOCKED_TOP_LEVEL:
            raise ModuleNotFoundError(f"blocked by the no-grpc-stack test: {fullname}")
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in _BLOCKED_GOOGLE):
            raise ModuleNotFoundError(f"blocked by the no-grpc-stack test: {fullname}")
        return None


sys.meta_path.insert(0, _BlockedImports())
"""


@contextmanager
def _python_without_grpc_stack(directory: Path) -> Generator[str, None, None]:
    shim: Final = directory / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text(SITECUSTOMIZE)
    pythonpath: Final = f"{shim}{os.pathsep}{os.environ['PYTHONPATH']}"
    probe: Final = subprocess.run(
        [sys.executable, "-P", "-c", "import grpc"],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=60,
    )
    output: Final = probe.stdout + probe.stderr
    assert probe.returncode != 0 and "grpc" in output, (
        f"The import shim did not block grpc, so this test would prove nothing: {output}"
    )
    yield pythonpath


@pytest.fixture(scope="module")
def proxy_without_grpc_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("no-grpc-stack")
    with _python_without_grpc_stack(directory) as pythonpath, gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, {"PYTHONPATH": pythonpath}, workers=2) as owned:
            yield owned


def test_chat_completions_serves_without_the_grpc_stack(proxy_without_grpc_stack: OwnedProxy) -> None:
    owned: Final = proxy_without_grpc_stack
    with (
        owned.gateway.scenario() as scenario,
        httpx.Client(base_url=owned.gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        upstream.get("/__observations").raise_for_status()
        response: Final = owned.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "no grpc stack"}]},
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-"), response.text
        observed: Final = upstream.get("/__observations")
        observed.raise_for_status()
        observed_requests: Final = JSON_OBJECT.validate_json(observed.content)["requests"]
        assert isinstance(observed_requests, list) and len(observed_requests) == 1, observed_requests
        received: Final = object_value(observed_requests[0])
        assert received["path"] == "/v1/chat/completions", observed_requests
        received_body: Final = object_value(received["body"])
        assert received_body["model"] == "gpt-4o-mini", observed_requests
        assert received_body["messages"] == [{"role": "user", "content": "no grpc stack"}], observed_requests
    log: Final = owned.log.read_text()
    leaked: Final = tuple(
        line for line in log.splitlines() if "ModuleNotFoundError" in line and ("grpc" in line or "polars" in line)
    )
    assert not leaked, f"Proxy logged a blocked import while serving without the grpc stack: {leaked}"


def test_streaming_chat_completion_serves_without_the_grpc_stack(proxy_without_grpc_stack: OwnedProxy) -> None:
    owned: Final = proxy_without_grpc_stack
    scenario_id: Final = f"nogrpcstream{uuid.uuid4().hex}"
    chunk: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{scenario_id}",
        "object": "chat.completion.chunk",
        "created": 1789788253,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "streamed without grpc"}, "finish_reason": None}
        ],
    }
    done: Final[dict[str, JsonValue]] = {
        **chunk,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    handle: Final = register_scenario(
        scenario_id,
        SseResponse(
            content_type="text/event-stream",
            frames=(f"data: {json.dumps(chunk)}", f"data: {json.dumps(done)}", "data: [DONE]"),
        ),
    )
    try:
        with owned.gateway.scenario() as scenario:
            model: Final = scenario.model(api_base=f"{handle.api_base()}/v1")
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "stream without grpc"}], "stream": True},
            )
            assert response.status_code == 200, response.text
            chunks: Final = tuple(
                JSON_OBJECT.validate_json(line.removeprefix("data: ").encode())
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            assert chunks, response.text
            assert chunks[-1]["id"] == f"chatcmpl-{scenario_id}", response.text
    finally:
        delete_scenario(handle)


def test_messages_and_responses_serve_without_the_grpc_stack(proxy_without_grpc_stack: OwnedProxy) -> None:
    owned: Final = proxy_without_grpc_stack
    scenario_id: Final = f"nogrpc{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        scenario_id,
        JsonResponse(
            content_type="application/json",
            body={
                "id": "resp_$REQUEST_ID",
                "object": "response",
                "created_at": 1789788253,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_upstream",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "served without grpc", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        ),
    )
    try:
        with owned.gateway.scenario() as scenario:
            model: Final = scenario.model(api_base=handle.api_base())
            responses: Final = owned.gateway.request(
                "POST", "/v1/responses", {"model": model, "input": "respond without grpc"}
            )
            assert responses.status_code == 200, responses.text
            responses_body: Final = JSON_OBJECT.validate_json(responses.content)
            response_id: Final = responses_body["id"]
            assert isinstance(response_id, str) and response_id.startswith("resp_"), responses.text
            output: Final = responses_body["output"]
            assert isinstance(output, list) and output, responses.text
            output_content: Final = object_value(output[0])["content"]
            assert isinstance(output_content, list) and output_content, responses.text
            assert object_value(output_content[0])["text"] == "served without grpc", responses.text
            messages: Final = owned.gateway.request(
                "POST",
                "/v1/messages",
                {"model": model, "max_tokens": 64, "messages": [{"role": "user", "content": "respond without grpc"}]},
            )
            assert messages.status_code == 200, messages.text
            payload: Final = JSON_OBJECT.validate_json(messages.content)
            assert isinstance(payload["id"], str) and payload["id"], messages.text
            content: Final = payload["content"]
            assert isinstance(content, list) and len(content) == 1, messages.text
            block: Final = object_value(content[0])
            assert block["type"] == "text" and block["text"] == "served without grpc", messages.text
    finally:
        delete_scenario(handle)


def _worker_gone(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return True
    return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_surviving_worker_keeps_serving_without_the_grpc_stack(proxy_without_grpc_stack: OwnedProxy) -> None:
    owned: Final = proxy_without_grpc_stack
    launcher: Final = psutil.Process(owned.process.pid)
    children: Final = launcher.children(recursive=True)
    workers: Final = tuple(child for child in children if any("spawn_main" in arg for arg in child.cmdline()))
    assert len(workers) == 2, [(child.pid, child.name(), child.cmdline()) for child in children]
    killed, survivor = workers
    os.kill(killed.pid, signal.SIGKILL)
    eventually(lambda: _worker_gone(killed.pid), lambda gone: gone, seconds=30)
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model()
        responses: Final = tuple(
            owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"surviving worker {attempt}"}]},
            )
            for attempt in range(5)
        )
        bodies: Final = tuple(JSON_OBJECT.validate_json(response.content) for response in responses)
        for response, body in zip(responses, bodies):
            assert response.status_code == 200, response.text
            assert isinstance(body["id"], str) and body["id"].startswith("chatcmpl-"), response.text
    assert psutil.pid_exists(survivor.pid) and psutil.Process(survivor.pid).is_running()
