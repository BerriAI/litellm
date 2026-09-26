import asyncio
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.liteadmin.endpoints import StartRequest, WorkerStart, inference_url, router

ADMIN_KEY: Final = "sk-liteadmin-unit-test"
START: Final = StartRequest.model_validate(
    {
        "access_token": ADMIN_KEY,
        "chat": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Check the team budget"}],
            "inference_base_url": "http://testserver",
        },
    }
)
REQUEST_ERROR: Final = {
    "type": "error",
    "message": "Invalid LiteAdmin request or gateway settings. Reload the page and try again.",
}
RUN_ERROR: Final = {
    "type": "error",
    "message": "LiteAdmin could not complete this request. Check your access and model, and review completed actions before continuing.",
}


class CallSpy(Protocol):
    def assert_not_awaited(self) -> None: ...
    def assert_called_once(self) -> None: ...
    def assert_called_once_with(self, *args: object) -> None: ...
    def configure_mock(self, **kwargs: object) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerBoundary:
    spawn: CallSpy
    write: CallSpy
    close: CallSpy
    readline: CallSpy
    wait: CallSpy
    terminate: CallSpy


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setattr(proxy_server, "master_key", ADMIN_KEY)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setenv("LITEADMIN_RUNTIME", str(tmp_path))
    monkeypatch.setenv("LITEADMIN_PYTHON", sys.executable)
    for name in ("LITEADMIN_GATEWAY_URL", "LITELLM_UI_API_DOC_BASE_URL", "PROXY_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    app: Final = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def process(monkeypatch: pytest.MonkeyPatch) -> WorkerBoundary:
    write: Final = MagicMock()
    close: Final = MagicMock()
    readline: Final = AsyncMock(side_effect=(b'{"type":"done"}\n', b""))
    wait: Final = AsyncMock(return_value=0)
    terminate: Final = MagicMock()
    worker: Final = MagicMock(
        spec=asyncio.subprocess.Process,
        stdin=MagicMock(spec=asyncio.StreamWriter, write=write, drain=AsyncMock(), close=close),
        stdout=MagicMock(spec=asyncio.StreamReader, readline=readline),
        wait=wait,
        terminate=terminate,
        returncode=0,
    )
    spawn: Final = AsyncMock(return_value=worker)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return WorkerBoundary(spawn, write, close, readline, wait, terminate)


@pytest.mark.parametrize("payload", ("not json", "{}", "x" * 180_001))
def test_malformed_or_oversized_start_is_rejected_before_starting_worker(
    client: TestClient, process: WorkerBoundary, payload: str
) -> None:
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_text(payload)
        assert websocket.receive_json() == REQUEST_ERROR
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
    process.spawn.assert_not_awaited()


def test_invalid_credentials_cannot_start_a_worker(client: TestClient, process: WorkerBoundary) -> None:
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": "sk-invalid"})
        assert websocket.receive_json() == RUN_ERROR
    process.spawn.assert_not_awaited()


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_authenticated_non_admin_cannot_start_a_worker(
    client: TestClient, process: WorkerBoundary, monkeypatch: pytest.MonkeyPatch, role: LitellmUserRoles
) -> None:
    async def custom_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_role=role, api_key=api_key)

    monkeypatch.setattr(proxy_server, "user_custom_auth", custom_auth)
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == RUN_ERROR
    process.spawn.assert_not_awaited()


def test_missing_runtime_has_an_actionable_error(
    client: TestClient, process: WorkerBoundary, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LITEADMIN_PYTHON", str(tmp_path / "not-installed"))
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == {
            "type": "error",
            "message": "The LiteAgents runtime is not installed. Configure LITEADMIN_PYTHON on the gateway.",
        }
    process.spawn.assert_not_awaited()


@pytest.mark.parametrize("header_name", ("Authorization", "X-LiteLLM-Key"))
def test_admin_turn_passes_credentials_and_trusted_destinations_to_worker(
    client: TestClient, process: WorkerBoundary, monkeypatch: pytest.MonkeyPatch, header_name: str
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": header_name})
    monkeypatch.setenv("LITEADMIN_GATEWAY_URL", "https://backend.example/management/")
    monkeypatch.setenv("LITELLM_UI_API_DOC_BASE_URL", "http://testserver")
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == {"type": "done"}
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
    expected: Final = WorkerStart(
        access_token=ADMIN_KEY,
        chat=START.chat,
        headers={header_name: "Bearer " + ADMIN_KEY, "Authorization": "Bearer " + ADMIN_KEY},
        management_url="https://backend.example/management",
        inference_url="http://testserver",
    )
    process.write.assert_called_once_with((expected.model_dump_json() + "\n").encode())
    process.close.assert_called_once()


def test_invalid_management_url_never_reaches_the_worker(
    client: TestClient, process: WorkerBoundary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITEADMIN_GATEWAY_URL", "file:///etc/passwd")
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == REQUEST_ERROR
    process.spawn.assert_not_awaited()


def test_worker_exit_failure_reports_uncertainty_about_completed_actions(
    client: TestClient, process: WorkerBoundary
) -> None:
    process.readline.configure_mock(side_effect=(b"",))
    process.wait.configure_mock(return_value=1)
    process.spawn.configure_mock(**{"return_value.returncode": 1})
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == {
            "type": "error",
            "message": "LiteAdmin could not complete the request. Review completed actions before continuing.",
        }
    process.close.assert_called_once()


def test_disconnect_terminates_worker_and_closes_its_input(client: TestClient, process: WorkerBoundary) -> None:
    lines: Final = iter((b'{"type":"message","text":"Started"}\n',))
    cancelled: Final = asyncio.Event()

    async def wait_for_eof() -> bytes:
        if line := next(lines, None):
            return line
        try:
            await asyncio.Event().wait()
            return b""
        finally:
            cancelled.set()

    process.readline.configure_mock(side_effect=wait_for_eof)
    process.spawn.configure_mock(**{"return_value.returncode": None})
    with client.websocket_connect("/liteadmin/chat") as websocket:
        websocket.send_json({**START.model_dump(mode="json"), "access_token": ADMIN_KEY})
        assert websocket.receive_json() == {"type": "message", "text": "Started"}
    process.terminate.assert_called_once()
    process.close.assert_called_once()
    assert cancelled.is_set()


@pytest.mark.parametrize(
    ("configured", "candidate", "expected"),
    (
        ("", "http://untrusted.example", "http://127.0.0.1:4000"),
        ("https://inference.example/v1", "https://inference.example/v1/", "https://inference.example/v1"),
        ("/gateway", "https://dashboard.example/gateway", "http://127.0.0.1:4000/gateway"),
    ),
)
def test_inference_destination_comes_only_from_trusted_gateway_settings(
    monkeypatch: pytest.MonkeyPatch, configured: str, candidate: str, expected: str
) -> None:
    monkeypatch.setenv("LITELLM_UI_API_DOC_BASE_URL", configured)
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    assert inference_url(candidate, "http://127.0.0.1:4000") == expected


def test_a_changed_inference_destination_requires_reloading_for_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_UI_API_DOC_BASE_URL", "https://inference.example")
    with pytest.raises(ValueError, match="Reload LiteAdmin"):
        inference_url("https://different.example", "http://127.0.0.1:4000")
