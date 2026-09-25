import asyncio
import os
from collections.abc import Callable, Coroutine, Mapping
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError
from starlette.types import Message, Scope

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

from .models import ChatRequest, Decision

router: Final = APIRouter()


class AuthSettings(BaseModel):
    litellm_key_header_name: str | None = None


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    access_token: SecretStr
    chat: ChatRequest


class WorkerStart(BaseModel):
    model_config = ConfigDict(frozen=True)
    access_token: str = Field(repr=False)
    chat: ChatRequest
    headers: Mapping[str, str]
    management_url: str
    inference_url: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


async def send_error(websocket: WebSocket, message: str) -> None:
    await websocket.send_text(ErrorEvent(message=message).model_dump_json())


async def _receive(websocket: WebSocket, maximum: int) -> str:
    text: Final = await websocket.receive_text()
    if len(text) > maximum:
        raise ValueError("Message is too large")
    return text


def inference_url(candidate: str, management: str) -> str:
    values: Final = (os.getenv("LITELLM_UI_API_DOC_BASE_URL", "").strip(), os.getenv("PROXY_BASE_URL", "").strip())
    configured: Final = next((value for value in values if value), "")
    if not configured:
        return management
    target: Final = httpx.URL(urljoin(management + "/", configured))
    expected: Final = httpx.URL(candidate)
    if target.scheme not in ("http", "https") or target.username or target.password or target.query or target.fragment:
        raise ValueError("Invalid inference gateway configuration")
    matches: Final = (
        str(target).rstrip("/") == str(expected).rstrip("/")
        if httpx.URL(configured).is_absolute_url
        else target.path.rstrip("/") == expected.path.rstrip("/")
    )
    if not matches:
        raise ValueError("Gateway settings changed. Reload LiteAdmin before continuing.")
    return str(target)


async def _authenticate(websocket: WebSocket, token: str) -> Mapping[str, str]:
    from litellm.proxy.proxy_server import (
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # Validated at the legacy settings boundary below
    )

    header_name: Final = AuthSettings.model_validate(general_settings).litellm_key_header_name or "Authorization"
    headers: Final = MappingProxyType({header_name: "Bearer " + token, "Authorization": "Bearer " + token})

    async def body() -> Message:
        message: Final[Message] = {"type": "http.request", "body": b"{}", "more_body": False}
        return message

    scope: Final[Scope] = {
        **websocket.scope,
        "type": "http",
        "method": "POST",
        "scheme": "https" if websocket.url.scheme == "wss" else "http",
        "headers": tuple((key.lower().encode(), value.encode()) for key, value in headers.items()),
    }
    request: Final = Request(scope, receive=body)
    auth: Final = await user_api_key_auth(request=request, api_key="Bearer " + token)
    if auth.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise PermissionError("LiteAdmin requires gateway administrator access.")
    return headers


async def _coordinate(
    work: Callable[[], Coroutine[object, object, None]], receive: Callable[[], Coroutine[object, object, None]]
) -> None:
    tasks: Final = (asyncio.create_task(work()), asyncio.create_task(receive()))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run(websocket: WebSocket, start: StartRequest, headers: Mapping[str, str]) -> None:
    root: Final = Path(__file__).resolve().parents[3]
    directory: Final = Path(os.getenv("LITEADMIN_RUNTIME", str(root / "liteadmin")))
    python: Final = Path(os.getenv("LITEADMIN_PYTHON", str(root / ".liteadmin-venv" / "bin" / "python")))
    if not python.is_file():
        await send_error(
            websocket, "The LiteAgents runtime is not installed. Configure LITEADMIN_PYTHON on the gateway."
        )
        return
    configured_management: Final = os.getenv("LITEADMIN_GATEWAY_URL")
    management: Final = (
        configured_management.rstrip("/")
        if configured_management
        else "http://127.0.0.1:" + str(TypeAdapter(tuple[str, int]).validate_python(websocket.scope.get("server"))[1])
    )
    management_target: Final = httpx.URL(management)
    if management_target.scheme not in ("http", "https") or not management_target.host:
        raise ValueError("Invalid management gateway configuration")
    payload: Final = WorkerStart(
        access_token=start.access_token.get_secret_value(),
        chat=start.chat,
        headers=headers,
        management_url=management,
        inference_url=inference_url(start.chat.inference_base_url, management),
    )
    inherited: Final = MappingProxyType(
        {name: value for name, value in os.environ.items() if name in ("PATH", "LANG", "SSL_CERT_FILE", "SSL_CERT_DIR")}
    )
    environment: Final = MappingProxyType(
        {**inherited, "PYTHONPATH": str(directory.parent), "PYDANTIC_AI_NO_BANNER": "1"}
    )
    process: Final = await asyncio.create_subprocess_exec(
        str(python),
        "-m",
        "liteadmin.worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=dict(environment),  # mutable-ok: uvloop requires a concrete dict for subprocess environments
        cwd=directory.parent,
        limit=256_000,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("Could not open the LiteAgents runtime")
    writer: Final = process.stdin
    reader: Final = process.stdout

    async def forward() -> None:
        while line := await reader.readline():
            await websocket.send_text(line.decode())
        if await process.wait() != 0:
            await send_error(
                websocket, "LiteAdmin could not complete the request. Review completed actions before continuing."
            )

    async def receive() -> None:
        while True:
            writer.write(
                (Decision.model_validate_json(await _receive(websocket, 1024)).model_dump_json() + "\n").encode()
            )
            await writer.drain()

    try:
        writer.write((payload.model_dump_json() + "\n").encode())
        await writer.drain()
        await _coordinate(forward, receive)
    finally:
        writer.close()
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            process.kill()
            await process.wait()


@router.websocket("/liteadmin/chat")
async def liteadmin_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        start: Final = StartRequest.model_validate_json(
            await asyncio.wait_for(_receive(websocket, 180_000), timeout=10)
        )
        headers: Final = await _authenticate(websocket, start.access_token.get_secret_value())
        await asyncio.wait_for(_run(websocket, start, headers), timeout=600)
    except WebSocketDisconnect:
        return
    except (ValidationError, ValueError):
        await send_error(websocket, "Invalid LiteAdmin request or gateway settings. Reload the page and try again.")
    except Exception as error:
        verbose_proxy_logger.warning("LiteAdmin request failed: %s", type(error).__name__)
        with suppress(RuntimeError, WebSocketDisconnect):
            await send_error(
                websocket,
                "LiteAdmin could not complete this request. Check your access and model, and review completed actions before continuing.",
            )
    finally:
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()
