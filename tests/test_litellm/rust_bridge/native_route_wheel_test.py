from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket as Socket
from typing import Final

REQUEST_STARTED: Final = threading.Event()
REQUEST_CANCELLED: Final = threading.Event()

ANTHROPIC_RESPONSE: Final = (
    b'{"id":"msg_native","type":"message","role":"assistant",'
    b'"model":"claude-sonnet-4-5","content":[{"type":"text","text":"native-message"}],'
    b'"stop_reason":"end_turn","stop_sequence":null,'
    b'"usage":{"input_tokens":2,"output_tokens":3}}'
)


def assert_ocr_request(path: str, headers: HTTPMessage, body: dict[str, object]) -> None:
    assert path == "/v1/ocr"
    assert headers.get("authorization") == "Bearer sk-native"
    assert body == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
        "include_image_base64": True,
    }


def assert_transcription_request(path: str, headers: HTTPMessage, body: dict[str, object]) -> None:
    assert path == "/model/mistral.voxtral-mini-3b-2507/converse"
    assert headers.get("authorization", "").startswith("AWS4-HMAC-SHA256 ")
    assert headers.get("x-amz-date")
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"][0]["audio"]["source"]["bytes"] == "AQI="
    assert "The audio language is en" in messages[0]["content"][1]["text"]


def assert_messages_request(path: str, headers: HTTPMessage, body: dict[str, object]) -> None:
    assert path == "/v1/messages"
    assert headers.get("x-api-key") == "sk-native"
    assert body == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello-from-messages"}],
    }


def assert_chat_completions_request(path: str, headers: HTTPMessage, body: dict[str, object]) -> None:
    assert path == "/v1/messages"
    assert headers.get("x-api-key") == "sk-native"
    assert body == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 17,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello-from-chat"}]}],
    }


@dataclass(frozen=True, slots=True)
class RouteCase:
    name: str
    request_kwargs: Mapping[str, object]
    assert_request: Callable[[str, HTTPMessage, dict[str, object]], None]
    success_value: Callable[[dict[str, object]], object]
    expected_success: str
    native_upstream_error: bool


ROUTE_CASES: Final = (
    RouteCase(
        name="ocr",
        request_kwargs={
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            "api_key": "sk-native",
            "custom_llm_provider": "mistral",
            "optional_params": {"include_image_base64": True},
        },
        assert_request=assert_ocr_request,
        success_value=lambda response: response["pages"][0]["markdown"],
        expected_success="native-ocr",
        native_upstream_error=True,
    ),
    RouteCase(
        name="transcription",
        request_kwargs={
            "model": "mistral.voxtral-mini-3b-2507",
            "audio": {"data": "AQI=", "format": "wav", "filename": "audio.wav"},
            "custom_llm_provider": "bedrock",
            "optional_params": {
                "aws_access_key_id": "native-access-key",
                "aws_secret_access_key": "native-secret-key",
                "aws_region_name": "us-east-1",
                "language": "en",
            },
        },
        assert_request=assert_transcription_request,
        success_value=lambda response: response["text"],
        expected_success="native-transcription",
        native_upstream_error=False,
    ),
    RouteCase(
        name="messages",
        request_kwargs={
            "model": "claude-sonnet-4-5",
            "body": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hello-from-messages"}],
            },
            "api_key": "sk-native",
            "custom_llm_provider": "anthropic",
        },
        assert_request=assert_messages_request,
        success_value=lambda response: response["content"][0]["text"],
        expected_success="native-message",
        native_upstream_error=False,
    ),
    RouteCase(
        name="chat_completions",
        request_kwargs={
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hello-from-chat"}],
            "optional_params": {"max_tokens": 17},
            "api_key": "sk-native",
        },
        assert_request=assert_chat_completions_request,
        success_value=lambda response: response["choices"][0]["message"]["content"],
        expected_success="native-message",
        native_upstream_error=True,
    ),
)
ROUTE_CASE_BY_NAME: Final = {case.name: case for case in ROUTE_CASES}


class NativeRouteHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        content_length: Final = int(self.headers.get("content-length", "0"))
        body: Final = json.loads(self.rfile.read(content_length))
        route: Final = self.headers.get("x-test-route")
        outcome: Final = self.headers.get("x-test-outcome")
        assert_native_request(route, outcome, self.path, self.headers, body)
        if outcome == "hang":
            REQUEST_STARTED.set()
            self.connection.settimeout(5)
            if connection_was_cancelled(self.connection):
                REQUEST_CANCELLED.set()
            return

        status: Final = 429 if outcome == "429" else 200
        response_body: Final = native_response(status, route)

        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response_body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, _message_format: str, *_args: object) -> None:
        pass


def connection_was_cancelled(connection: Socket) -> bool:
    try:
        return connection.recv(1) == b""
    except TimeoutError:
        return False
    except OSError:
        return True


def assert_native_request(
    route: str | None,
    outcome: str | None,
    path: str,
    headers: HTTPMessage,
    body: object,
) -> None:
    case: Final = ROUTE_CASE_BY_NAME.get(route or "")
    if case is None:
        raise AssertionError(f"unexpected route marker: {route!r}")
    if outcome not in {"success", "429", "hang"}:
        raise AssertionError(f"unexpected outcome marker: {outcome!r}")
    if not isinstance(body, dict):
        raise TypeError(f"{route} sent {type(body).__name__}, expected a JSON object")
    case.assert_request(path, headers, body)


def native_response(status: int, route: str | None) -> bytes:
    if status == 429:
        return b'{"error":"native-rate-limit"}'
    if route == "ocr":
        return b'{"pages":[{"index":0,"markdown":"native-ocr"}]}'
    if route == "transcription":
        return b'{"output":{"message":{"content":[{"text":"native-transcription"}]}}}'
    return ANTHROPIC_RESPONSE


def load_native(native_path: Path) -> object:
    module_spec: Final = importlib.util.spec_from_file_location("litellm.rust_bridge._native", native_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot create native extension import specification")
    native_module: Final = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native_module)
    return native_module


def route_kwargs(route: str, api_base: str, outcome: str) -> dict[str, object]:
    case: Final = ROUTE_CASE_BY_NAME.get(route)
    if case is None:
        raise AssertionError(f"unknown route: {route}")
    common: Final = {
        "api_base": api_base,
        "extra_headers": {"x-test-outcome": outcome, "x-test-route": route},
        "timeout_seconds": 3.0,
    }
    return common | dict(case.request_kwargs)


def assert_success(route: str, response: object) -> None:
    case: Final = ROUTE_CASE_BY_NAME[route]
    if not isinstance(response, dict):
        raise TypeError(f"{route} returned {type(response).__name__}, expected dict")
    actual: Final = case.success_value(response)
    if actual != case.expected_success:
        raise AssertionError(f"{route} returned {actual!r}, expected {case.expected_success!r}")


def assert_rate_limit(native: object, route: str, error: BaseException) -> None:
    case: Final = ROUTE_CASE_BY_NAME[route]
    if case.native_upstream_error:
        upstream_error: Final = native.RustUpstreamError
        if not isinstance(error, upstream_error) or error.args[0] != 429:
            raise AssertionError(f"{route} returned the wrong 429 error: {error!r}")
        return
    if not isinstance(error, RuntimeError) or "429" not in str(error):
        raise AssertionError(f"{route} returned the wrong 429 error: {error!r}")


def exercise_sync(native: object, api_base: str) -> None:
    for case in ROUTE_CASES:
        function: Final = getattr(native, case.name)
        assert_success(case.name, function(**route_kwargs(case.name, api_base, "success")))
        try:
            function(**route_kwargs(case.name, api_base, "429"))
        except (RuntimeError, native.RustUpstreamError) as error:
            assert_rate_limit(native, case.name, error)
        else:
            raise AssertionError(f"{case.name} accepted a 429 response")


async def exercise_async(native: object, api_base: str) -> None:
    for case in ROUTE_CASES:
        function: Final = getattr(native, f"a{case.name}")
        assert_success(case.name, await function(**route_kwargs(case.name, api_base, "success")))
        try:
            await function(**route_kwargs(case.name, api_base, "429"))
        except (RuntimeError, native.RustUpstreamError) as error:
            assert_rate_limit(native, case.name, error)
        else:
            raise AssertionError(f"a{case.name} accepted a 429 response")


async def exercise_async_concurrency(native: object, api_base: str) -> None:
    responses: Final = await asyncio.wait_for(
        asyncio.gather(*(native.amessages(**route_kwargs("messages", api_base, "success")) for _ in range(32))),
        timeout=15,
    )
    for response in responses:
        assert_success("messages", response)


def exercise_routes(native_path: Path, api_base: str) -> object:
    native: Final = load_native(native_path)
    if hasattr(native, "_trace"):
        raise AssertionError("release wheel exposed trace-parity diagnostics")
    exercise_sync(native, api_base)
    asyncio.run(exercise_async(native, api_base))
    asyncio.run(exercise_async_concurrency(native, api_base))
    return native


def exercise_signal(native: object, api_base: str) -> int:
    try:
        native.messages(
            **route_kwargs("messages", api_base, "hang"),
        )
    except KeyboardInterrupt:
        sys.stdout.write("KeyboardInterrupt\n")
        sys.stdout.flush()
        sys.stdin.read(1)
        return 0
    raise AssertionError("sync native route ignored SIGINT")


def verify_sigint(native_path: Path, api_base: str) -> None:
    REQUEST_STARTED.clear()
    REQUEST_CANCELLED.clear()
    process: Final = subprocess.Popen(
        (sys.executable, __file__, "child", str(native_path), api_base),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not REQUEST_STARTED.wait(30):
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            raise AssertionError(
                f"native route matrix did not reach the hanging upstream\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        os.kill(process.pid, signal.SIGINT)
        if not REQUEST_CANCELLED.wait(5):
            raise AssertionError("interrupted native route did not cancel its upstream future")
        if process.poll() is not None:
            raise AssertionError("signal child exited before cancellation was observed")
        stdout, stderr = process.communicate(input="\n", timeout=5)
        if process.returncode != 0 or stdout != "KeyboardInterrupt\n":
            raise AssertionError(
                f"signal child exited with status {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def verify_wheel(wheel: Path) -> int:
    with tempfile.TemporaryDirectory() as temporary_directory, zipfile.ZipFile(wheel) as archive:
        wheel_root: Final = Path(temporary_directory)
        for member in archive.infolist():
            target: Final = wheel_root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        native_members: Final = tuple(
            member
            for member in archive.infolist()
            if member.filename.startswith("litellm/rust_bridge/_native.") and member.filename.endswith(".so")
        )
        if len(native_members) != 1:
            raise AssertionError(f"expected one native extension, found {len(native_members)}")
        native_path: Final = wheel_root / native_members[0].filename

        server: Final = ThreadingHTTPServer(("127.0.0.1", 0), NativeRouteHandler)
        server_thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        api_base: Final = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            verify_sigint(native_path, api_base)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
    return 0


def main() -> int:
    if len(sys.argv) == 2:
        return verify_wheel(Path(sys.argv[1]))
    if len(sys.argv) == 4 and sys.argv[1] == "child":
        native: Final = exercise_routes(Path(sys.argv[2]), sys.argv[3])
        return exercise_signal(native, sys.argv[3])
    sys.stderr.write(f"usage: {Path(sys.argv[0]).name} WHEEL\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
