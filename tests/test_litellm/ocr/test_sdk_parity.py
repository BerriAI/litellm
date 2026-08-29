from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, Protocol, cast

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

MODEL: Final = "mistral/mistral-ocr-latest"
API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-ocr-parity-fallback"
JSON_OBJECT: Final = TypeAdapter(dict[str, object])
GIL_STATS: Final = TypeAdapter(dict[str, int])

STATIC_OCR_RESPONSE: Final[dict[str, object]] = {
    "pages": [
        {
            "index": 0,
            "markdown": "# Static OCR\n\nProvider fixture text.",
            "dimensions": {"dpi": 200, "height": 1000, "width": 800},
            "blocks": [{"type": "title", "content": "Static OCR"}],
            "header": "Fixture header",
            "footer": "Fixture footer",
        }
    ],
    "model": "mistral-ocr-static",
    "document_annotation": {"language": "en"},
    "usage_info": {"pages_processed": 1, "credits": 0.25, "provider_extra": "preserved"},
    "top_level_extra": "dropped by both implementations",
}


class SDKRoute(str, Enum):
    OCR = "ocr"
    AOCR = "aocr"


class SDKInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: SDKRoute
    kwargs: dict[str, object]


class ExceptionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_name: str
    status_code: int | None
    message: str


class SDKReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    rust_enabled: bool
    native_callable_loaded: bool
    native_handled_case: bool
    response_type: str | None
    response_json: dict[str, object] | None
    exception_json: ExceptionReport | None


class ProviderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    authorization: str | None
    content_type: str | None
    parity_case: str | None
    body: dict[str, object]
    user_agent: str | None


@dataclass(frozen=True, slots=True)
class Execution:
    report: SDKReport
    request: ProviderRequest


@dataclass(frozen=True, slots=True)
class CallOutcome:
    response_type: str | None
    response_json: dict[str, object] | None
    exception_json: ExceptionReport | None


class _NativeBridge(Protocol):
    ocr: object
    aocr: object

    def gil_stats(self) -> object: ...


class _StaticOcrProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _StaticOcrHandler)
        self.requests: queue.Queue[ProviderRequest] = queue.Queue()
        self.response_body: bytes = json.dumps(STATIC_OCR_RESPONSE, sort_keys=True, separators=(",", ":")).encode()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_single_request(self) -> ProviderRequest:
        try:
            first: Final = self.requests.get(timeout=5)
        except queue.Empty as error:
            raise AssertionError("expected one provider request, received none") from error
        try:
            extra: Final = self.requests.get_nowait()
        except queue.Empty:
            return first
        raise AssertionError(f"expected one provider request, received an extra request: {extra.model_dump_json()}")


class _StaticOcrHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, _StaticOcrProvider)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(length))
        content_type_header: Final = self.headers.get("content-type")
        content_type: Final = content_type_header.split(";", 1)[0].lower() if content_type_header else None
        provider.requests.put(
            ProviderRequest(
                method=self.command,
                path=self.path,
                authorization=self.headers.get("authorization"),
                content_type=content_type,
                parity_case=self.headers.get("x-parity-case"),
                body=body,
                user_agent=self.headers.get("user-agent"),
            )
        )
        status: Final = 200 if self.path == "/v1/ocr" else 404
        response_body: Final = provider.response_body if status == 200 else b'{"error":"unexpected path"}'
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _static_provider() -> Generator[_StaticOcrProvider]:
    server: Final = _StaticOcrProvider()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _qualified_name(value: object) -> str:
    value_type: Final = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _exception_report(error: Exception) -> ExceptionReport:
    raw_status_code: Final = getattr(error, "status_code", None)
    status_code: Final = raw_status_code if isinstance(raw_status_code, int) else None
    return ExceptionReport(class_name=_qualified_name(error), status_code=status_code, message=str(error))


def _gil_release_count(native_bridge: _NativeBridge) -> int:
    stats: Final = GIL_STATS.validate_python(native_bridge.gil_stats())
    return stats["releases"]


def _response_outcome(response: BaseModel) -> CallOutcome:
    response_json: Final = JSON_OBJECT.validate_python(response.model_dump(mode="json"))
    return CallOutcome(
        response_type=_qualified_name(response),
        response_json=response_json,
        exception_json=None,
    )


def _capture_sdk_call(
    sdk_input: SDKInput,
    mock_url: str,
) -> CallOutcome:
    import litellm
    from litellm.llms.base_llm.ocr.transformation import OCRResponse

    call_kwargs: Final[dict[str, object]] = {
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-parity-case": sdk_input.route.value},
        **sdk_input.kwargs,
    }

    try:
        if sdk_input.route is SDKRoute.OCR:
            sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
            return _response_outcome(sync_route(**call_kwargs))

        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        return _response_outcome(asyncio.run(async_route(**call_kwargs)))
    except Exception as error:
        return CallOutcome(
            response_type=None,
            response_json=None,
            exception_json=_exception_report(error),
        )


def _native_sdk_report(outcome: CallOutcome, native_handled_case: bool) -> SDKReport:
    return SDKReport(
        rust_enabled=True,
        native_callable_loaded=True,
        native_handled_case=native_handled_case,
        response_type=outcome.response_type,
        response_json=outcome.response_json,
        exception_json=outcome.exception_json,
    )


def _execute_sdk_case(sdk_input: SDKInput, mock_url: str) -> SDKReport:
    from litellm.rust_bridge.loader import get_native_bridge
    from litellm.rust_bridge.ocr import load_rust_aocr, load_rust_ocr, rust_ocr_enabled

    rust_enabled: Final = rust_ocr_enabled()
    if not rust_enabled:
        outcome: Final = _capture_sdk_call(sdk_input, mock_url)
        return SDKReport(
            rust_enabled=False,
            native_callable_loaded=False,
            native_handled_case=False,
            response_type=outcome.response_type,
            response_json=outcome.response_json,
            exception_json=outcome.exception_json,
        )

    raw_native_bridge: Final = get_native_bridge()
    if raw_native_bridge is None:
        raise RuntimeError("LITELLM_USE_RUST_OCR=1 but the native bridge is unavailable")
    native_bridge: Final = cast(_NativeBridge, raw_native_bridge)

    native_callable: Final = load_rust_ocr() if sdk_input.route is SDKRoute.OCR else load_rust_aocr()
    expected_native_callable: Final = native_bridge.ocr if sdk_input.route is SDKRoute.OCR else native_bridge.aocr
    if native_callable is None or native_callable is not expected_native_callable:
        raise RuntimeError(f"native {sdk_input.route.value} callable is unavailable")

    if sdk_input.route is SDKRoute.AOCR:
        async_outcome: Final = _capture_sdk_call(sdk_input, mock_url)
        return _native_sdk_report(async_outcome, native_handled_case=True)

    before_gil_releases: Final = _gil_release_count(native_bridge)
    sync_outcome: Final = _capture_sdk_call(sdk_input, mock_url)
    after_gil_releases: Final = _gil_release_count(native_bridge)
    native_handled_case: Final = after_gil_releases == before_gil_releases + 1
    return _native_sdk_report(sync_outcome, native_handled_case)


def _run_execution(
    case_file: Path,
    report_file: Path,
    provider: _StaticOcrProvider,
    rust_enabled: bool,
) -> Execution:
    env: Final = {
        **os.environ,
        "LITELLM_USE_RUST_OCR": "1" if rust_enabled else "0",
        "LITELLM_USER_AGENT": PYTHON_HTTP_SENTINEL,
    }
    completed: Final = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(case_file),
            provider.url,
            str(report_file),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"SDK subprocess failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    report: Final = SDKReport.model_validate_json(report_file.read_text(encoding="utf-8"))
    return Execution(report=report, request=provider.take_single_request())


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_static_mistral_ocr_sdk_parity(route: SDKRoute, tmp_path: Path) -> None:
    sdk_input: Final = SDKInput(
        route=route,
        kwargs={
            "model": MODEL,
            "document": {"type": "document_url", "document_url": "https://example.com/static.pdf"},
            "pages": [0, 1],
            "include_image_base64": True,
            "include_blocks": True,
            "table_format": "html",
            "id": "static-ocr-parity",
        },
    )
    case_file: Final = tmp_path / f"{route.value}-sdk-input.json"
    case_file.write_text(sdk_input.model_dump_json(indent=2), encoding="utf-8")
    with _static_provider() as python_mock, _static_provider() as rust_mock:
        python: Final = _run_execution(
            case_file,
            tmp_path / f"{route.value}-python-report.json",
            python_mock,
            False,
        )
        rust: Final = _run_execution(
            case_file,
            tmp_path / f"{route.value}-rust-report.json",
            rust_mock,
            True,
        )

    assert python.report.rust_enabled is False
    assert rust.report.rust_enabled is True
    assert rust.report.native_callable_loaded is True
    assert rust.report.native_handled_case is True
    assert python.request.user_agent == PYTHON_HTTP_SENTINEL
    assert rust.request.user_agent != PYTHON_HTTP_SENTINEL
    assert rust.request.model_dump(exclude={"user_agent"}) == python.request.model_dump(exclude={"user_agent"})
    assert rust.report.response_type == python.report.response_type
    assert rust.report.response_json == python.report.response_json
    assert rust.report.exception_json == python.report.exception_json
    assert python.report.exception_json is None
    assert python.report.response_json is not None


def _child_main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_sdk_parity.py CASE_FILE MOCK_URL REPORT_FILE")
    case_file: Final = Path(sys.argv[1])
    mock_url: Final = sys.argv[2]
    report_file: Final = Path(sys.argv[3])
    sdk_input: Final = SDKInput.model_validate_json(case_file.read_text(encoding="utf-8"))
    report: Final = _execute_sdk_case(sdk_input, mock_url)
    report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    _child_main()
