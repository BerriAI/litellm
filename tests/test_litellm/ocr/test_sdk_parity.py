from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from enum import Enum
from pathlib import Path
from typing import Final, Protocol, cast

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

from tests.test_litellm.ocr.fixture_models import OcrFixture, OcrFixtureResponse, ProviderWireRequest
from tests.test_litellm.parity.compare import assert_parity
from tests.test_litellm.parity.models import (
    CapturedRequest,
    ExceptionReport,
    NativeEvidence,
    ParityTrace,
    ReplayResponse,
    SDKOutput,
    SDKReport,
)
from tests.test_litellm.parity.replay import replay_json_response
from tests.test_litellm.parity.runner import PythonScriptRunner, run_execution

API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-ocr-parity-fallback"
GIL_STATS: Final = TypeAdapter(dict[str, int])


class SDKRoute(str, Enum):
    OCR = "ocr"
    AOCR = "aocr"


class SDKInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: SDKRoute
    kwargs: dict[str, object]


class _NativeBridge(Protocol):
    ocr: object
    aocr: object

    def gil_stats(self) -> object: ...


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


def _response_trace(response: BaseModel) -> ParityTrace:
    output: Final = SDKOutput(
        response_type=_qualified_name(response),
        response_json=response.model_dump(mode="json"),
    )
    return ParityTrace(outputs=(output,), exception=None)


def _capture_sdk_call(sdk_input: SDKInput, mock_url: str) -> ParityTrace:
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
            return _response_trace(sync_route(**call_kwargs))
        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        return _response_trace(asyncio.run(async_route(**call_kwargs)))
    except Exception as error:
        return ParityTrace(outputs=(), exception=_exception_report(error))


def _native_sdk_report(trace: ParityTrace, native_handled_case: bool) -> SDKReport:
    return SDKReport(
        trace=trace,
        native=NativeEvidence(
            rust_enabled=True,
            native_callable_loaded=True,
            native_handled_case=native_handled_case,
        ),
    )


def _execute_sdk_case(sdk_input: SDKInput, mock_url: str) -> SDKReport:
    from litellm.rust_bridge.loader import get_native_bridge
    from litellm.rust_bridge.ocr import load_rust_aocr, load_rust_ocr, rust_ocr_enabled

    rust_enabled: Final = rust_ocr_enabled()
    if not rust_enabled:
        return SDKReport(
            trace=_capture_sdk_call(sdk_input, mock_url),
            native=NativeEvidence(
                rust_enabled=False,
                native_callable_loaded=False,
                native_handled_case=False,
            ),
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
        async_trace: Final = _capture_sdk_call(sdk_input, mock_url)
        return _native_sdk_report(async_trace, native_handled_case=bool(async_trace.outputs))

    before_gil_releases: Final = _gil_release_count(native_bridge)
    trace: Final = _capture_sdk_call(sdk_input, mock_url)
    after_gil_releases: Final = _gil_release_count(native_bridge)
    return _native_sdk_report(trace, native_handled_case=after_gil_releases == before_gil_releases + 1)


def _replay_response(response: OcrFixtureResponse) -> ReplayResponse:
    return ReplayResponse(status_code=response.status_code, headers=response.headers, body=response.body)


def _provider_wire_request(request: CapturedRequest) -> ProviderWireRequest:
    return ProviderWireRequest(method=request.method, path=request.path, body=request.body)


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_recorded_ocr_sdk_parity(ocr_fixture: OcrFixture, route: SDKRoute, tmp_path: Path) -> None:
    sdk_input: Final = SDKInput(route=route, kwargs=ocr_fixture.request.sdk_kwargs)
    case_file: Final = tmp_path / f"{ocr_fixture.request.provider}-{route.value}-sdk-input.json"
    case_file.write_text(sdk_input.model_dump_json(indent=2), encoding="utf-8")
    expected_request: Final = ocr_fixture.request.provider_request
    runner: Final = PythonScriptRunner(
        entrypoint=Path(__file__),
        rust_env_var="LITELLM_USE_RUST_OCR",
        python_user_agent=PYTHON_HTTP_SENTINEL,
    )

    with (
        replay_json_response(expected_request.path, _replay_response(ocr_fixture.response)) as python_provider,
        replay_json_response(expected_request.path, _replay_response(ocr_fixture.response)) as rust_provider,
    ):
        python: Final = run_execution(
            runner,
            case_file,
            tmp_path / f"{route.value}-python-report.json",
            python_provider,
            rust_enabled=False,
        )
        rust: Final = run_execution(
            runner,
            case_file,
            tmp_path / f"{route.value}-rust-report.json",
            rust_provider,
            rust_enabled=True,
        )

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    assert tuple(_provider_wire_request(request) for request in python.requests) == (expected_request,)
    assert tuple(_provider_wire_request(request) for request in rust.requests) == (expected_request,)


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
