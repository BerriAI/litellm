from __future__ import annotations

import asyncio
import os
import sys
import traceback
from collections.abc import Awaitable, Callable, Coroutine, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, cast

import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import get_native_bridge
from litellm.rust_bridge import ocr as rust_ocr_bridge
from litellm.rust_bridge.ocr import RustAocr, RustOcr
from tests.route_parity.compare import assert_model_parity, assert_parity, assert_request_parity
from tests.route_parity.fixtures.store import recorded_fixtures
from tests.route_parity.inprocess import run_in_process
from tests.route_parity.models import (
    SDKCommand,
    SDKError,
    SDKReport,
    SDKSuccess,
    WorkerFailure,
    WorkerResult,
    WorkerSuccess,
    sdk_error_report,
)
from tests.route_parity.replay import replay_server
from tests.route_parity.runner import (
    PythonScriptRunner,
    PythonScriptWorker,
    execution_worker_pair,
    parity_worker_main,
    run_execution,
)
from tests.test_litellm.ocr.fixtures.models import OcrParityCase, OcrSdkInput

API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-ocr-parity-fallback"
FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"


class SDKRoute(str, Enum):
    OCR = "ocr"
    AOCR = "aocr"


@dataclass(frozen=True, slots=True)
class InvalidOcrCase:
    name: str
    model: str
    document: object
    expected_exception_type: str
    expected_status_code: int
    expected_message: str
    extra_kwargs: tuple[tuple[str, object], ...] = ()
    expected_rust_calls: int = 0


INVALID_OCR_CASES: Final = (
    InvalidOcrCase(
        name="unsupported_provider",
        model="openai/gpt-4o",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="OCR is not supported for provider: openai",
    ),
    InvalidOcrCase(
        name="unsupported_reducto_model",
        model="reducto/parse-v4",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="OCR is not supported for provider: reducto",
    ),
    InvalidOcrCase(
        name="unknown_provider_prefix",
        model="not_a_provider/model",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.BadRequestError",
        expected_status_code=400,
        expected_message="LLM Provider NOT provided",
    ),
    InvalidOcrCase(
        name="non_object_document",
        model="mistral/mistral-ocr-latest",
        document=[],
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="document must be a dict",
    ),
    InvalidOcrCase(
        name="missing_document_type",
        model="mistral/mistral-ocr-latest",
        document={},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="Invalid document type: None",
    ),
    InvalidOcrCase(
        name="unsupported_document_type",
        model="mistral/mistral-ocr-latest",
        document={"type": "text", "text": "not a document"},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="Invalid document type: text",
    ),
    InvalidOcrCase(
        name="missing_document_url",
        model="azure_ai/doc-intelligence/prebuilt-read",
        document={"type": "document_url"},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="Document URL is required",
        expected_rust_calls=1,
    ),
    InvalidOcrCase(
        name="invalid_request_format",
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.UnsupportedParamsError",
        expected_status_code=400,
        expected_message="Invalid `req_format`: 'bogus'",
        extra_kwargs=(("req_format", "bogus"),),
    ),
    InvalidOcrCase(
        name="invalid_document_intelligence_pages",
        model="azure_ai/doc-intelligence/prebuilt-read",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="`pages` integers must be >= 0",
        extra_kwargs=(("pages", [-1]),),
    ),
    InvalidOcrCase(
        name="invalid_document_intelligence_features",
        model="azure_ai/doc-intelligence/prebuilt-read",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="Invalid `features` for Azure Document Intelligence",
        extra_kwargs=(("features", [1]),),
    ),
    InvalidOcrCase(
        name="invalid_header_value",
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,AA=="},
        expected_exception_type="litellm.exceptions.InternalServerError",
        expected_status_code=500,
        expected_message="Header value must be str or bytes",
        extra_kwargs=(("extra_headers", {"x-invalid": 1}),),
    ),
)


def _call_kwargs(sdk_input: OcrSdkInput, mock_url: str, route: SDKRoute) -> dict[str, object]:
    return {
        **sdk_input.as_sdk_kwargs(),
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-litellm-parity-route": route.value},
    }


def _execute_sdk_call(
    call_kwargs: dict[str, object],
    route: SDKRoute,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    import litellm

    try:
        if route is SDKRoute.OCR:
            sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
            response: Final = sync_route(**call_kwargs)
            return SDKSuccess(response=response.model_dump(mode="json"))
        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        async_response: Final = event_loop.run_until_complete(async_route(**call_kwargs))
        return SDKSuccess(response=async_response.model_dump(mode="json"))
    except Exception as error:
        return sdk_error_report(error)


def _execute_sdk_case(
    sdk_input: OcrSdkInput,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    call_kwargs: Final = _call_kwargs(sdk_input, mock_url, route)
    return _execute_sdk_call(call_kwargs, route, event_loop)


def _execute_recorded_sdk_case(
    sdk_input: OcrSdkInput,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> OCRResponse | SDKError:
    import litellm

    call_kwargs: Final = _call_kwargs(sdk_input, mock_url, route)
    try:
        if route is SDKRoute.OCR:
            sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
            return sync_route(**call_kwargs)
        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        return event_loop.run_until_complete(async_route(**call_kwargs))
    except Exception as error:
        return sdk_error_report(error)


def _execute_invalid_sdk_case(
    case: InvalidOcrCase,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    call_kwargs: Final = {
        "model": case.model,
        "document": case.document,
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-litellm-parity-route": route.value},
        **dict(case.extra_kwargs),
    }
    return _execute_sdk_call(call_kwargs, route, event_loop)


class _RustOcrSpy:
    def __init__(self, delegate: RustOcr) -> None:
        self.delegate: Final = delegate
        self.calls = 0

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls += 1
        return self.delegate(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        )


class _RustAocrSpy:
    def __init__(self, delegate: RustAocr) -> None:
        self.delegate: Final = delegate
        self.calls = 0

    async def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls += 1
        result: Final[Awaitable[dict[str, object]]] = self.delegate(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout_seconds=timeout_seconds,
        )
        return await result


@contextmanager
def _restore_rust_ocr_state() -> Generator[None]:
    enabled: Final = rust_ocr_bridge._rust_ocr_enabled  # pyright: ignore[reportPrivateUsage]  # restore test state
    ocr_impl: Final = rust_ocr_bridge._rust_ocr_impl  # pyright: ignore[reportPrivateUsage]  # restore test state
    aocr_impl: Final = rust_ocr_bridge._rust_aocr_impl  # pyright: ignore[reportPrivateUsage]  # restore test state
    try:
        yield
    finally:
        rust_ocr_bridge.use_litellm_rust(enabled, ocr=ocr_impl, aocr=aocr_impl)


def _native_spies() -> tuple[_RustOcrSpy, _RustAocrSpy]:
    native_bridge: Final = get_native_bridge()
    if native_bridge is None:
        pytest.fail("native Rust bridge is required for OCR parity testing")
    sync_spy: Final = _RustOcrSpy(cast(RustOcr, getattr(native_bridge, "ocr")))
    async_spy: Final = _RustAocrSpy(cast(RustAocr, getattr(native_bridge, "aocr")))
    return sync_spy, async_spy


@pytest.fixture(scope="module")
def sdk_workers() -> Generator[tuple[PythonScriptWorker, PythonScriptWorker]]:
    runner: Final = PythonScriptRunner(
        entrypoint=Path(__file__),
        rust_env_var="LITELLM_USE_RUST_OCR",
        python_user_agent=PYTHON_HTTP_SENTINEL,
        route_label="OCR",
    )
    with execution_worker_pair(runner) as workers:
        yield workers


@pytest.fixture(scope="module")
def startup_ocr_fixture() -> OcrParityCase:
    default_directory: Final = Path(__file__).with_name("fixtures") / "data"
    configured: Final = os.environ.get(FIXTURE_DIR_ENV)
    directory: Final = Path(configured).expanduser() if configured is not None else default_directory
    fixtures: Final = recorded_fixtures(directory, OcrParityCase)
    if not fixtures:
        pytest.skip(f"no recorded fixtures in {directory}")
    return fixtures[0]


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_recorded_ocr_sdk_parity(
    ocr_fixture: OcrParityCase,
    route: SDKRoute,
) -> None:
    sync_spy, async_spy = _native_spies()
    event_loop: Final = asyncio.new_event_loop()
    try:
        with _restore_rust_ocr_state(), replay_server() as provider:
            rust_ocr_bridge.use_litellm_rust(False, ocr=sync_spy, aocr=async_spy)
            python: Final = run_in_process(
                provider,
                ocr_fixture.provider_responses,
                lambda mock_url: _execute_recorded_sdk_case(ocr_fixture.litellm_input, route, mock_url, event_loop),
            )
            assert sync_spy.calls == 0
            assert async_spy.calls == 0

            rust_ocr_bridge.use_litellm_rust(True, ocr=sync_spy, aocr=async_spy)
            rust: Final = run_in_process(
                provider,
                ocr_fixture.provider_responses,
                lambda mock_url: _execute_recorded_sdk_case(ocr_fixture.litellm_input, route, mock_url, event_loop),
            )
    finally:
        event_loop.close()

    assert sync_spy.calls == (1 if route is SDKRoute.OCR else 0)
    assert async_spy.calls == (1 if route is SDKRoute.AOCR else 0)
    assert_request_parity(python.requests, rust.requests)
    if any(response.status_code >= 400 for response in ocr_fixture.provider_responses):
        assert isinstance(python.response, SDKError)
    if isinstance(python.response, SDKError):
        assert python.response == rust.response
    else:
        assert isinstance(rust.response, OCRResponse)
        assert_model_parity(python.response, rust.response)


@pytest.mark.parametrize("case", INVALID_OCR_CASES, ids=tuple(case.name for case in INVALID_OCR_CASES))
@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_invalid_ocr_sdk_parity(case: InvalidOcrCase, route: SDKRoute) -> None:
    sync_spy, async_spy = _native_spies()
    event_loop: Final = asyncio.new_event_loop()
    try:
        with _restore_rust_ocr_state(), replay_server() as provider:
            rust_ocr_bridge.use_litellm_rust(False, ocr=sync_spy, aocr=async_spy)
            python: Final = run_in_process(
                provider,
                (),
                lambda mock_url: _execute_invalid_sdk_case(case, route, mock_url, event_loop),
            )
            assert sync_spy.calls == 0
            assert async_spy.calls == 0

            rust_ocr_bridge.use_litellm_rust(True, ocr=sync_spy, aocr=async_spy)
            rust: Final = run_in_process(
                provider,
                (),
                lambda mock_url: _execute_invalid_sdk_case(case, route, mock_url, event_loop),
            )
    finally:
        event_loop.close()

    assert sync_spy.calls == (case.expected_rust_calls if route is SDKRoute.OCR else 0)
    assert async_spy.calls == (case.expected_rust_calls if route is SDKRoute.AOCR else 0)
    assert python.requests == ()
    assert rust.requests == ()
    assert python.response == rust.response
    assert isinstance(python.response, SDKError)
    assert python.response.exception_type == case.expected_exception_type
    assert python.response.status_code == case.expected_status_code
    assert case.expected_message in python.response.message


def test_ocr_subprocess_startup_smoke(
    startup_ocr_fixture: OcrParityCase,
    tmp_path: Path,
    sdk_workers: tuple[PythonScriptWorker, PythonScriptWorker],
) -> None:
    case_file: Final = tmp_path / "ocr-startup-smoke.json"
    case_file.write_text(startup_ocr_fixture.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8")
    python_worker, rust_worker = sdk_workers
    python: Final = run_execution(
        python_worker,
        case_file,
        SDKRoute.OCR.value,
        startup_ocr_fixture.provider_responses,
    )
    rust: Final = run_execution(
        rust_worker,
        case_file,
        SDKRoute.OCR.value,
        startup_ocr_fixture.provider_responses,
    )

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)


def _execute_worker_command(
    command_json: str,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerResult:
    try:
        command: Final = SDKCommand.model_validate_json(command_json)
        case_file: Final = Path(command.case_file)
        route: Final = SDKRoute(command.route)
        case: Final = OcrParityCase.model_validate_json(case_file.read_text(encoding="utf-8"))
        return WorkerSuccess(report=_execute_sdk_case(case.litellm_input, route, mock_url, event_loop))
    except Exception:
        return WorkerFailure(error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(_execute_worker_command, sys.argv[2])
