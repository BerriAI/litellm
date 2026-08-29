from __future__ import annotations

import asyncio
import sys
import traceback
from collections.abc import Callable, Coroutine, Generator
from enum import Enum
from pathlib import Path
from typing import Final, cast

import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm.ocr.fixture_models import MistralOcrParityInput, OcrParityCase
from tests.test_litellm.parity.compare import assert_parity
from tests.test_litellm.parity.models import SDKCommand, SDKReport, WorkerFailure, WorkerResult, WorkerSuccess
from tests.test_litellm.parity.runner import (
    WORKER_RESULT_PREFIX,
    PythonScriptRunner,
    PythonScriptWorker,
    execution_worker,
    run_execution,
)

API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-ocr-parity-fallback"


class SDKRoute(str, Enum):
    OCR = "ocr"
    AOCR = "aocr"


def _call_kwargs(sdk_input: MistralOcrParityInput, mock_url: str, route: SDKRoute) -> dict[str, object]:
    return {
        **sdk_input.as_sdk_kwargs(),
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-ocr-parity-route": route.value},
    }


def _execute_sdk_case(
    sdk_input: MistralOcrParityInput,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    import litellm

    call_kwargs: Final = _call_kwargs(sdk_input, mock_url, route)
    if route is SDKRoute.OCR:
        sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
        response: Final = sync_route(**call_kwargs)
        return SDKReport(response=response)
    async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
    async_response: Final = event_loop.run_until_complete(async_route(**call_kwargs))
    return SDKReport(response=async_response)


@pytest.fixture(scope="module")
def sdk_workers() -> Generator[tuple[PythonScriptWorker, PythonScriptWorker]]:
    runner: Final = PythonScriptRunner(
        entrypoint=Path(__file__),
        rust_env_var="LITELLM_USE_RUST_OCR",
        python_user_agent=PYTHON_HTTP_SENTINEL,
    )
    with execution_worker(runner, rust_enabled=False) as python_worker:
        with execution_worker(runner, rust_enabled=True) as rust_worker:
            yield python_worker, rust_worker


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_recorded_ocr_sdk_parity(
    ocr_fixture: OcrParityCase,
    route: SDKRoute,
    tmp_path: Path,
    sdk_workers: tuple[PythonScriptWorker, PythonScriptWorker],
) -> None:
    case_file: Final = tmp_path / f"{route.value}-ocr-parity-case.json"
    case_file.write_text(ocr_fixture.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8")
    response: Final = ocr_fixture.provider_response
    response_body: Final = response.body_bytes()
    response_headers: Final = tuple((header.name, header.value) for header in response.headers)
    python_worker, rust_worker = sdk_workers
    python: Final = run_execution(
        python_worker,
        case_file,
        route.value,
        response.status_code,
        response_headers,
        response_body,
    )
    rust: Final = run_execution(
        rust_worker,
        case_file,
        route.value,
        response.status_code,
        response_headers,
        response_body,
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


def _worker_main(mock_url: str) -> None:
    event_loop: Final = asyncio.new_event_loop()
    try:
        for line in sys.stdin:
            sys.stdout.write(
                f"{WORKER_RESULT_PREFIX}{_execute_worker_command(line, mock_url, event_loop).model_dump_json()}\n"
            )
            sys.stdout.flush()
    finally:
        event_loop.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    _worker_main(sys.argv[2])
