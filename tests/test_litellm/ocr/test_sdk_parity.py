from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from enum import Enum
from pathlib import Path
from typing import Final, cast

import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from tests.test_litellm.ocr.fixture_models import MistralOcrParityInput, OcrParityCase
from tests.test_litellm.parity.compare import assert_parity
from tests.test_litellm.parity.models import SDKReport
from tests.test_litellm.parity.replay import replay_response
from tests.test_litellm.parity.runner import PythonScriptRunner, run_execution

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


def _execute_sdk_case(sdk_input: MistralOcrParityInput, route: SDKRoute, mock_url: str) -> SDKReport:
    import litellm

    call_kwargs: Final = _call_kwargs(sdk_input, mock_url, route)
    if route is SDKRoute.OCR:
        sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
        response: Final = sync_route(**call_kwargs)
        return SDKReport(response=response)
    async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
    async_response: Final = asyncio.run(async_route(**call_kwargs))
    return SDKReport(response=async_response)


@pytest.mark.parametrize("route", tuple(SDKRoute), ids=tuple(route.value for route in SDKRoute))
def test_recorded_ocr_sdk_parity(ocr_fixture: OcrParityCase, route: SDKRoute, tmp_path: Path) -> None:
    case_file: Final = tmp_path / f"{route.value}-ocr-parity-case.json"
    case_file.write_text(ocr_fixture.model_dump_json(indent=2, exclude_unset=True), encoding="utf-8")
    response: Final = ocr_fixture.upstream_response
    response_body: Final = response.body_bytes()
    response_headers: Final = tuple((header.name, header.value) for header in response.headers)
    runner: Final = PythonScriptRunner(
        entrypoint=Path(__file__),
        rust_env_var="LITELLM_USE_RUST_OCR",
        python_user_agent=PYTHON_HTTP_SENTINEL,
    )

    with replay_response(response.status_code, response_headers, response_body) as python_provider:
        python: Final = run_execution(
            runner,
            case_file,
            route.value,
            tmp_path / f"{route.value}-python-report.json",
            python_provider,
            rust_enabled=False,
        )
    with replay_response(response.status_code, response_headers, response_body) as rust_provider:
        rust: Final = run_execution(
            runner,
            case_file,
            route.value,
            tmp_path / f"{route.value}-rust-report.json",
            rust_provider,
            rust_enabled=True,
        )

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)


def _child_main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: test_sdk_parity.py CASE_FILE ROUTE MOCK_URL REPORT_FILE")
    case_file: Final = Path(sys.argv[1])
    route: Final = SDKRoute(sys.argv[2])
    mock_url: Final = sys.argv[3]
    report_file: Final = Path(sys.argv[4])
    case: Final = OcrParityCase.model_validate_json(case_file.read_text(encoding="utf-8"))
    report: Final = _execute_sdk_case(case.input, route, mock_url)
    report_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    _child_main()
