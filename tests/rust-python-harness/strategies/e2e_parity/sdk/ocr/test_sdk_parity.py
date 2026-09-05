from __future__ import annotations

import asyncio
import sys
import tempfile
import traceback
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .....shared.parity.compare import assert_parity
from .....shared.parity.fixtures.store import fixture_id, recorded_fixtures
from .....shared.parity.models import (
    SDKCommand,
    SDKError,
    SDKReport,
    SDKSuccess,
    WorkerFailure,
    WorkerResult,
    WorkerSuccess,
    sdk_error_report,
)
from .....shared.parity.runner import (
    ExecutionVariant,
    SubprocessRunner,
    SubprocessWorker,
    execution_worker_pair,
    parity_worker_main,
)
from ...runner import E2ECheck
from .fixtures.config import configured_fixture_directory
from .fixtures.models import OcrParityCase, OcrSdkInput

API_KEY: Final = "test-key"
PYTHON_HTTP_SENTINEL: Final = "python-ocr-parity-fallback"
PYTHON_VARIANT: Final = ExecutionVariant(name="Python", environment=(("LITELLM_RUST", "0"),))
RUST_VARIANT: Final = ExecutionVariant(name="Rust", environment=(("LITELLM_RUST", "1"),))


class SDKRoute(str, Enum):
    OCR = "ocr"
    AOCR = "aocr"


class InvalidOcrCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model: str
    document: JsonValue
    expected_exception_type: str
    expected_status_code: int
    expected_message: str
    extra_kwargs: tuple[tuple[str, JsonValue], ...] = ()


class RecordedOcrWorkerCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["recorded"] = "recorded"
    case: OcrParityCase


class InvalidOcrWorkerCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["invalid"] = "invalid"
    case: InvalidOcrCase


OcrWorkerCase = Annotated[RecordedOcrWorkerCase | InvalidOcrWorkerCase, Field(discriminator="kind")]
OCR_WORKER_CASE_ADAPTER: Final[TypeAdapter[OcrWorkerCase]] = TypeAdapter(OcrWorkerCase)


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
    ),
    InvalidOcrCase(
        name="missing_image_url",
        model="azure_ai/doc-intelligence/prebuilt-read",
        document={"type": "image_url"},
        expected_exception_type="litellm.exceptions.APIConnectionError",
        expected_status_code=500,
        expected_message="Document URL is required",
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


def _execute_invalid_sdk_case(
    case: InvalidOcrCase,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> SDKReport:
    call_kwargs: Final[dict[str, object]] = {
        "model": case.model,
        "document": case.document,
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-litellm-parity-route": route.value},
        **dict(case.extra_kwargs),
    }
    return _execute_sdk_call(call_kwargs, route, event_loop)


def _check_recorded_ocr_sdk_parity(
    ocr_fixture: OcrParityCase,
    route: SDKRoute,
    case_file: Path,
    sdk_workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    python_worker, rust_worker = sdk_workers
    python: Final = python_worker.execute(case_file, route.value, ocr_fixture.provider_responses)
    rust: Final = rust_worker.execute(case_file, route.value, ocr_fixture.provider_responses)

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    if any(response.status_code >= 400 for response in ocr_fixture.provider_responses):
        assert isinstance(python.report, SDKError)


def _check_invalid_ocr_sdk_parity(
    case: InvalidOcrCase,
    route: SDKRoute,
    case_file: Path,
    sdk_workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    python_worker, rust_worker = sdk_workers
    python: Final = python_worker.execute(case_file, route.value, ())
    rust: Final = rust_worker.execute(case_file, route.value, ())

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    assert python.requests == ()
    assert rust.requests == ()
    assert isinstance(python.report, SDKError)
    assert python.report.exception_type == case.expected_exception_type
    assert python.report.status_code == case.expected_status_code
    assert case.expected_message in python.report.message


def _recorded_check_name(fixture: OcrParityCase, route: SDKRoute) -> str:
    case_input: Final = fixture.litellm_input
    provider: Final = case_input.custom_llm_provider
    prefix: Final = f"{provider}/{case_input.model}" if provider else case_input.model
    return f"recorded:{route.value}:{fixture_id(case_input, prefix)}"


def _write_worker_case(directory: Path, index: int, case: OcrWorkerCase) -> Path:
    case_file: Final = directory / f"case-{index}.json"
    case_file.write_text(OCR_WORKER_CASE_ADAPTER.dump_json(case).decode("utf-8"), encoding="utf-8")
    return case_file


@contextmanager
def parity_checks() -> Generator[tuple[E2ECheck, ...]]:
    fixtures: Final = tuple(
        fixture
        for fixture in recorded_fixtures(configured_fixture_directory(), OcrParityCase)
        if fixture.litellm_input.contract not in {"reducto_v3", "reducto_legacy"}
    )
    runner: Final = SubprocessRunner(
        entrypoint=Path(__file__),
        baseline_user_agent=PYTHON_HTTP_SENTINEL,
        route_label="OCR",
    )
    with tempfile.TemporaryDirectory(prefix="litellm-ocr-parity-") as raw_directory:
        directory: Final = Path(raw_directory)
        recorded_files: Final = tuple(
            _write_worker_case(directory, index, RecordedOcrWorkerCase(case=fixture))
            for index, fixture in enumerate(fixtures)
        )
        invalid_files: Final = tuple(
            _write_worker_case(directory, len(recorded_files) + index, InvalidOcrWorkerCase(case=case))
            for index, case in enumerate(INVALID_OCR_CASES)
        )
        with execution_worker_pair(runner, PYTHON_VARIANT, RUST_VARIANT) as workers:
            recorded: Final = tuple(
                E2ECheck(
                    _recorded_check_name(fixture, route),
                    partial(_check_recorded_ocr_sdk_parity, fixture, route, case_file, workers),
                )
                for fixture, case_file in zip(fixtures, recorded_files, strict=True)
                for route in SDKRoute
            )
            invalid: Final = tuple(
                E2ECheck(
                    f"invalid:{route.value}:{case.name}",
                    partial(_check_invalid_ocr_sdk_parity, case, route, case_file, workers),
                )
                for case, case_file in zip(INVALID_OCR_CASES, invalid_files, strict=True)
                for route in SDKRoute
            )
            yield (*recorded, *invalid)


def _execute_worker_command(
    command_json: str,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerResult:
    try:
        command: Final = SDKCommand.model_validate_json(command_json)
        case_file: Final = Path(command.case_file)
        route: Final = SDKRoute(command.route)
        worker_case: Final = OCR_WORKER_CASE_ADAPTER.validate_json(case_file.read_bytes())
        match worker_case:
            case RecordedOcrWorkerCase(case=recorded):
                return WorkerSuccess(report=_execute_sdk_case(recorded.litellm_input, route, mock_url, event_loop))
            case InvalidOcrWorkerCase(case=invalid):
                return WorkerSuccess(report=_execute_invalid_sdk_case(invalid, route, mock_url, event_loop))
    except Exception:
        return WorkerFailure(error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(_execute_worker_command, sys.argv[2])
