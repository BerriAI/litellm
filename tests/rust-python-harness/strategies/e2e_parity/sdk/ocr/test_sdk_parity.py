from __future__ import annotations

import asyncio
import datetime
import queue
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Coroutine, Generator, Mapping
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Final, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .....shared.parity.compare import assert_parity
from .....shared.parity.fixtures.store import fixture_id, recorded_fixtures
from .....shared.parity.models import (
    JSON_VALUE_ADAPTER,
    CallbackObservation,
    Execution,
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
CALLBACK_DELAY_SECONDS: Final = 0.05
CALLBACK_DRAIN_TIMEOUT_SECONDS: Final = 10.0
CALLBACK_TERMINALS: Final[tuple[Literal["success", "failure"], ...]] = ("success", "failure")
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


class CallbackOcrWorkerCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["callback"] = "callback"
    case: OcrParityCase
    terminal: Literal["success", "failure"]


OcrWorkerCase = Annotated[
    RecordedOcrWorkerCase | InvalidOcrWorkerCase | CallbackOcrWorkerCase,
    Field(discriminator="kind"),
]
OCR_WORKER_CASE_ADAPTER: Final[TypeAdapter[OcrWorkerCase]] = TypeAdapter(OcrWorkerCase)


class RecordingCallback(CustomLogger):
    def __init__(self) -> None:
        self.message_logging: Final = True
        self.turn_off_message_logging: Final = False
        self._observations: Final[queue.SimpleQueue[CallbackObservation]] = queue.SimpleQueue()

    def _normalized_kwargs(self, value: object, key: str | None = None) -> JsonValue:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if key != "api_base":
                return value
            parsed: Final = urlsplit(value)
            return urlunsplit(("", "", parsed.path, parsed.query, parsed.fragment))
        if isinstance(value, datetime.datetime):
            return "datetime"
        if isinstance(value, Exception):
            return sdk_error_report(value).model_dump(mode="json")
        if isinstance(value, BaseModel):
            return self._normalized_kwargs(value.model_dump(mode="json"), key)
        if isinstance(value, Mapping):
            if any(not isinstance(map_key, str) for map_key in value):
                raise TypeError("callback kwarg mappings must use string keys")
            return {
                map_key: self._normalized_kwargs(map_value, map_key)
                for map_key, map_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalized_kwargs(item) for item in value]
        raise TypeError(f"unsupported callback kwarg type: {type(value)}")

    def _record(
        self,
        hook: Literal[
            "log_success_event",
            "async_log_success_event",
            "log_failure_event",
            "async_log_failure_event",
        ],
        phase: Literal["success", "failure"],
        kwargs: dict[str, object],
        response_obj: object,
    ) -> None:
        raw_litellm_params: Final = kwargs.get("litellm_params")
        litellm_params: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], raw_litellm_params) if isinstance(raw_litellm_params, Mapping) else {}
        )
        raw_metadata: Final = litellm_params.get("metadata")
        metadata_mapping: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        )
        metadata: Final = JSON_VALUE_ADAPTER.validate_python(
            {key: metadata_mapping[key] for key in ("callback_profile", "sdk_route") if key in metadata_mapping}
        )
        raw_error: Final = kwargs.get("exception")
        error: Final = sdk_error_report(raw_error) if isinstance(raw_error, Exception) else None
        normalized_kwargs: Final = self._normalized_kwargs(kwargs)
        payload_source: Final = (
            response_obj.model_dump(mode="json") if isinstance(response_obj, BaseModel) else response_obj
        )
        payload: Final = JSON_VALUE_ADAPTER.validate_python(payload_source)
        raw_model: Final = kwargs.get("model")
        raw_call_type: Final = kwargs.get("call_type")
        raw_call_id: Final = kwargs.get("litellm_call_id")
        self._observations.put(
            CallbackObservation(
                hook=hook,
                phase=phase,
                model=raw_model if isinstance(raw_model, str) else None,
                call_type=str(raw_call_type) if raw_call_type is not None else None,
                litellm_call_id=raw_call_id if isinstance(raw_call_id, str) else None,
                metadata=metadata,
                kwargs=normalized_kwargs,
                payload=payload,
                error=error,
            )
        )

    def log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        del start_time, end_time
        time.sleep(CALLBACK_DELAY_SECONDS)
        self._record("log_success_event", "success", kwargs, response_obj)

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        del start_time, end_time
        await asyncio.sleep(CALLBACK_DELAY_SECONDS)
        self._record("async_log_success_event", "success", kwargs, response_obj)

    def log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        del start_time, end_time
        time.sleep(CALLBACK_DELAY_SECONDS)
        self._record("log_failure_event", "failure", kwargs, response_obj)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        del start_time, end_time
        await asyncio.sleep(CALLBACK_DELAY_SECONDS)
        self._record("async_log_failure_event", "failure", kwargs, response_obj)

    def observations(self) -> tuple[CallbackObservation, ...]:
        observations: Final = tuple(self._observations.get_nowait() for _ in range(self._observations.qsize()))
        return tuple(sorted(observations, key=lambda observation: observation.hook))


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


def _callback_call_id(route: SDKRoute, terminal: Literal["success", "failure"]) -> str:
    return f"ocr-callback-{route.value}-{terminal}"


def _callback_metadata(route: SDKRoute, terminal: Literal["success", "failure"]) -> dict[str, str]:
    return {"callback_profile": terminal, "sdk_route": route.value}


def _drain_callback_delivery(route: SDKRoute, event_loop: asyncio.AbstractEventLoop) -> None:
    if route is SDKRoute.AOCR:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        async def drain_async_callbacks() -> None:
            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=CALLBACK_DRAIN_TIMEOUT_SECONDS)
            await GLOBAL_LOGGING_WORKER.stop()

        event_loop.run_until_complete(drain_async_callbacks())

    from litellm.litellm_core_utils.thread_pool_executor import executor

    executor.shutdown(wait=True, cancel_futures=False)


def _execute_callback_sdk_case(
    case: OcrParityCase,
    route: SDKRoute,
    terminal: Literal["success", "failure"],
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerSuccess:
    callback: Final = RecordingCallback()
    call_kwargs: Final = {
        **_call_kwargs(case.litellm_input, mock_url, route),
        "callbacks": [callback],
        "litellm_call_id": _callback_call_id(route, terminal),
        "litellm_trace_id": _callback_call_id(route, terminal),
        "metadata": _callback_metadata(route, terminal),
    }
    report: Final = _execute_sdk_call(call_kwargs, route, event_loop)
    _drain_callback_delivery(route, event_loop)
    return WorkerSuccess(report=report, callbacks=callback.observations())


def _assert_callback_lifecycle(
    execution: Execution,
    route: SDKRoute,
    terminal: Literal["success", "failure"],
) -> None:
    observations: Final = execution.callbacks
    assert observations is not None, f"{route.value} {terminal} callbacks were not observed"
    expected_hooks: Final = (
        (f"log_{terminal}_event",)
        if route is SDKRoute.OCR
        else ("async_log_success_event",)
        if terminal == "success"
        else ("async_log_failure_event", "log_failure_event")
    )
    actual_hooks: Final = tuple(observation.hook for observation in observations)
    assert actual_hooks == expected_hooks, (
        f"{route.value} {terminal} expected callback hooks {expected_hooks}, received {actual_hooks}"
    )
    expected_call_id: Final = _callback_call_id(route, terminal)
    expected_metadata: Final = _callback_metadata(route, terminal)
    for observation in observations:
        assert observation.phase == terminal
        assert observation.call_type == route.value
        assert observation.litellm_call_id == expected_call_id
        assert observation.metadata == expected_metadata
        assert observation.model
        if terminal == "success":
            assert isinstance(execution.report, SDKSuccess)
            assert observation.payload == execution.report.response
            assert observation.error is None
        else:
            assert isinstance(execution.report, SDKError)
            assert observation.payload is None
            assert observation.error is not None
            assert observation.error.exception_type
            assert observation.error.message
            assert observation.error.status_code is not None
            assert observation.error.status_code >= 400


def _check_callback_ocr_sdk_parity(
    case: OcrParityCase,
    route: SDKRoute,
    terminal: Literal["success", "failure"],
    case_file: Path,
    runner: SubprocessRunner,
) -> None:
    with execution_worker_pair(runner, PYTHON_VARIANT, RUST_VARIANT) as workers:
        python_worker, rust_worker = workers
        python: Final = python_worker.execute(case_file, route.value, case.provider_responses)
        rust: Final = rust_worker.execute(case_file, route.value, case.provider_responses)

    _assert_callback_lifecycle(python, route, terminal)
    _assert_callback_lifecycle(rust, route, terminal)
    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)


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


def _callback_fixture(
    fixtures: tuple[OcrParityCase, ...],
    terminal: Literal["success", "failure"],
) -> OcrParityCase:
    matching: Final = tuple(
        fixture
        for fixture in fixtures
        if fixture.litellm_input.contract == "mistral"
        and (
            all(response.status_code < 400 for response in fixture.provider_responses)
            if terminal == "success"
            else any(response.status_code >= 400 for response in fixture.provider_responses)
        )
    )
    if not matching:
        raise AssertionError(f"no recorded Mistral OCR {terminal} fixture is available for callback parity")
    return min(matching, key=lambda fixture: fixture_id(fixture.litellm_input, fixture.litellm_input.model))


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
        callback_cases: Final[tuple[tuple[Literal["success", "failure"], OcrParityCase], ...]] = tuple(
            (terminal, _callback_fixture(fixtures, terminal)) for terminal in CALLBACK_TERMINALS
        )
        callback_files: Final = tuple(
            _write_worker_case(
                directory,
                len(recorded_files) + len(invalid_files) + index,
                CallbackOcrWorkerCase(case=case, terminal=terminal),
            )
            for index, (terminal, case) in enumerate(callback_cases)
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
            callbacks: Final = tuple(
                E2ECheck(
                    f"callback:{route.value}:{terminal}",
                    partial(_check_callback_ocr_sdk_parity, case, route, terminal, case_file, runner),
                )
                for (terminal, case), case_file in zip(callback_cases, callback_files, strict=True)
                for route in SDKRoute
            )
            yield (*recorded, *invalid, *callbacks)


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
            case CallbackOcrWorkerCase(case=callback_case, terminal=terminal):
                return _execute_callback_sdk_case(callback_case, route, terminal, mock_url, event_loop)
    except Exception:
        return WorkerFailure(error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(_execute_worker_command, sys.argv[2])
