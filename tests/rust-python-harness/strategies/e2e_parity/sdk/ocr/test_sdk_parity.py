from __future__ import annotations

import asyncio
import json
import math
import sys
import tempfile
import traceback
import uuid
from collections.abc import Callable, Coroutine, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .....shared.parity.billing import BillingObserver
from .....shared.parity.compare import assert_parity
from .....shared.parity.fixtures.store import fixture_id, recorded_fixtures
from .....shared.parity.models import (
    BillingObservation,
    BillingUsageMetric,
    SDKCommand,
    SDKError,
    SDKReport,
    SDKSuccess,
    WorkerFailure,
    WorkerResult,
    WorkerSuccess,
    sdk_error_report,
)
from .....shared.parity.recorded_http import RecordedHttpResponse, RecordedResponse
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
CALLBACK_TIMEOUT_SECONDS: Final = 10.0
CONTROLLED_PAGE_COST: Final = 0.017
CONTROLLED_ANNOTATION_COST: Final = 0.031
CONTROLLED_INPUT_TOKEN_COST: Final = 0.0007
CONTROLLED_OUTPUT_TOKEN_COST: Final = 0.0019
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
JSON_OBJECT_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


class _CacheClearable(Protocol):
    def cache_clear(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExactBillingCase:
    name: str
    fixture: OcrParityCase
    responses: tuple[RecordedResponse, ...]
    expected_usage: tuple[BillingUsageMetric, ...]
    expected_input_cost: float
    expected_output_cost: float


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


def _ocr_billable_usage(response: object) -> tuple[BillingUsageMetric, ...]:
    if not isinstance(response, OCRResponse):
        raise ValueError(f"expected OCRResponse, got {type(response).__name__}")
    usage: Final = response.usage_info
    if usage is None:
        return ()
    values: Final = (
        ("pages_processed", usage.pages_processed),
        ("pages_processed_annotation", usage.pages_processed_annotation),
        ("credits", usage.credits),
        ("prompt_tokens", usage.prompt_tokens),
        ("completion_tokens", usage.completion_tokens),
        ("total_tokens", usage.total_tokens),
    )
    return tuple(
        BillingUsageMetric(name=name, value=value)
        for name, value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _pricing_provider(sdk_input: OcrSdkInput) -> str:
    if sdk_input.custom_llm_provider is not None:
        return sdk_input.custom_llm_provider
    return {
        "mistral": "mistral",
        "azure_mistral": "azure_ai",
        "azure_document_intelligence": "azure_ai",
        "vertex_mistral": "vertex_ai",
        "vertex_deepseek": "vertex_ai",
        "reducto_v3": "reducto",
        "reducto_legacy": "reducto",
    }[sdk_input.contract]


def _pricing_model_key(sdk_input: OcrSdkInput, provider: str) -> str:
    return sdk_input.model if sdk_input.model.startswith(f"{provider}/") else f"{provider}/{sdk_input.model}"


def _pricing_model_name(sdk_input: OcrSdkInput) -> str:
    provider: Final = _pricing_provider(sdk_input)
    return sdk_input.model.removeprefix(f"{provider}/")


@contextmanager
def _controlled_ocr_pricing(sdk_input: OcrSdkInput) -> Generator[None]:
    import litellm

    provider: Final = _pricing_provider(sdk_input)
    model_key: Final = _pricing_model_key(sdk_input, provider)
    original_model_cost: Final = cast(  # cast-ok: LiteLLM's public model_cost is a mapping loaded from JSON
        Mapping[str, object], litellm.model_cost
    )
    get_model_info_cache: Final = cast(  # cast-ok: get_model_info attaches cache_clear at module initialization
        _CacheClearable, litellm.get_model_info
    )
    raw_entry: Final = original_model_cost.get(model_key, {})
    existing_entry: Final = cast(  # cast-ok: the runtime mapping check establishes the narrowed branch
        Mapping[str, object], raw_entry if isinstance(raw_entry, Mapping) else {}
    )
    controlled_cost_fields: Final = frozenset(
        {
            "input_cost_per_token",
            "output_cost_per_token",
            "ocr_cost_per_page",
            "annotation_cost_per_page",
            "ocr_cost_per_credit",
        }
    )
    base_entry: Final = {key: value for key, value in existing_entry.items() if key not in controlled_cost_fields}
    pricing: Final = (
        {
            "input_cost_per_token": CONTROLLED_INPUT_TOKEN_COST,
            "output_cost_per_token": CONTROLLED_OUTPUT_TOKEN_COST,
        }
        if sdk_input.contract == "vertex_deepseek"
        else {
            "ocr_cost_per_page": CONTROLLED_PAGE_COST,
            "annotation_cost_per_page": CONTROLLED_ANNOTATION_COST,
        }
    )
    entry: Final = {
        **base_entry,
        "litellm_provider": provider,
        "mode": "ocr",
        **pricing,
    }
    litellm.model_cost = {  # test-quality-ok: each parity worker is isolated and needs controlled pricing
        **original_model_cost,
        model_key: entry,
    }
    get_model_info_cache.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost  # test-quality-ok: restore isolated worker pricing after each case
        get_model_info_cache.cache_clear()


def _execute_sdk_call(
    call_kwargs: dict[str, object],
    route: SDKRoute,
    event_loop: asyncio.AbstractEventLoop,
    capture_billing: bool,
) -> tuple[SDKReport, BillingObservation | None]:
    import litellm

    call_id: Final = f"ocr-parity-{uuid.uuid4()}"
    observer: Final = BillingObserver(call_id, _ocr_billable_usage) if capture_billing else None
    callback: Final = observer.async_log_success_event if observer is not None and route is SDKRoute.AOCR else observer
    observed_call_kwargs: Final = (
        {
            **call_kwargs,
            "litellm_call_id": call_id,
            "success_callback": [callback],
        }
        if observer is not None
        else call_kwargs
    )
    if route is SDKRoute.OCR:
        try:
            sync_route: Final = cast(Callable[..., OCRResponse], litellm.ocr)
            response: Final = sync_route(**observed_call_kwargs)
        except Exception as error:
            if observer is not None:
                observer.assert_no_success_callback()
            return sdk_error_report(error), None
        billing: Final = observer.observation(CALLBACK_TIMEOUT_SECONDS) if observer is not None else None
        return SDKSuccess(response=response.model_dump(mode="json")), billing

    try:
        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        async_response: Final = event_loop.run_until_complete(async_route(**observed_call_kwargs))
    except Exception as error:
        event_loop.run_until_complete(GLOBAL_LOGGING_WORKER.flush())
        if observer is not None:
            observer.assert_no_success_callback()
        return sdk_error_report(error), None
    async_billing: Final = (
        event_loop.run_until_complete(asyncio.to_thread(observer.observation, CALLBACK_TIMEOUT_SECONDS))
        if observer is not None
        else None
    )
    event_loop.run_until_complete(GLOBAL_LOGGING_WORKER.flush())
    return SDKSuccess(response=async_response.model_dump(mode="json")), async_billing


def _execute_sdk_case(
    sdk_input: OcrSdkInput,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
    capture_billing: bool,
) -> tuple[SDKReport, BillingObservation | None]:
    call_kwargs: Final = _call_kwargs(sdk_input, mock_url, route)
    if not capture_billing:
        return _execute_sdk_call(call_kwargs, route, event_loop, capture_billing=False)
    with _controlled_ocr_pricing(sdk_input):
        return _execute_sdk_call(call_kwargs, route, event_loop, capture_billing=True)


def _execute_invalid_sdk_case(
    case: InvalidOcrCase,
    route: SDKRoute,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
    capture_billing: bool,
) -> tuple[SDKReport, BillingObservation | None]:
    call_kwargs: Final[dict[str, object]] = {
        "model": case.model,
        "document": case.document,
        "api_base": mock_url,
        "api_key": API_KEY,
        "extra_headers": {"x-litellm-parity-route": route.value},
        **dict(case.extra_kwargs),
    }
    return _execute_sdk_call(call_kwargs, route, event_loop, capture_billing=capture_billing)


def _check_recorded_ocr_sdk_parity(
    ocr_fixture: OcrParityCase,
    route: SDKRoute,
    case_file: Path,
    sdk_workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    python_worker, rust_worker = sdk_workers
    python: Final = python_worker.execute(
        case_file,
        route.value,
        ocr_fixture.provider_responses,
        capture_billing=True,
    )
    rust: Final = rust_worker.execute(
        case_file,
        route.value,
        ocr_fixture.provider_responses,
        capture_billing=True,
    )

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    _assert_recorded_billing(python.report, python.billing, ocr_fixture.litellm_input, route)
    if any(response.status_code >= 400 for response in ocr_fixture.provider_responses):
        assert isinstance(python.report, SDKError)
        assert python.billing is None


def _assert_recorded_billing(
    report: SDKReport,
    billing: BillingObservation | None,
    sdk_input: OcrSdkInput,
    route: SDKRoute,
) -> None:
    if isinstance(report, SDKError):
        assert billing is None
        return
    assert isinstance(report, SDKSuccess)
    assert billing is not None
    assert billing.callback_count == 1
    assert billing.call_type == route.value
    assert billing.pricing_model == _pricing_model_name(sdk_input)
    assert billing.custom_llm_provider == _pricing_provider(sdk_input)
    assert billing.cost_calculation_status == "calculated"
    assert billing.cost_failure_diagnostic is None
    positive_billable_usage: Final = tuple(
        metric for metric in billing.billable_usage if metric.name != "total_tokens" and metric.value > 0
    )
    assert positive_billable_usage
    assert billing.response_cost is not None
    assert billing.response_cost > 0
    assert billing.cost_breakdown is not None
    assert billing.cost_breakdown.total_cost is not None
    assert math.isclose(billing.cost_breakdown.total_cost, billing.response_cost, rel_tol=1e-12)


def _check_exact_billing_parity(
    exact_case: ExactBillingCase,
    route: SDKRoute,
    case_file: Path,
    sdk_workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    python_worker, rust_worker = sdk_workers
    python: Final = python_worker.execute(
        case_file,
        route.value,
        exact_case.responses,
        capture_billing=True,
    )
    rust: Final = rust_worker.execute(
        case_file,
        route.value,
        exact_case.responses,
        capture_billing=True,
    )

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    _assert_recorded_billing(python.report, python.billing, exact_case.fixture.litellm_input, route)
    assert python.billing is not None
    expected_cost: Final = exact_case.expected_input_cost + exact_case.expected_output_cost
    assert python.billing.billable_usage == exact_case.expected_usage
    assert python.billing.response_cost is not None
    assert math.isclose(python.billing.response_cost, expected_cost, rel_tol=1e-12)
    assert python.billing.cost_breakdown is not None
    assert python.billing.cost_breakdown.input_cost is not None
    assert python.billing.cost_breakdown.output_cost is not None
    assert math.isclose(python.billing.cost_breakdown.input_cost, exact_case.expected_input_cost, rel_tol=1e-12)
    assert math.isclose(python.billing.cost_breakdown.output_cost, exact_case.expected_output_cost, rel_tol=1e-12)


def _check_invalid_ocr_sdk_parity(
    case: InvalidOcrCase,
    route: SDKRoute,
    case_file: Path,
    sdk_workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    python_worker, rust_worker = sdk_workers
    python: Final = python_worker.execute(case_file, route.value, (), capture_billing=True)
    rust: Final = rust_worker.execute(case_file, route.value, (), capture_billing=True)

    assert_parity(python, rust, PYTHON_HTTP_SENTINEL)
    assert python.requests == ()
    assert rust.requests == ()
    assert isinstance(python.report, SDKError)
    assert python.billing is None
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


def _response_json(response: RecordedResponse) -> Mapping[str, object] | None:
    if not isinstance(response, RecordedHttpResponse) or response.status_code >= 400:
        return None
    try:
        return JSON_OBJECT_ADAPTER.validate_json(response.body_bytes())
    except ValueError:
        return None


def _usage_from_fixture(fixture: OcrParityCase) -> Mapping[str, object] | None:
    if len(fixture.provider_responses) != 1:
        return None
    response_json: Final = _response_json(fixture.provider_responses[0])
    if response_json is None:
        return None
    raw_usage: Final = response_json.get("usage_info", response_json.get("usage"))
    return (
        cast(Mapping[str, object], raw_usage)  # cast-ok: fixture JSON objects have string keys
        if isinstance(raw_usage, Mapping)
        else None
    )


def _with_usage(response: RecordedResponse, usage: Mapping[str, int]) -> RecordedHttpResponse:
    if not isinstance(response, RecordedHttpResponse):
        raise AssertionError("exact OCR billing case requires a non-streaming HTTP response")
    response_json: Final = _response_json(response)
    if response_json is None:
        raise AssertionError("exact OCR billing case requires a successful JSON response")
    body: Final = json.dumps({**response_json, "usage_info": dict(usage)}, separators=(",", ":")).encode()
    return RecordedHttpResponse.from_bytes(response.status_code, response.headers, body)


def _exact_billing_cases(fixtures: tuple[OcrParityCase, ...]) -> tuple[ExactBillingCase, ...]:
    def has_usage(fixture: OcrParityCase, expected: Mapping[str, int]) -> bool:
        usage: Final = _usage_from_fixture(fixture)
        return usage is not None and all(usage.get(name) == value for name, value in expected.items())

    page_fixture: Final = next(
        fixture
        for fixture in fixtures
        if fixture.litellm_input.contract == "mistral"
        and fixture.litellm_input.model.endswith("mistral-ocr-2512")
        and has_usage(fixture, {"pages_processed": 5})
    )
    token_fixture: Final = next(
        fixture
        for fixture in fixtures
        if fixture.litellm_input.contract == "vertex_deepseek"
        and has_usage(fixture, {"prompt_tokens": 281, "completion_tokens": 6})
    )
    annotation_response: Final = _with_usage(
        page_fixture.provider_responses[0],
        {"pages_processed": 2, "pages_processed_annotation": 3},
    )
    return (
        ExactBillingCase(
            name="pages",
            fixture=page_fixture,
            responses=page_fixture.provider_responses,
            expected_usage=(BillingUsageMetric(name="pages_processed", value=5),),
            expected_input_cost=5 * CONTROLLED_PAGE_COST,
            expected_output_cost=0.0,
        ),
        ExactBillingCase(
            name="annotation",
            fixture=page_fixture,
            responses=(annotation_response,),
            expected_usage=(
                BillingUsageMetric(name="pages_processed", value=2),
                BillingUsageMetric(name="pages_processed_annotation", value=3),
            ),
            expected_input_cost=2 * CONTROLLED_PAGE_COST + 3 * CONTROLLED_ANNOTATION_COST,
            expected_output_cost=0.0,
        ),
        ExactBillingCase(
            name="vertex-deepseek-tokens",
            fixture=token_fixture,
            responses=token_fixture.provider_responses,
            expected_usage=(
                BillingUsageMetric(name="prompt_tokens", value=281),
                BillingUsageMetric(name="completion_tokens", value=6),
                BillingUsageMetric(name="total_tokens", value=287),
            ),
            expected_input_cost=281 * CONTROLLED_INPUT_TOKEN_COST,
            expected_output_cost=6 * CONTROLLED_OUTPUT_TOKEN_COST,
        ),
    )


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
        exact_cases: Final = _exact_billing_cases(fixtures)
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
            exact: Final = tuple(
                E2ECheck(
                    f"billing:{route.value}:{exact_case.name}",
                    partial(
                        _check_exact_billing_parity,
                        exact_case,
                        route,
                        recorded_files[fixtures.index(exact_case.fixture)],
                        workers,
                    ),
                )
                for exact_case in exact_cases
                for route in SDKRoute
            )
            yield (*recorded, *invalid, *exact)


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
                report, billing = _execute_sdk_case(
                    recorded.litellm_input,
                    route,
                    mock_url,
                    event_loop,
                    command.capture_billing,
                )
                return WorkerSuccess(report=report, billing=billing)
            case InvalidOcrWorkerCase(case=invalid):
                report, billing = _execute_invalid_sdk_case(
                    invalid,
                    route,
                    mock_url,
                    event_loop,
                    command.capture_billing,
                )
                return WorkerSuccess(report=report, billing=billing)
    except Exception:
        return WorkerFailure(error=traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(_execute_worker_command, sys.argv[2])
