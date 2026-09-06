from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .....shared.parity.fixtures.store import fixture_id, recorded_fixtures
from .....shared.parity.models import (
    SDKError,
    SDKReport,
    sdk_error_report,
    sdk_success,
)
from .....shared.parity.normalization import NormalizationSpec
from .....shared.parity.recorded_http import RecordedExchange, ReplayItem
from .....shared.parity.runner import (
    ExecutionVariant,
    parity_worker_main,
)
from ...runner import E2ECheck
from ..contract import BaseSdkParityContract, contract_checks, execute_contract_command
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
            return sdk_success(response)
        async_route: Final = cast(Callable[..., Coroutine[object, object, OCRResponse]], litellm.aocr)
        async_response: Final = event_loop.run_until_complete(async_route(**call_kwargs))
        return sdk_success(async_response)
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


def _recorded_check_name(fixture: OcrParityCase, route: SDKRoute) -> str:
    case_input: Final = fixture.litellm_input
    provider: Final = case_input.custom_llm_provider
    prefix: Final = f"{provider}/{case_input.model}" if provider else case_input.model
    return f"recorded:{route.value}:{fixture_id(case_input, prefix)}"


class OcrContract(BaseSdkParityContract[OcrWorkerCase]):
    @property
    def name(self) -> str:
        return "OCR"

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(route.value for route in SDKRoute)

    @property
    def baseline(self) -> ExecutionVariant:
        return PYTHON_VARIANT

    @property
    def candidate(self) -> ExecutionVariant:
        return RUST_VARIANT

    @property
    def baseline_user_agent(self) -> str:
        return PYTHON_HTTP_SENTINEL

    def cases(self) -> tuple[OcrWorkerCase, ...]:
        fixtures: Final = tuple(
            fixture
            for fixture in recorded_fixtures(configured_fixture_directory(), OcrParityCase)
            if fixture.litellm_input.contract not in {"reducto_v3", "reducto_legacy"}
        )
        return (
            *(RecordedOcrWorkerCase(case=fixture) for fixture in fixtures),
            *(InvalidOcrWorkerCase(case=case) for case in INVALID_OCR_CASES),
        )

    def dump_case(self, case: OcrWorkerCase) -> bytes:
        return OCR_WORKER_CASE_ADAPTER.dump_json(case)

    def load_case(self, data: bytes) -> OcrWorkerCase:
        return OCR_WORKER_CASE_ADAPTER.validate_json(data)

    def case_name(self, case: OcrWorkerCase, mode: str) -> str:
        route: Final = SDKRoute(mode)
        match case:
            case RecordedOcrWorkerCase(case=recorded):
                return _recorded_check_name(recorded, route)
            case InvalidOcrWorkerCase(case=invalid):
                return f"invalid:{mode}:{invalid.name}"

    def responses(self, case: OcrWorkerCase) -> tuple[ReplayItem, ...]:
        match case:
            case RecordedOcrWorkerCase(case=recorded):
                if not recorded.provider_requests:
                    return recorded.provider_responses
                return tuple(
                    RecordedExchange(request=request, response=response)
                    for request, response in zip(
                        recorded.provider_requests,
                        recorded.provider_responses,
                        strict=True,
                    )
                )
            case InvalidOcrWorkerCase():
                return ()

    def normalization(self, case: OcrWorkerCase) -> NormalizationSpec:
        del case
        return NormalizationSpec()

    def execute(
        self,
        case: OcrWorkerCase,
        mode: str,
        mock_url: str,
        event_loop: asyncio.AbstractEventLoop,
    ) -> SDKReport:
        route: Final = SDKRoute(mode)
        match case:
            case RecordedOcrWorkerCase(case=recorded):
                return _execute_sdk_case(recorded.litellm_input, route, mock_url, event_loop)
            case InvalidOcrWorkerCase(case=invalid):
                return _execute_invalid_sdk_case(invalid, route, mock_url, event_loop)

    def assert_baseline(self, case: OcrWorkerCase, report: SDKReport) -> None:
        match case:
            case RecordedOcrWorkerCase(case=recorded):
                if any(response.status_code >= 400 for response in recorded.provider_responses):
                    assert isinstance(report, SDKError)
            case InvalidOcrWorkerCase(case=invalid):
                assert isinstance(report, SDKError)
                assert report.exception_type == invalid.expected_exception_type
                assert report.status_code == invalid.expected_status_code
                assert invalid.expected_message in report.message


CONTRACT: Final = OcrContract()


def parity_checks() -> AbstractContextManager[tuple[E2ECheck, ...]]:
    return contract_checks(CONTRACT, Path(__file__))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--parity-worker":
        raise SystemExit("usage: test_sdk_parity.py --parity-worker MOCK_URL")
    parity_worker_main(lambda line, url, loop: execute_contract_command(CONTRACT, line, url, loop), sys.argv[2])
