from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable
from functools import partial
from types import FunctionType
from typing import Final, Protocol, cast

import pytest
from pydantic import StrictInt, StrictStr, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.rust_bridge import get_native_bridge
from litellm.rust_bridge import ocr as rust_ocr_bridge
from tests.sdk_function_trace import (
    FunctionTraceEvent,
    TraceScenario,
    TraceStep,
    assert_function_trace_parity,
)
from tests.sdk_function_trace.mock_provider import MockProviderResponse, mock_provider
from tests.sdk_function_trace.profiler import profile_python

MODEL: Final = "mistral-ocr-latest"
DOCUMENT: Final[dict[str, str]] = {
    "type": "document_url",
    "document_url": "https://example.com/document.pdf",
}
OPTIONAL_PARAMS: Final[dict[str, object]] = {"pages": [0], "unsupported": True}
RESPONSE_DATA: Final[dict[str, object]] = {
    "pages": [{"index": 0, "markdown": "hello"}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1},
}
PROVIDER_RESPONSE: Final = MockProviderResponse(
    status_code=200,
    headers=(("content-type", "application/json"),),
    body=json.dumps(RESPONSE_DATA).encode(),
)


class _TraceEventPayload(TypedDict):
    function: ReadOnly[StrictStr]
    depth: ReadOnly[StrictInt]


class _TraceResponsePayload(TypedDict):
    response: ReadOnly[object]
    trace: ReadOnly[list[_TraceEventPayload]]


_TRACE_RESPONSE: Final = TypeAdapter(_TraceResponsePayload)


class _NativeTraceOcr(Protocol):
    def __call__(
        self,
        *,
        model: str,
        document: dict[str, str],
        api_key: str,
        api_base: str,
        optional_params: dict[str, object],
        trace: bool,
    ) -> object: ...


def _invoke_python() -> object:
    previous_enabled: Final = rust_ocr_bridge.rust_ocr_enabled()
    rust_ocr_bridge.use_litellm_rust(False)
    try:
        with mock_provider(PROVIDER_RESPONSE) as api_base:
            return litellm.ocr(
                model=f"mistral/{MODEL}",
                document=DOCUMENT,
                api_key="test-key",
                api_base=api_base,
                pages=[0],
                unsupported=True,
            )
    finally:
        rust_ocr_bridge.use_litellm_rust(previous_enabled)


def _invoke_rust(*, asynchronous: bool = False) -> tuple[FunctionTraceEvent, ...]:
    bridge: Final = get_native_bridge()
    if bridge is None:
        raise AssertionError("The native Rust bridge is required for function-trace parity")
    trace_ocr: Final = cast(_NativeTraceOcr, bridge.aocr if asynchronous else bridge.ocr)
    with mock_provider(PROVIDER_RESPONSE) as api_base:
        invoke: Final = partial(
            trace_ocr,
            model=f"mistral/{MODEL}",
            document=DOCUMENT,
            api_key="test-key",
            api_base=api_base,
            optional_params=OPTIONAL_PARAMS,
            trace=True,
        )

        async def invoke_async() -> object:
            return await cast(Awaitable[object], invoke())

        raw_result: Final = asyncio.run(invoke_async()) if asynchronous else invoke()
    result: Final = _TRACE_RESPONSE.validate_python(raw_result)
    return tuple(
        FunctionTraceEvent(
            function=event["function"],
            depth=event["depth"],
        )
        for event in result["trace"]
    )


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_mistral_ocr_transformation_function_trace_parity(asynchronous: bool) -> None:
    assert_function_trace_parity(
        TraceScenario(
            steps=(
                TraceStep(cast(FunctionType, MistralOCRConfig.get_supported_ocr_params), depth=0),
                TraceStep(cast(FunctionType, MistralOCRConfig.map_ocr_params), depth=0),
                TraceStep(cast(FunctionType, MistralOCRConfig.get_supported_ocr_params), depth=1),
                TraceStep(cast(FunctionType, MistralOCRConfig.transform_ocr_request), depth=0),
                TraceStep(cast(FunctionType, MistralOCRConfig.transform_ocr_response), depth=0),
            ),
            invoke_python=_invoke_python,
            invoke_rust=partial(_invoke_rust, asynchronous=asynchronous),
        )
    )


class First:
    @staticmethod
    def run() -> None:
        return None


class Second:
    @staticmethod
    def run() -> None:
        return None


def test_profiler_matches_code_objects_and_keeps_repeated_calls() -> None:
    with profile_python((First.run,)) as profiler:
        Second.run()
        First.run()
        First.run()

    assert profiler.events == [
        FunctionTraceEvent(function="run", depth=0),
        FunctionTraceEvent(function="run", depth=0),
    ]


def test_profiler_records_selected_function_nesting_depth() -> None:
    class Nested:
        @staticmethod
        def run() -> None:
            First.run()

    with profile_python((Nested.run, First.run)) as profiler:
        Nested.run()

    assert profiler.events == [
        FunctionTraceEvent(function="run", depth=0),
        FunctionTraceEvent(function="run", depth=1),
    ]


def test_profiler_restores_previous_profiler_after_failure() -> None:
    previous: Final = sys.getprofile()

    with pytest.raises(RuntimeError, match="stop"):
        with profile_python((First.run,)):
            raise RuntimeError("stop")

    assert sys.getprofile() is previous
