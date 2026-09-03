from __future__ import annotations

import json
from typing import Final

import pytest

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, pipeline_issues, pipeline_steps, step, trace_diff
from ..execution import RouteFixture, RouteSpec, collect_trace

STEPS: Final = (
    step("ocr", r"ocr/main\.py:\d+ a?ocr$", "ocr"),
    step("prepare_ocr_call", r"ocr/main\.py:\d+ _prepare_ocr_request$", "prepare_ocr_call"),
    step("get_provider_ocr_config", r"ProviderConfigManager\.get_provider_ocr_config$", "ocr_provider_config"),
    step("supported_ocr_params", r"get_supported_ocr_params$", "supported_ocr_params"),
    step("map_ocr_params", r"(?<!async_)map_ocr_params$", "map_ocr_params"),
    step("validate_environment", r"(?<!_)validate_environment$", "validate_environment"),
    step("complete_url", r"get_complete_url$", "complete_url"),
    step("transform_ocr_request", r"(?<!async_)transform_ocr_request$", "transform_ocr_request"),
    step("execute_ocr_provider_call", r"BaseLLMHTTPHandler\.(?:async_)?ocr$", "execute_ocr_provider_call"),
    step("http_request", r"AsyncHTTPHandler\.post$|HTTPHandler\.post$", "http_request"),
    step("transform_ocr_response", r"(?<!async_)transform_ocr_response$", "transform_ocr_response"),
)
EDGES: Final = (
    ("ocr", "prepare_ocr_call"),
    ("prepare_ocr_call", "get_provider_ocr_config"),
    ("get_provider_ocr_config", "transform_ocr_request"),
    ("supported_ocr_params", "transform_ocr_request"),
    ("map_ocr_params", "transform_ocr_request"),
    ("validate_environment", "http_request"),
    ("complete_url", "http_request"),
    ("transform_ocr_request", "http_request"),
    ("execute_ocr_provider_call", "http_request"),
    ("http_request", "transform_ocr_response"),
)


def _fixture(engine: Engine) -> RouteFixture:
    response: Final = json.dumps(
        {
            "pages": [{"index": 0, "markdown": "hello"}],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1},
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": "mistral/mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            **({"optional_params": {"pages": [0]}} if engine == "rust" else {"pages": [0]}),
        },
        provider_response=RecordedHttpResponse.from_bytes(
            200, (HttpHeader(name="content-type", value="application/json"),), response
        ),
    )


SPEC: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _fixture)


@pytest.mark.parametrize("asynchronous", (False, True), ids=("sync", "async"))
def test_trace_parity(asynchronous: bool) -> None:
    python: Final = pipeline_steps("python", collect_trace(SPEC, "python", asynchronous=asynchronous), STEPS)
    rust: Final = pipeline_steps("rust", collect_trace(SPEC, "rust", asynchronous=asynchronous), STEPS)

    assert pipeline_issues("python", python, STEPS, EDGES) == ()
    assert pipeline_issues("rust", rust, STEPS, EDGES) == ()
    assert trace_diff(python, rust).matches
    assert python == rust
