from __future__ import annotations

import json
from typing import Final, cast

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, mapping
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite

COMMON_MAPPINGS: Final = (
    mapping(rust_span="ocr", python_frame=r"ocr/main\.py:\d+ a?ocr$"),
    mapping(rust_span="prepare_ocr_call", python_frame=r"ocr/main\.py:\d+ _prepare_ocr_request$"),
    mapping(rust_span="ocr_provider_config", python_frame=r"ProviderConfigManager\.get_provider_ocr_config$"),
    mapping(rust_span="supported_ocr_params", python_frame=r"get_supported_ocr_params$"),
    mapping(rust_span="map_ocr_params", python_frame=r"(?<!async_)map_ocr_params$"),
    mapping(rust_span="validate_environment", python_frame=r"(?<!_)validate_environment$"),
    mapping(rust_span="complete_url", python_frame=r"get_complete_url$"),
    mapping(rust_span="transform_ocr_request", python_frame=r"(?<!async_)transform_ocr_request$"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
)

SYNC_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(span="python_transform_ocr_response_wrapper", python_frame=r"BaseLLMHTTPHandler\._transform_ocr_response$"),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"MistralOCRConfig\.transform_ocr_response$",
    ),
)

ASYNC_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(span="python_ocr_wrapper", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.async_ocr$"),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"MistralOCRConfig\.transform_ocr_response$",
    ),
)

AZURE_COMMON_MAPPINGS: Final = (
    *COMMON_MAPPINGS[:7],
    mapping(
        rust_span="transform_ocr_request",
        python_frame=(
            r"AzureAIOCRConfig\.(?:async_)?transform_ocr_request$"
            r"|MistralOCRConfig\.transform_ocr_request$"
        ),
    ),
    COMMON_MAPPINGS[-1],
)
AZURE_SYNC_MAPPINGS: Final = (
    *AZURE_COMMON_MAPPINGS,
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(
        span="python_transform_ocr_response_wrapper",
        python_frame=r"BaseLLMHTTPHandler\._transform_ocr_response$",
    ),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"MistralOCRConfig\.transform_ocr_response$",
    ),
)
AZURE_ASYNC_MAPPINGS: Final = (
    *AZURE_COMMON_MAPPINGS,
    mapping(span="python_ocr_wrapper", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.async_ocr$"),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"MistralOCRConfig\.transform_ocr_response$",
    ),
)


def _fixture(engine: Engine, model: str, document: dict[str, str] | None = None) -> RouteFixture:
    response: Final = json.dumps(
        {
            "pages": [{"index": 0, "markdown": "hello"}],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1},
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": model,
            "document": document or {"type": "document_url", "document_url": "https://example.com/document.pdf"},
            **({"optional_params": {"pages": [0]}} if engine == "rust" else {"pages": [0]}),
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
    )


def _mistral_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "mistral/mistral-ocr-latest")


def _azure_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(
        engine,
        "azure_ai/pixtral-12b-2409",
        {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
    )


def _vertex_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _fixture(
        engine,
        "vertex_ai/mistral-ocr-maas",
        {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
    )
    vertex: Final = {"vertex_project": "trace-project", "vertex_location": "us-central1"}
    optional_params: Final = cast(dict[str, object], fixture.kwargs.get("optional_params", {}))
    return RouteFixture(
        kwargs={
            **fixture.kwargs,
            **({"optional_params": {**optional_params, **vertex}} if engine == "rust" else vertex),
        },
        provider_responses=fixture.provider_responses,
    )


def _vertex_deepseek_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    vertex: Final = {"vertex_project": "trace-project", "vertex_location": "us-central1"}
    return RouteFixture(
        kwargs={
            "model": "vertex_ai/deepseek-ocr-maas",
            "document": {"type": "image_url", "image_url": "data:image/png;base64,aGVsbG8="},
            **({"optional_params": vertex} if engine == "rust" else vertex),
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                json.dumps(
                    {
                        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).encode(),
            ),
        ),
    )


def _azure_document_intelligence_fixture(engine: Engine, base_url: str) -> RouteFixture:
    completed: Final = json.dumps(
        {
            "status": "succeeded",
            "analyzeResult": {
                "content": "hello",
                "pages": [
                    {
                        "pageNumber": 1,
                        "width": 8.5,
                        "height": 11,
                        "unit": "inch",
                        "lines": [{"content": "hello"}],
                    }
                ],
            },
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": "azure_ai/doc-intelligence/prebuilt-read",
            "document": {
                "type": "document_url",
                "document_url": "data:application/pdf;base64,aGVsbG8=",
            },
            **({"optional_params": {"pages": [0]}} if engine == "rust" else {"pages": [0]}),
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                202,
                (
                    HttpHeader(name="content-type", value="application/json"),
                    HttpHeader(name="operation-location", value=f"{base_url}/operations/trace"),
                ),
                b"{}",
            ),
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                completed,
            ),
        ),
    )


VERTEX_COMMON_MAPPINGS: Final = (
    *COMMON_MAPPINGS[:7],
    mapping(
        rust_span="transform_ocr_request",
        python_frame=(
            r"VertexAIOCRConfig\.(?:async_)?transform_ocr_request$"
            r"|MistralOCRConfig\.transform_ocr_request$"
        ),
    ),
    COMMON_MAPPINGS[-1],
)
VERTEX_SYNC_MAPPINGS: Final = (
    *VERTEX_COMMON_MAPPINGS,
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(span="python_transform_ocr_response_wrapper", python_frame=r"BaseLLMHTTPHandler\._transform_ocr_response$"),
    mapping(rust_span="transform_ocr_response", python_frame=r"MistralOCRConfig\.transform_ocr_response$"),
)
VERTEX_ASYNC_MAPPINGS: Final = (
    *VERTEX_COMMON_MAPPINGS,
    mapping(span="python_ocr_wrapper", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.async_ocr$"),
    mapping(rust_span="transform_ocr_response", python_frame=r"MistralOCRConfig\.transform_ocr_response$"),
)

DEEPSEEK_COMMON_MAPPINGS: Final = (
    mapping(rust_span="ocr", python_frame=r"ocr/main\.py:\d+ a?ocr$"),
    mapping(rust_span="prepare_ocr_call", python_frame=r"ocr/main\.py:\d+ _prepare_ocr_request$"),
    mapping(rust_span="ocr_provider_config", python_frame=r"ProviderConfigManager\.get_provider_ocr_config$"),
    mapping(rust_span="supported_ocr_params", python_frame=r"get_supported_ocr_params$"),
    mapping(rust_span="map_ocr_params", python_frame=r"(?<!async_)map_ocr_params$"),
    mapping(rust_span="validate_environment", python_frame=r"(?<!_)validate_environment$"),
    mapping(rust_span="complete_url", python_frame=r"get_complete_url$"),
    mapping(
        rust_span="transform_ocr_request",
        python_frame=r"VertexAIDeepSeekOCRConfig\.transform_ocr_request$",
    ),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"VertexAIDeepSeekOCRConfig\.transform_ocr_response$",
    ),
)
DEEPSEEK_SYNC_MAPPINGS: Final = (
    *DEEPSEEK_COMMON_MAPPINGS,
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(span="python_transform_ocr_response_wrapper", python_frame=r"BaseLLMHTTPHandler\._transform_ocr_response$"),
)
DEEPSEEK_ASYNC_MAPPINGS: Final = (
    *DEEPSEEK_COMMON_MAPPINGS,
    mapping(span="python_ocr_wrapper", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(
        span="python_async_transform_ocr_request",
        python_frame=r"VertexAIDeepSeekOCRConfig\.async_transform_ocr_request$",
    ),
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.async_ocr$"),
)

DOCUMENT_INTELLIGENCE_COMMON_MAPPINGS: Final = (
    mapping(rust_span="ocr", python_frame=r"ocr/main\.py:\d+ a?ocr$"),
    mapping(rust_span="prepare_ocr_call", python_frame=r"ocr/main\.py:\d+ _prepare_ocr_request$"),
    mapping(rust_span="ocr_provider_config", python_frame=r"ProviderConfigManager\.get_provider_ocr_config$"),
    mapping(
        rust_span="supported_ocr_params", python_frame=r"AzureDocumentIntelligenceOCRConfig\.get_supported_ocr_params$"
    ),
    mapping(rust_span="map_ocr_params", python_frame=r"AzureDocumentIntelligenceOCRConfig\.map_ocr_params$"),
    mapping(
        rust_span="validate_environment", python_frame=r"AzureDocumentIntelligenceOCRConfig\.validate_environment$"
    ),
    mapping(rust_span="complete_url", python_frame=r"AzureDocumentIntelligenceOCRConfig\.get_complete_url$"),
    mapping(
        rust_span="transform_ocr_request", python_frame=r"AzureDocumentIntelligenceOCRConfig\.transform_ocr_request$"
    ),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(
        rust_span="poll_document_intelligence",
        python_frame=r"AzureDocumentIntelligenceOCRConfig\._poll_operation_(?:sync|async)$",
    ),
    mapping(
        rust_span="transform_ocr_response",
        python_frame=r"AzureDocumentIntelligenceOCRConfig\._transform_completed_response$",
    ),
)
DOCUMENT_INTELLIGENCE_SYNC_MAPPINGS: Final = (
    *DOCUMENT_INTELLIGENCE_COMMON_MAPPINGS,
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(span="python_transform_ocr_response_wrapper", python_frame=r"BaseLLMHTTPHandler\._transform_ocr_response$"),
    mapping(
        span="python_provider_transform_response",
        python_frame=r"AzureDocumentIntelligenceOCRConfig\.transform_ocr_response$",
    ),
    mapping(span="python_poll_http_request", python_frame=r"HTTPHandler\.get$"),
)
DOCUMENT_INTELLIGENCE_ASYNC_MAPPINGS: Final = (
    *DOCUMENT_INTELLIGENCE_COMMON_MAPPINGS,
    mapping(span="python_ocr_wrapper", python_frame=r"BaseLLMHTTPHandler\.ocr$"),
    mapping(rust_span="execute_ocr_provider_call", python_frame=r"BaseLLMHTTPHandler\.async_ocr$"),
    mapping(
        span="python_provider_transform_response",
        python_frame=r"AzureDocumentIntelligenceOCRConfig\.async_transform_ocr_response$",
    ),
    mapping(span="python_poll_http_request", python_frame=r"AsyncHTTPHandler\.get$"),
)


SPEC: Final = RouteSpec("ocr", ("ocr", "aocr"), ("ocr", "aocr"), _mistral_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="mistral",
            fixture=_mistral_fixture,
            mappings=COMMON_MAPPINGS,
            sync_mappings=SYNC_MAPPINGS,
            async_mappings=ASYNC_MAPPINGS,
        ),
        TraceScenario(
            name="azure-ai",
            fixture=_azure_fixture,
            mappings=AZURE_COMMON_MAPPINGS,
            sync_mappings=AZURE_SYNC_MAPPINGS,
            async_mappings=AZURE_ASYNC_MAPPINGS,
        ),
        TraceScenario(
            name="azure-document-intelligence",
            fixture=_azure_document_intelligence_fixture,
            mappings=DOCUMENT_INTELLIGENCE_COMMON_MAPPINGS,
            sync_mappings=DOCUMENT_INTELLIGENCE_SYNC_MAPPINGS,
            async_mappings=DOCUMENT_INTELLIGENCE_ASYNC_MAPPINGS,
        ),
        TraceScenario(
            name="vertex-ai",
            fixture=_vertex_fixture,
            mappings=VERTEX_COMMON_MAPPINGS,
            sync_mappings=VERTEX_SYNC_MAPPINGS,
            async_mappings=VERTEX_ASYNC_MAPPINGS,
        ),
        TraceScenario(
            name="vertex-deepseek",
            fixture=_vertex_deepseek_fixture,
            mappings=DEEPSEEK_COMMON_MAPPINGS,
            sync_mappings=DEEPSEEK_SYNC_MAPPINGS,
            async_mappings=DEEPSEEK_ASYNC_MAPPINGS,
        ),
    ),
)
