from __future__ import annotations

from typing import Final

from .....shared.tracing.steps import Engine, mapping
from ...fixtures import json_response, responses_body, responses_stream_events, sse_response
from ...models import GatewayRouteSpec, RouteFixture, TraceScenario, TraceSuite

MAPPINGS: Final = (
    mapping(
        span="python_responses_gateway_route", python_frame=r"response_api_endpoints/endpoints\.py:\d+ responses_api$"
    ),
    mapping(span="python_gateway_service", python_frame=r"ProxyBaseLLMRequestProcessing\.base_process_llm_request$"),
    mapping(rust_span="responses_gateway_route"),
    mapping(span="python_responses", python_frame=r"responses/main\.py:\d+ a?responses$"),
    mapping(span="python_provider_config", python_frame=r"ProviderConfigManager\.get_provider_responses_api_config$"),
    mapping(rust_span="validate_environment", python_frame=r"OpenAIResponsesAPIConfig\.validate_environment$"),
    mapping(rust_span="complete_url", python_frame=r"OpenAIResponsesAPIConfig\.get_complete_url$"),
    mapping(rust_span="transform_request", python_frame=r"OpenAIResponsesAPIConfig\.transform_responses_api_request$"),
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"OpenAIResponsesAPIConfig\.transform_response_api_response$"),
    mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$|Logging\.success_handler$"),
)

STREAM_MAPPINGS: Final = (
    mapping(span="python_stream_iterator", python_frame=r"ResponsesAPIStreamingIterator\.__init__$"),
    mapping(span="python_stream_next", python_frame=r"ResponsesAPIStreamingIterator\.__anext__$"),
    mapping(span="python_stream_transform", python_frame=r"OpenAIResponsesAPIConfig\.transform_streaming_response$"),
    mapping(span="python_downstream_stream", python_frame=r"DataGenerator\.__anext__$|async_data_generator$"),
)


def _fixture(_engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model_alias": "trace-model",
            "provider_model": "openai/gpt-5",
            "body": {"model": "trace-model", "input": "hello"},
        },
        provider_responses=(json_response(responses_body()),),
    )


def _stream_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _fixture(engine, base_url)
    return fixture.with_body(stream=True).derive(
        provider_responses=(sse_response(responses_stream_events()),),
    )


TRACE_SUITE: Final = TraceSuite(
    route=GatewayRouteSpec("responses"),
    scenarios=(
        TraceScenario(name="async-openai", fixture=_fixture, mappings=MAPPINGS, asynchronous=True),
        TraceScenario(
            name="async-openai-downstream-stream",
            fixture=_stream_fixture,
            mappings=(*MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
    ),
)
