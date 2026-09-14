from __future__ import annotations

from typing import Final

from .....shared.tracing.steps import Engine, mapping
from ...fixtures import anthropic_response_body, anthropic_stream_events, json_response, sse_response
from ...models import GatewayRouteSpec, RouteFixture, TraceScenario, TraceSuite

MAPPINGS: Final = (
    mapping(span="python_chat_gateway_route", python_frame=r"proxy_server\.py:\d+ chat_completion$"),
    mapping(span="python_gateway_service", python_frame=r"ProxyBaseLLMRequestProcessing\.base_process_llm_request$"),
    mapping(span="python_chat_entrypoint", python_frame=r"main\.py:\d+ a?completion$"),
    mapping(span="python_provider_config", python_frame=r"ProviderConfigManager\.get_provider_chat_config$"),
    mapping(rust_span="validate_environment", python_frame=r"(?<!_)validate_environment$"),
    mapping(rust_span="transform_request", python_frame=r"AnthropicConfig\.transform_request$"),
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"AnthropicConfig\.transform_response$"),
    mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$|Logging\.success_handler$"),
)

STREAM_MAPPINGS: Final = (
    mapping(span="python_stream_wrapper", python_frame=r"CustomStreamWrapper\.__init__$"),
    mapping(span="python_stream_next", python_frame=r"CustomStreamWrapper\.__anext__$"),
    mapping(span="python_stream_chunk", python_frame=r"CustomStreamWrapper\.chunk_creator$"),
    mapping(span="python_downstream_stream", python_frame=r"DataGenerator\.__anext__$|async_data_generator$"),
)


def _fixture(_engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model_alias": "trace-model",
            "provider_model": "anthropic/claude-sonnet-5",
            "body": {
                "model": "trace-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            },
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _stream_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _fixture(engine, base_url)
    return fixture.with_body(stream=True).derive(
        provider_responses=(sse_response(anthropic_stream_events()),),
    )


TRACE_SUITE: Final = TraceSuite(
    route=GatewayRouteSpec("chat_completions", rust_supported=False),
    scenarios=(
        TraceScenario(name="async-anthropic", fixture=_fixture, mappings=MAPPINGS, asynchronous=True),
        TraceScenario(
            name="async-anthropic-downstream-stream",
            fixture=_stream_fixture,
            mappings=(*MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
    ),
)
