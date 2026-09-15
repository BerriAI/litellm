from __future__ import annotations

from typing import Final

from .....shared.tracing.steps import Engine, mapping
from ...fixtures import (
    anthropic_response_body,
    anthropic_stream_events,
    json_response,
    responses_body,
    responses_stream_events,
    sse_response,
)
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite

COMMON_MAPPINGS: Final = (
    mapping(span="python_responses", python_frame=r"responses/main\.py:\d+ a?responses$"),
    mapping(
        span="python_responses_provider_config",
        python_frame=r"ProviderConfigManager\.get_provider_responses_api_config$",
    ),
    mapping(rust_span="responses_provider_config"),
    mapping(rust_span="validate_environment", python_frame=r"validate_environment$"),
    mapping(rust_span="complete_url", python_frame=r"get_complete_url$"),
    mapping(
        rust_span="transform_request",
        python_frame=r"(?<!AzureOpenAIResponsesAPIConfig\.)transform_responses_api_request$",
    ),
    mapping(
        rust_span="execute_responses_provider_call",
        python_frame=r"BaseLLMHTTPHandler\.(?:async_)?response_api_handler$",
    ),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"transform_response_api_response$"),
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$|Logging\.success_handler$"),
)

STREAM_MAPPINGS: Final = (
    mapping(
        span="python_responses_stream_iterator",
        python_frame=r"(?:Sync)?ResponsesAPIStreamingIterator\.__init__$",
    ),
    mapping(
        span="python_responses_stream_next",
        python_frame=r"(?:Sync)?ResponsesAPIStreamingIterator\.__a?next__$",
    ),
    mapping(span="python_responses_stream_transform", python_frame=r"transform_streaming_response$"),
)

FAILURE_MAPPINGS: Final = (
    mapping(span="python_exception_mapping", python_frame=r"(?<!_)exception_type$"),
    mapping(span="python_failure_callback", python_frame=r"Logging\.failure_handler$"),
    mapping(span="python_async_failure_callback", python_frame=r"Logging\.async_failure_handler$"),
)

AZURE_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(
        span="python_azure_transform_request",
        python_frame=r"AzureOpenAIResponsesAPIConfig\.transform_responses_api_request$",
    ),
)

BRIDGE_MAPPINGS: Final = (
    mapping(span="python_responses", python_frame=r"responses/main\.py:\d+ a?responses$"),
    mapping(
        span="python_responses_chat_bridge", python_frame=r"ResponsesToCompletionBridgeHandler\.response_api_handler$"
    ),
    mapping(span="python_chat_completions", python_frame=r"main\.py:\d+ a?completion$"),
    mapping(span="python_chat_transform_request", python_frame=r"AnthropicConfig\.transform_request$"),
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(span="python_chat_transform_response", python_frame=r"AnthropicConfig\.transform_response$"),
    mapping(span="python_chat_to_responses", python_frame=r"LiteLLMResponsesTransformationHandler\..*response"),
    mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$|Logging\.success_handler$"),
)


def _native_fixture(engine: Engine, provider: str) -> RouteFixture:
    model: Final = "gpt-5"
    return RouteFixture(
        kwargs={
            "model": f"{provider}/{model}",
            "input": "hello",
            **({"body": {"model": model, "input": "hello"}} if engine == "rust" else {}),
        },
        provider_responses=(json_response(responses_body(model=model)),),
    )


def _openai_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _native_fixture(engine, "openai")


def _azure_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _native_fixture(engine, "azure")
    return fixture.derive(kwargs={"api_version": "2025-04-01-preview"})


def _openai_stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(engine, _base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(responses_stream_events()),),
        consume_stream=True,
    )


def _provider_error_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(engine, _base_url)
    return fixture.derive(
        provider_responses=(
            json_response({"error": {"message": "bad request", "type": "invalid_request_error"}}, status=400),
        ),
        expected_failure=True,
    )


def _stream_failed_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(engine, base_url)
    failed_response: Final[dict[str, object]] = {
        **responses_body(),
        "status": "failed",
        "output": [],
        "error": {"message": "stream failed", "type": "server_error", "code": "server_error"},
    }
    events: Final = (
        (
            "response.created",
            {"type": "response.created", "response": {**failed_response, "status": "in_progress", "error": None}},
        ),
        ("response.failed", {"type": "response.failed", "response": failed_response}),
    )
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(events),),
        expected_failure=True,
        consume_stream=True,
    )


def _anthropic_bridge_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "input": "hello",
            "max_output_tokens": 16,
            **({"body": {"model": "claude-sonnet-5", "input": "hello"}} if engine == "rust" else {}),
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _anthropic_bridge_stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_bridge_fixture(engine, _base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


SPEC: Final = RouteSpec("responses", ("responses", "aresponses"), None, _openai_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(name="sync-openai", fixture=_openai_fixture, mappings=COMMON_MAPPINGS, asynchronous=False),
        TraceScenario(name="async-openai", fixture=_openai_fixture, mappings=COMMON_MAPPINGS, asynchronous=True),
        TraceScenario(
            name="sync-openai-stream",
            fixture=_openai_stream_fixture,
            mappings=(*COMMON_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=False,
        ),
        TraceScenario(
            name="async-openai-stream",
            fixture=_openai_stream_fixture,
            mappings=(*COMMON_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-openai-provider-error",
            fixture=_provider_error_fixture,
            mappings=(*COMMON_MAPPINGS, *FAILURE_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-openai-stream-failed",
            fixture=_stream_failed_fixture,
            mappings=(*COMMON_MAPPINGS, *STREAM_MAPPINGS, *FAILURE_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(name="async-azure", fixture=_azure_fixture, mappings=AZURE_MAPPINGS, asynchronous=True),
        TraceScenario(
            name="async-anthropic-chat-bridge",
            fixture=_anthropic_bridge_fixture,
            mappings=BRIDGE_MAPPINGS,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-chat-bridge-stream",
            fixture=_anthropic_bridge_stream_fixture,
            mappings=(
                *BRIDGE_MAPPINGS,
                mapping(span="python_chat_stream_wrapper", python_frame=r"CustomStreamWrapper\.__init__$"),
                mapping(span="python_chat_stream_next", python_frame=r"CustomStreamWrapper\.__anext__$"),
                mapping(
                    span="python_responses_bridge_stream_iterator",
                    python_frame=r"LiteLLMCompletionStreamingIterator\.__init__$|LiteLLMCompletionStreamingIterator\.__anext__$",
                ),
            ),
            asynchronous=True,
        ),
    ),
)
