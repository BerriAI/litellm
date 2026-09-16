from __future__ import annotations

from typing import Final

from .....shared.tracing.steps import Engine, mapping
from ...fixtures import (
    anthropic_response_body,
    anthropic_stream_events,
    aws_event_stream_response,
    json_response,
    sse_response,
)
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite

COMMON_MAPPINGS: Final = (
    mapping(span="python_provider_config", python_frame=r"ProviderConfigManager\.get_provider_chat_config$"),
    mapping(rust_span="chat_completions_provider_config"),
    mapping(
        span="python_supported_openai_params",
        python_frame=r"litellm_core_utils/get_supported_openai_params\.py:\d+ get_supported_openai_params$",
    ),
    mapping(
        span="python_provider_supported_openai_params",
        python_frame=r"AnthropicConfig\.get_supported_openai_params$",
    ),
    mapping(rust_span="supported_openai_params"),
    mapping(rust_span="validate_environment", python_frame=r"(?<!_)validate_environment$"),
    mapping(rust_span="transform_request", python_frame=r"(?<!async_)transform_request$"),
    mapping(rust_span="execute_chat_completions_provider_call"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"(?<!async_)transform_response$"),
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(span="python_logging_post_call", python_frame=r"Logging\.post_call$"),
    mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$|Logging\.success_handler$"),
)

STREAM_MAPPINGS: Final = (
    mapping(span="python_stream_wrapper", python_frame=r"CustomStreamWrapper\.__init__$"),
    mapping(span="python_stream_next", python_frame=r"CustomStreamWrapper\.__next__$|CustomStreamWrapper\.__anext__$"),
    mapping(span="python_stream_chunk", python_frame=r"CustomStreamWrapper\.chunk_creator$"),
    mapping(span="python_stream_finalize", python_frame=r"CustomStreamWrapper\._finalize_completed_stream$"),
)

FAILURE_MAPPINGS: Final = (
    mapping(span="python_exception_mapping", python_frame=r"(?<!_)exception_type$"),
    mapping(span="python_failure_callback", python_frame=r"Logging\.failure_handler$"),
    mapping(span="python_async_failure_callback", python_frame=r"Logging\.async_failure_handler$"),
)

SYNC_MAPPINGS: Final = (
    mapping(span="python_chat_completions", python_frame=r"main\.py:\d+ completion$"),
    mapping(rust_span="chat_completions"),
    mapping(span="python_completion_handler", python_frame=r"ChatCompletion\.completion$"),
    *COMMON_MAPPINGS,
)

ASYNC_MAPPINGS: Final = (
    mapping(span="python_chat_completions", python_frame=r"main\.py:\d+ acompletion$"),
    mapping(rust_span="chat_completions"),
    mapping(span="python_completion_wrapper", python_frame=r"main\.py:\d+ completion$"),
    mapping(span="python_completion_handler", python_frame=r"ChatCompletion\.completion$"),
    mapping(span="python_async_completion_handler", python_frame=r"acompletion_function$"),
    *COMMON_MAPPINGS,
)


def _anthropic_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hello"}],
            **({"optional_params": {"max_tokens": 16}} if engine == "rust" else {"max_tokens": 16}),
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _bedrock_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    response: Final[dict[str, object]] = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        "metrics": {"latencyMs": 1},
    }
    credentials: Final = {
        "aws_access_key_id": "test-access",
        "aws_secret_access_key": "test-secret",
        "aws_region_name": "us-east-1",
    }
    return RouteFixture(
        kwargs={
            "model": "bedrock/us-east-1/anthropic.claude-v2",
            "messages": [{"role": "user", "content": "hello"}],
            **(
                {"optional_params": {**credentials, "maxTokens": 16}}
                if engine == "rust"
                else {**credentials, "max_tokens": 16}
            ),
        },
        provider_responses=(json_response(response),),
    )


def _anthropic_stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(engine, _base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


def _bedrock_stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(engine, _base_url)
    events: Final[tuple[dict[str, object], ...]] = (
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3}}},
    )
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(aws_event_stream_response(events),),
        consume_stream=True,
    )


def _provider_error_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(engine, _base_url)
    return fixture.derive(
        provider_responses=(
            json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
                status=400,
            ),
        ),
        expected_failure=True,
    )


def _stream_error_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(engine, base_url)
    events: Final = (
        anthropic_stream_events()[0],
        ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}),
    )
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(events),),
        expected_failure=True,
        consume_stream=True,
    )


SPEC: Final = RouteSpec(
    "chat_completions",
    ("completion", "acompletion"),
    ("chat_completions", "achat_completions"),
    _anthropic_fixture,
)
BEDROCK_COMMON_MAPPINGS: Final = (
    mapping(rust_span="chat_completions_provider_config"),
    mapping(rust_span="supported_openai_params"),
    mapping(rust_span="execute_chat_completions_provider_call"),
    mapping(rust_span="validate_environment"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(span="python_transform_response", python_frame=r"AmazonConverseConfig\._transform_response$"),
)
BEDROCK_SYNC_MAPPINGS: Final = (
    mapping(span="python_chat_completions", python_frame=r"main\.py:\d+ completion$"),
    mapping(rust_span="chat_completions"),
    mapping(span="python_transform_request", python_frame=r"AmazonConverseConfig\._transform_request$"),
    *BEDROCK_COMMON_MAPPINGS,
)
BEDROCK_ASYNC_MAPPINGS: Final = (
    mapping(span="python_chat_completions", python_frame=r"main\.py:\d+ acompletion$"),
    mapping(span="python_completion_wrapper", python_frame=r"main\.py:\d+ completion$"),
    mapping(rust_span="chat_completions"),
    *BEDROCK_COMMON_MAPPINGS,
)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="sync-anthropic",
            fixture=_anthropic_fixture,
            mappings=SYNC_MAPPINGS,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-anthropic",
            fixture=_anthropic_fixture,
            mappings=ASYNC_MAPPINGS,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-anthropic-stream",
            fixture=_anthropic_stream_fixture,
            mappings=(*SYNC_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=False,
        ),
        TraceScenario(
            name="async-anthropic-stream",
            fixture=_anthropic_stream_fixture,
            mappings=(*ASYNC_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-provider-error",
            fixture=_provider_error_fixture,
            mappings=(*ASYNC_MAPPINGS, *FAILURE_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-stream-error",
            fixture=_stream_error_fixture,
            mappings=(*ASYNC_MAPPINGS, *STREAM_MAPPINGS, *FAILURE_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-bedrock",
            fixture=_bedrock_fixture,
            mappings=BEDROCK_SYNC_MAPPINGS,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-bedrock",
            fixture=_bedrock_fixture,
            mappings=BEDROCK_ASYNC_MAPPINGS,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            mappings=(*BEDROCK_SYNC_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=False,
        ),
        TraceScenario(
            name="async-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            mappings=(*BEDROCK_ASYNC_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
    ),
)
