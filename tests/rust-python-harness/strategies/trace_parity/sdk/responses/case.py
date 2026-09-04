from __future__ import annotations

import json
from typing import Final

from .....shared.parity.recorded_http import (
    HttpHeader,
    RecordedExchange,
    RecordedHttpResponse,
    RecordedRequestMatcher,
)
from .....shared.tracing.steps import Engine, mapping
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _response(body: dict[str, object], path: str) -> tuple[RecordedExchange, ...]:
    return (
        RecordedExchange(
            request=RecordedRequestMatcher(method="POST", path=path),
            response=RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                json.dumps(body).encode(),
            ),
        ),
    )


def _openai_native_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "openai/gpt-5",
            "input": "hello",
            **({"optional_params": {"max_output_tokens": 16}} if engine == "rust" else {"max_output_tokens": 16}),
        },
        provider_responses=_response(
            {
                "id": "resp_trace",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {
                        "id": "msg_trace",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            },
            "/responses",
        ),
    )


def _anthropic_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "input": "hello",
            **({"optional_params": {"max_output_tokens": 16}} if engine == "rust" else {"max_output_tokens": 16}),
        },
        provider_responses=_response(
            {
                "id": "msg_trace",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
            "/v1/messages",
        ),
    )


def _bedrock_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "bedrock/us-east-1/anthropic.claude-v2",
            "input": "hello",
            **({"optional_params": {"max_output_tokens": 16}} if engine == "rust" else {"max_output_tokens": 16}),
        },
        provider_responses=_response(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
                "metrics": {"latencyMs": 1},
            },
            "/model/anthropic.claude-v2/converse",
        ),
    )


def _openai_adapter_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "openai/gpt-5",
            "input": "hello",
            "use_chat_completions_api": True,
            **({"optional_params": {"max_output_tokens": 16}} if engine == "rust" else {"max_output_tokens": 16}),
        },
        provider_responses=_response(
            {
                "id": "chatcmpl_trace",
                "object": "chat.completion",
                "created": 123,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
            "/chat/completions",
        ),
    )


NATIVE_INTERNAL: Final = (
    mapping(rust_span="responses_transport"),
    mapping(rust_span="execute_responses_provider_call"),
    mapping(rust_span="http_request"),
)

ADAPTER_INTERNAL: Final = (
    mapping(rust_span="responses_transport"),
    mapping(rust_span="transform_responses_request_to_chat_completions"),
    mapping(rust_span="chat_completions"),
    mapping(rust_span="chat_completions_provider_config"),
    mapping(rust_span="supported_openai_params"),
    mapping(rust_span="execute_chat_completions_provider_call"),
    mapping(rust_span="validate_environment"),
    mapping(rust_span="transform_request"),
    mapping(rust_span="http_request"),
    mapping(rust_span="transform_response"),
    mapping(rust_span="transform_chat_completions_response_to_responses"),
)

BEDROCK_ADAPTER_INTERNAL: Final = tuple(
    item for item in ADAPTER_INTERNAL if item.span not in {"transform_request", "transform_response"}
)
OPENAI_ADAPTER_INTERNAL: Final = tuple(
    item for item in ADAPTER_INTERNAL if item.span != "chat_completions_provider_config"
)

SYNC_ROOT: Final = (mapping(rust_span="responses", python_frame=r"main\.py:\d+ responses$"),)
ASYNC_ROOT: Final = (
    mapping(rust_span="responses", python_frame=r"main\.py:\d+ aresponses$"),
    mapping(span="python_responses_wrapper", python_frame=r"main\.py:\d+ responses$"),
)

SPEC: Final = RouteSpec(
    "responses",
    ("responses", "aresponses"),
    ("responses", "aresponses"),
    _openai_native_fixture,
)

TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="openai-native",
            fixture=_openai_native_fixture,
            mappings=NATIVE_INTERNAL,
            sync_mappings=(*SYNC_ROOT, *NATIVE_INTERNAL),
            async_mappings=(*ASYNC_ROOT, *NATIVE_INTERNAL),
        ),
        TraceScenario(
            name="anthropic-adapter",
            fixture=_anthropic_fixture,
            mappings=ADAPTER_INTERNAL,
            sync_mappings=(*SYNC_ROOT, *ADAPTER_INTERNAL),
            async_mappings=(*ASYNC_ROOT, *ADAPTER_INTERNAL),
        ),
        TraceScenario(
            name="bedrock-adapter",
            fixture=_bedrock_fixture,
            mappings=BEDROCK_ADAPTER_INTERNAL,
            sync_mappings=(*SYNC_ROOT, *BEDROCK_ADAPTER_INTERNAL),
            async_mappings=(*ASYNC_ROOT, *BEDROCK_ADAPTER_INTERNAL),
        ),
        TraceScenario(
            name="openai-forced-adapter",
            fixture=_openai_adapter_fixture,
            mappings=OPENAI_ADAPTER_INTERNAL,
            sync_mappings=(*SYNC_ROOT, *OPENAI_ADAPTER_INTERNAL),
            async_mappings=(*ASYNC_ROOT, *OPENAI_ADAPTER_INTERNAL),
        ),
    ),
)
