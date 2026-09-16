from __future__ import annotations

import json
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, mapping
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
    response: Final = json.dumps(
        {
            "id": "msg_trace",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    ).encode()
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hello"}],
            **({"optional_params": {"max_tokens": 16}} if engine == "rust" else {"max_tokens": 16}),
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
    )


def _bedrock_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    response: Final = json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
            "metrics": {"latencyMs": 1},
        }
    ).encode()
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
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
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
            name="anthropic",
            fixture=_anthropic_fixture,
            mappings=COMMON_MAPPINGS,
            sync_mappings=SYNC_MAPPINGS,
            async_mappings=ASYNC_MAPPINGS,
        ),
        TraceScenario(
            name="bedrock",
            fixture=_bedrock_fixture,
            mappings=BEDROCK_COMMON_MAPPINGS,
            sync_mappings=BEDROCK_SYNC_MAPPINGS,
            async_mappings=BEDROCK_ASYNC_MAPPINGS,
        ),
    ),
)
