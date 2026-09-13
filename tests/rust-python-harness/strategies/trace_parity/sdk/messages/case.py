from __future__ import annotations

import json
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, mapping
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite

COMMON_MAPPINGS: Final = (
    mapping(rust_span="messages", python_frame=r"anthropic_interface/messages/__init__\.py:\d+ a?create$"),
    mapping(
        span="python_messages_provider_config",
        python_frame=r"ProviderConfigManager\.get_provider_anthropic_messages_config$",
    ),
    mapping(rust_span="messages_provider_config"),
    mapping(rust_span="validate_environment", python_frame=r"validate_anthropic_messages_environment$"),
    mapping(rust_span="complete_url", python_frame=r"get_complete_url$"),
    mapping(
        span="python_messages_entry_handler",
        python_frame=r"messages/handler\.py:\d+ anthropic_messages_handler$",
    ),
    mapping(
        span="python_messages_handler_wrapper",
        python_frame=r"BaseLLMHTTPHandler\.anthropic_messages_handler$",
    ),
    mapping(
        rust_span="execute_messages_provider_call",
        python_frame=r"BaseLLMHTTPHandler\.async_anthropic_messages_handler$",
    ),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"(?<!async_)transform_anthropic_messages_response$"),
)

ANTHROPIC_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)

AZURE_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"AzureAnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
    mapping(
        span="python_anthropic_transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)


def _fixture(engine: Engine, provider: str) -> RouteFixture:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
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
            "model": f"{provider}/claude-sonnet-5",
            **({"body": {**conversation, "model": "claude-sonnet-5"}} if engine == "rust" else conversation),
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200, (HttpHeader(name="content-type", value="application/json"),), response
            ),
        ),
    )


def _anthropic_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "anthropic")


def _azure_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "azure_ai")


SPEC: Final = RouteSpec("messages", ("create", "acreate"), ("messages", "amessages"), _anthropic_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(name="anthropic", fixture=_anthropic_fixture, mappings=ANTHROPIC_MAPPINGS, modes=("async",)),
        TraceScenario(name="azure-ai", fixture=_azure_fixture, mappings=AZURE_MAPPINGS, modes=("async",)),
    ),
)
