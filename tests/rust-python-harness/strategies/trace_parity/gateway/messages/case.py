from __future__ import annotations

import json
from typing import Final

from .....shared.parity.recorded_http import HttpHeader, RecordedHttpResponse
from .....shared.tracing.steps import Engine, mapping
from ...models import GatewayRouteSpec, RouteFixture, TraceScenario, TraceSuite


GATEWAY_MAPPINGS: Final = (
    mapping(
        span="python_messages_gateway_route",
        python_frame=r"anthropic_endpoints/endpoints\.py:\d+ anthropic_response$",
    ),
    mapping(rust_span="messages_gateway_route"),
    mapping(
        span="python_messages_gateway_service",
        python_frame=r"ProxyBaseLLMRequestProcessing\.base_process_llm_request$",
    ),
    mapping(rust_span="messages_gateway_service"),
    mapping(rust_span="messages"),
    mapping(
        span="python_messages_provider_config",
        python_frame=r"ProviderConfigManager\.get_provider_anthropic_messages_config$",
    ),
    mapping(rust_span="messages_provider_config"),
    mapping(rust_span="validate_environment", python_frame=r"validate_anthropic_messages_environment$"),
    mapping(rust_span="complete_url", python_frame=r"get_complete_url$"),
    mapping(span="python_messages_entry_handler", python_frame=r"messages/handler\.py:\d+ anthropic_messages_handler$"),
    mapping(span="python_messages_handler_wrapper", python_frame=r"BaseLLMHTTPHandler\.anthropic_messages_handler$"),
    mapping(
        rust_span="execute_messages_provider_call",
        python_frame=r"BaseLLMHTTPHandler\.async_anthropic_messages_handler$",
    ),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$|HTTPHandler\.post$"),
    mapping(rust_span="transform_response", python_frame=r"(?<!async_)transform_anthropic_messages_response$"),
)


def _fixture(_engine: Engine, provider: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model_alias": "trace-model",
            "provider_model": f"{provider}/claude-sonnet-5",
            "body": {
                "model": "trace-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            },
        },
        provider_responses=(
            RecordedHttpResponse.from_bytes(
                200,
                (HttpHeader(name="content-type", value="application/json"),),
                json.dumps(
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
                ).encode(),
            ),
        ),
    )


def _anthropic_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "anthropic")


def _azure_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "azure_ai")


ANTHROPIC_MAPPINGS: Final = (
    *GATEWAY_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)
AZURE_MAPPINGS: Final = (
    *GATEWAY_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"AzureAnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
    mapping(
        span="python_anthropic_transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)

TRACE_SUITE: Final = TraceSuite(
    route=GatewayRouteSpec("messages"),
    scenarios=(
        TraceScenario(name="anthropic", fixture=_anthropic_fixture, mappings=ANTHROPIC_MAPPINGS, modes=("async",)),
        TraceScenario(name="azure-ai", fixture=_azure_fixture, mappings=AZURE_MAPPINGS, modes=("async",)),
    ),
)
