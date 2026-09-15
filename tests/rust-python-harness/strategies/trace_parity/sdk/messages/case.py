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
    mapping(rust_span="messages", python_frame=r"anthropic_interface/messages/__init__\.py:\d+ a?create$"),
    mapping(span="python_sanitize_empty_content", python_frame=r"strip_empty_content_blocks_from_anthropic_messages$"),
    mapping(span="python_sanitize_tool_ids", python_frame=r"sanitize_tool_use_ids_in_anthropic_messages$"),
    mapping(
        span="python_flatten_web_search", python_frame=r"flatten_unencrypted_web_search_results_in_anthropic_messages$"
    ),
    mapping(span="python_cache_control", python_frame=r"AnthropicCacheControlHook\.maybe_inject_cache_control$"),
    mapping(span="python_pre_request_hooks", python_frame=r"_execute_pre_request_hooks$"),
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
    mapping(span="python_logging_pre_call", python_frame=r"Logging\.pre_call$"),
    mapping(span="python_logging_post_call", python_frame=r"Logging\.post_call$"),
)

SUCCESS_MAPPINGS: Final = (mapping(span="python_success_callback", python_frame=r"Logging\.async_success_handler$"),)
FAILURE_MAPPINGS: Final = (
    mapping(span="python_failure_callback", python_frame=r"Logging\.failure_handler$"),
    mapping(span="python_async_failure_callback", python_frame=r"Logging\.async_failure_handler$"),
    mapping(span="python_exception_mapping", python_frame=r"(?<!_)exception_type$"),
)
STREAM_MAPPINGS: Final = (
    mapping(span="python_stream_wrapper", python_frame=r"AnthropicMessagesStreamingResponse\.__init__$"),
    mapping(span="python_stream_next", python_frame=r"AnthropicMessagesStreamingResponse\.__anext__$"),
    mapping(
        span="python_stream_iterator",
        python_frame=r"BaseAnthropicMessagesStreamingIterator\.get_async_streaming_response_iterator$",
    ),
    mapping(span="python_stream_chunks", python_frame=r"PassThroughStreamingHandler\.chunk_processor$"),
    mapping(
        span="python_stream_logging",
        python_frame=r"PassThroughStreamingHandler\._route_streaming_logging_to_handler$",
    ),
)

ANTHROPIC_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    *SUCCESS_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)

ANTHROPIC_FAILURE_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    *FAILURE_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)

AZURE_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    *SUCCESS_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"AzureAnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
    mapping(
        span="python_anthropic_transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
)

BEDROCK_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    *SUCCESS_MAPPINGS,
    mapping(
        rust_span="transform_request",
        python_frame=r"AmazonAnthropicClaudeMessagesConfig\.transform_anthropic_messages_request$",
    ),
    mapping(
        span="python_anthropic_transform_request",
        python_frame=r"(?<!Azure)AnthropicMessagesConfig\.transform_anthropic_messages_request$",
    ),
    mapping(
        span="python_bedrock_provider_config",
        python_frame=r"BedrockModelInfo\.get_bedrock_provider_config_for_messages_api$",
    ),
    mapping(span="python_aws_signing", python_frame=r"sign_request_off_loop_if_aws$"),
    mapping(
        span="python_aws_sign_request",
        python_frame=r"AmazonAnthropicClaudeMessagesConfig\.sign_request$|BaseAWSLLM\._sign_request$",
    ),
)

RETRY_MAPPINGS: Final = (
    *BEDROCK_MAPPINGS,
    mapping(
        span="python_retry_request_transform",
        python_frame=r"transform_anthropic_messages_request_on_http_error$",
    ),
    mapping(
        span="python_strip_invalid_thinking",
        python_frame=r"strip_thinking_blocks_from_anthropic_messages_request_dict$",
    ),
)

MOCK_MAPPINGS: Final = (
    *COMMON_MAPPINGS,
    mapping(span="python_mock_response", python_frame=r"messages/utils\.py:\d+ mock_response$"),
)


def _fixture(engine: Engine, provider: str) -> RouteFixture:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    return RouteFixture(
        kwargs={
            "model": f"{provider}/claude-sonnet-5",
            **({"body": {**conversation, "model": "claude-sonnet-5"}} if engine == "rust" else conversation),
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _anthropic_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "anthropic")


def _azure_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _fixture(engine, "azure_ai")


def _bedrock_kwargs(engine: Engine) -> dict[str, object]:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    return {
        "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        **(
            {"body": {**conversation, "model": "anthropic.claude-3-sonnet-20240229-v1:0"}}
            if engine == "rust"
            else conversation
        ),
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_region_name": "us-east-1",
    }


def _bedrock_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    response_fixture: Final = _fixture(engine, "anthropic")
    return RouteFixture(kwargs=_bedrock_kwargs(engine), provider_responses=response_fixture.provider_responses)


def _bedrock_retry_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    success_fixture: Final = _bedrock_fixture(engine, _base_url)
    messages: Final = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "old reasoning", "signature": ""},
                {"type": "text", "text": "partial answer"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    kwargs: Final = {
        **_bedrock_kwargs(engine),
        **(
            {"body": {"messages": messages, "max_tokens": 16, "model": "anthropic.claude-3-sonnet-20240229-v1:0"}}
            if engine == "rust"
            else {"messages": messages}
        ),
    }
    return success_fixture.derive(
        kwargs=kwargs,
        provider_responses=(
            json_response({"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}, status=400),
            *success_fixture.provider_responses,
        ),
    )


def _mock_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _fixture(engine, "anthropic")
    return fixture.derive(kwargs={"mock_response": "hello from mock"}, provider_responses=())


def _provider_error_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    fixture: Final = _fixture(engine, "anthropic")
    return fixture.derive(
        provider_responses=(
            json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
                status=400,
            ),
        ),
        expected_failure=True,
    )


def _sync_unsupported_fixture(engine: Engine, base_url: str) -> RouteFixture:
    if engine == "rust":
        return _anthropic_fixture(engine, base_url)
    fixture: Final = _fixture(engine, "anthropic")
    return fixture.derive(provider_responses=(), expected_failure=True)


def _stream_fixture_for(engine: Engine, provider: str) -> RouteFixture:
    fixture: Final = _fixture(engine, provider)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


def _stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _stream_fixture_for(engine, "anthropic")


def _azure_stream_fixture(engine: Engine, _base_url: str) -> RouteFixture:
    return _stream_fixture_for(engine, "azure_ai")


def _bedrock_stream_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(engine, base_url)
    events: Final = tuple(payload for _, payload in anthropic_stream_events())
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(aws_event_stream_response(events),),
        consume_stream=True,
    )


def _bedrock_stream_error_fixture(engine: Engine, base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(engine, base_url)
    start: Final = anthropic_stream_events(model="anthropic.claude-3-sonnet-20240229-v1:0")[0][1]
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(aws_event_stream_response((start, {"type": "message_stop"}), corrupt_last_frame=True),),
        expected_failure=True,
        consume_stream=True,
    )


SPEC: Final = RouteSpec("messages", ("create", "acreate"), ("messages", "amessages"), _anthropic_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="async-anthropic", fixture=_anthropic_fixture, mappings=ANTHROPIC_MAPPINGS, asynchronous=True
        ),
        TraceScenario(name="async-azure-ai", fixture=_azure_fixture, mappings=AZURE_MAPPINGS, asynchronous=True),
        TraceScenario(name="async-bedrock", fixture=_bedrock_fixture, mappings=BEDROCK_MAPPINGS, asynchronous=True),
        TraceScenario(
            name="async-bedrock-invalid-thinking-retry",
            fixture=_bedrock_retry_fixture,
            mappings=RETRY_MAPPINGS,
            asynchronous=True,
        ),
        TraceScenario(name="async-mock-response", fixture=_mock_fixture, mappings=MOCK_MAPPINGS, asynchronous=True),
        TraceScenario(
            name="async-anthropic-provider-error",
            fixture=_provider_error_fixture,
            mappings=ANTHROPIC_FAILURE_MAPPINGS,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-stream",
            fixture=_stream_fixture,
            mappings=(*ANTHROPIC_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-azure-ai-stream",
            fixture=_azure_stream_fixture,
            mappings=(*AZURE_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            mappings=(*BEDROCK_MAPPINGS, *STREAM_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="async-bedrock-event-stream-error",
            fixture=_bedrock_stream_error_fixture,
            mappings=(*BEDROCK_MAPPINGS, *STREAM_MAPPINGS, *FAILURE_MAPPINGS),
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-unsupported",
            fixture=_sync_unsupported_fixture,
            mappings=ANTHROPIC_MAPPINGS,
            asynchronous=False,
        ),
    ),
)
