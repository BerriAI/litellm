from __future__ import annotations

from typing import Final

from ...fixtures import (
    anthropic_response_body,
    anthropic_stream_events,
    aws_event_stream_response,
    json_response,
    sse_response,
)
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _fixture(provider: str) -> RouteFixture:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    return RouteFixture(
        kwargs={
            "model": f"{provider}/claude-sonnet-5",
            **conversation,
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _anthropic_fixture(_base_url: str) -> RouteFixture:
    return _fixture("anthropic")


def _azure_fixture(_base_url: str) -> RouteFixture:
    return _fixture("azure_ai")


def _bedrock_kwargs() -> dict[str, object]:
    conversation: Final = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16}
    return {
        "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        **conversation,
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "aws_region_name": "us-east-1",
    }


def _bedrock_fixture(_base_url: str) -> RouteFixture:
    response_fixture: Final = _fixture("anthropic")
    return RouteFixture(kwargs=_bedrock_kwargs(), provider_responses=response_fixture.provider_responses)


def _bedrock_retry_fixture(_base_url: str) -> RouteFixture:
    success_fixture: Final = _bedrock_fixture(_base_url)
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
    kwargs: Final = {**_bedrock_kwargs(), "messages": messages}
    return success_fixture.derive(
        kwargs=kwargs,
        provider_responses=(
            json_response({"message": "messages.1.content.0: Invalid `signature` in `thinking` block"}, status=400),
            *success_fixture.provider_responses,
        ),
    )


def _mock_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _fixture("anthropic")
    return fixture.derive(kwargs={"mock_response": "hello from mock"}, provider_responses=())


def _provider_error_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _fixture("anthropic")
    return fixture.derive(
        provider_responses=(
            json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
                status=400,
            ),
        ),
        expected_failure=True,
    )


def _sync_unsupported_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _fixture("anthropic")
    return fixture.derive(provider_responses=(), expected_failure=True)


def _stream_fixture_for(provider: str) -> RouteFixture:
    fixture: Final = _fixture(provider)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


def _stream_fixture(_base_url: str) -> RouteFixture:
    return _stream_fixture_for("anthropic")


def _azure_stream_fixture(_base_url: str) -> RouteFixture:
    return _stream_fixture_for("azure_ai")


def _bedrock_stream_fixture(base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(base_url)
    events: Final = tuple(payload for _, payload in anthropic_stream_events())
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(aws_event_stream_response(events),),
        consume_stream=True,
    )


def _bedrock_stream_error_fixture(base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(base_url)
    start: Final = anthropic_stream_events(model="anthropic.claude-3-sonnet-20240229-v1:0")[0][1]
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(aws_event_stream_response((start, {"type": "message_stop"}), corrupt_last_frame=True),),
        expected_failure=True,
        consume_stream=True,
    )


SPEC: Final = RouteSpec("messages", ("create", "acreate"), _anthropic_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(name="async-anthropic", fixture=_anthropic_fixture, asynchronous=True),
        TraceScenario(name="async-azure-ai", fixture=_azure_fixture, asynchronous=True),
        TraceScenario(name="async-bedrock", fixture=_bedrock_fixture, asynchronous=True),
        TraceScenario(
            name="async-bedrock-invalid-thinking-retry",
            fixture=_bedrock_retry_fixture,
            asynchronous=True,
        ),
        TraceScenario(name="async-mock-response", fixture=_mock_fixture, asynchronous=True),
        TraceScenario(
            name="async-anthropic-provider-error",
            fixture=_provider_error_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-stream",
            fixture=_stream_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-azure-ai-stream",
            fixture=_azure_stream_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-bedrock-event-stream-error",
            fixture=_bedrock_stream_error_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-unsupported",
            fixture=_sync_unsupported_fixture,
            asynchronous=False,
        ),
    ),
)
