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


def _anthropic_fixture(_base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _bedrock_fixture(_base_url: str) -> RouteFixture:
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
            **credentials,
            "max_tokens": 16,
        },
        provider_responses=(json_response(response),),
    )


def _anthropic_stream_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(_base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


def _bedrock_stream_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _bedrock_fixture(_base_url)
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


def _provider_error_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(_base_url)
    return fixture.derive(
        provider_responses=(
            json_response(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}},
                status=400,
            ),
        ),
        expected_failure=True,
    )


def _stream_error_fixture(base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_fixture(base_url)
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
    _anthropic_fixture,
)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(
            name="sync-anthropic",
            fixture=_anthropic_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-anthropic",
            fixture=_anthropic_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-anthropic-stream",
            fixture=_anthropic_stream_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-anthropic-stream",
            fixture=_anthropic_stream_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-provider-error",
            fixture=_provider_error_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-stream-error",
            fixture=_stream_error_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-bedrock",
            fixture=_bedrock_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-bedrock",
            fixture=_bedrock_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="sync-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-bedrock-event-stream",
            fixture=_bedrock_stream_fixture,
            asynchronous=True,
        ),
    ),
)
