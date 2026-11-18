from __future__ import annotations

from typing import Final

from ...fixtures import (
    anthropic_response_body,
    anthropic_stream_events,
    json_response,
    responses_body,
    responses_stream_events,
    sse_response,
)
from ...models import RouteFixture, RouteSpec, TraceScenario, TraceSuite


def _native_fixture(provider: str) -> RouteFixture:
    model: Final = "gpt-5"
    return RouteFixture(
        kwargs={
            "model": f"{provider}/{model}",
            "input": "hello",
        },
        provider_responses=(json_response(responses_body(model=model)),),
    )


def _openai_fixture(_base_url: str) -> RouteFixture:
    return _native_fixture("openai")


def _azure_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _native_fixture("azure")
    return fixture.derive(kwargs={"api_version": "2025-04-01-preview"})


def _openai_stream_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(_base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(responses_stream_events()),),
        consume_stream=True,
    )


def _provider_error_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(_base_url)
    return fixture.derive(
        provider_responses=(
            json_response({"error": {"message": "bad request", "type": "invalid_request_error"}}, status=400),
        ),
        expected_failure=True,
    )


def _stream_failed_fixture(base_url: str) -> RouteFixture:
    fixture: Final = _openai_fixture(base_url)
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


def _anthropic_bridge_fixture(_base_url: str) -> RouteFixture:
    return RouteFixture(
        kwargs={
            "model": "anthropic/claude-sonnet-5",
            "input": "hello",
            "max_output_tokens": 16,
        },
        provider_responses=(json_response(anthropic_response_body()),),
    )


def _anthropic_bridge_stream_fixture(_base_url: str) -> RouteFixture:
    fixture: Final = _anthropic_bridge_fixture(_base_url)
    return fixture.derive(
        kwargs={"stream": True},
        provider_responses=(sse_response(anthropic_stream_events()),),
        consume_stream=True,
    )


SPEC: Final = RouteSpec("responses", ("responses", "aresponses"), _openai_fixture)
TRACE_SUITE: Final = TraceSuite(
    route=SPEC,
    scenarios=(
        TraceScenario(name="sync-openai", fixture=_openai_fixture, asynchronous=False),
        TraceScenario(name="async-openai", fixture=_openai_fixture, asynchronous=True),
        TraceScenario(
            name="sync-openai-stream",
            fixture=_openai_stream_fixture,
            asynchronous=False,
        ),
        TraceScenario(
            name="async-openai-stream",
            fixture=_openai_stream_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-openai-provider-error",
            fixture=_provider_error_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-openai-stream-failed",
            fixture=_stream_failed_fixture,
            asynchronous=True,
        ),
        TraceScenario(name="async-azure", fixture=_azure_fixture, asynchronous=True),
        TraceScenario(
            name="async-anthropic-chat-bridge",
            fixture=_anthropic_bridge_fixture,
            asynchronous=True,
        ),
        TraceScenario(
            name="async-anthropic-chat-bridge-stream",
            fixture=_anthropic_bridge_stream_fixture,
            asynchronous=True,
        ),
    ),
)
