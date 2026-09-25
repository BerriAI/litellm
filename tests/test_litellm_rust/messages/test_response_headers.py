"""Upstream response headers reach the caller of a streamed Messages call before its first chunk.

The proxy forwards ``_hidden_params["additional_headers"]`` as ``llm_provider-*`` response headers and
has to send them before the SSE body starts, so the stream object must carry them at hand-off time.

The Python handler only serves async streams, so the sync case runs against the native route alone.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL

pytestmark = pytest.mark.requires_rust_extension

UPSTREAM_HEADERS: Final = {"request-id": "req_upstream_123", "anthropic-ratelimit-requests-remaining": "41"}
STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS, headers=dict(UPSTREAM_HEADERS))


@pytest.fixture(params=(Rollout.PYTHON_ONLY, Rollout.RUST_REQUIRED), ids=("python", "rust"))
def rollout(request: pytest.FixtureRequest) -> Iterator[Rollout]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, request.param), *catalog.RULES)):
        yield request.param


def arguments(server: RecordingServer) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        "stream": True,
    }


def assert_served_by(server: RecordingServer, rollout: Rollout) -> None:
    assert len(server.requests) == 1
    served_by_python: Final = server.requests[0].headers.get("user-agent", "").startswith("litellm/")
    assert served_by_python is (rollout is Rollout.PYTHON_ONLY)


def assert_upstream_headers_surfaced(stream: object) -> None:
    additional: Final = get_hidden_params_dict(stream).get("additional_headers")
    assert isinstance(additional, dict)
    expected: Final = process_response_headers(dict(UPSTREAM_HEADERS))
    assert expected
    assert {name: additional.get(name) for name in expected} == expected


@pytest.mark.asyncio
async def test_streamed_upstream_headers_are_on_the_stream_before_the_first_chunk(
    recording_server: RecordingServer, rollout: Rollout
) -> None:
    recording_server.enqueue(STREAM)

    stream: Final = await litellm.anthropic.messages.acreate(**arguments(recording_server))
    assert isinstance(stream, AsyncIterator)

    assert_served_by(recording_server, rollout)
    assert_upstream_headers_surfaced(stream)
    assert b"".join([chunk async for chunk in stream]) == b"".join(STREAM.payloads())


def test_sync_streamed_upstream_headers_are_on_the_stream_before_the_first_chunk(
    recording_server: RecordingServer,
) -> None:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED), *catalog.RULES)):
        recording_server.enqueue(STREAM)

        stream: Final = litellm.anthropic.messages.create(**arguments(recording_server))
    assert isinstance(stream, Iterator)

    assert_served_by(recording_server, Rollout.RUST_REQUIRED)
    assert_upstream_headers_surfaced(stream)
    assert b"".join(stream) == b"".join(STREAM.payloads())
