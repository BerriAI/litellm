"""The catalog target keeps bridged Messages calls on Python under RUST_REQUIRED rules.

A vertex_ai/gemini /v1/messages call reaches its provider through the chat-completions
bridge, so its target is CHAT_COMPLETIONS and no Messages rule can admit it to Rust.
"""

from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture(autouse=True)
def require_rust_for_vertex_messages() -> Iterator[None]:
    with rebound(
        catalog,
        "RULES",
        (RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED, providers=frozenset({"vertex_ai"})), *catalog.RULES),
    ):
        yield


@pytest.mark.asyncio
async def test_vertex_gemini_is_served_by_python(recording_server: RecordingServer) -> None:
    recording_server.expected_requests = 0

    response: Final = await litellm.anthropic.messages.acreate(
        model="vertex_ai/gemini-3.8-flash",
        messages=[dict(message) for message in MESSAGES],
        max_tokens=16,
        mock_response="ok",
    )

    assert recording_server.requests == []
    assert "x-litellm-rust" not in get_hidden_params_dict(response).get("additional_headers", {})


@pytest.mark.asyncio
async def test_vertex_claude_is_served_natively(recording_server: RecordingServer) -> None:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)

    response: Final = await litellm.anthropic.messages.acreate(
        model="vertex_ai/claude-sonnet-5-5",
        messages=[dict(message) for message in MESSAGES],
        max_tokens=16,
        api_base=recording_server.base_url,
        api_key="test-token",
        vertex_project="test-project",
        vertex_location="us-central1",
    )

    assert len(recording_server.requests) == 1
    assert get_hidden_params_dict(response)["additional_headers"]["x-litellm-rust"] == "true"
