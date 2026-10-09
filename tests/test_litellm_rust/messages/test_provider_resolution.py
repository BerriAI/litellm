"""Routing and execution read the same resolver, so what admits a call is what serves it."""

from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

BARE_MODEL: Final = "claude-sonnet-5"
PREFIXED_MODEL: Final = f"anthropic/{BARE_MODEL}"
OTHER_MODEL: Final = "claude-haiku-4-5"


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": PREFIXED_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def served_model(server: RecordingServer) -> str:
    assert len(server.requests) == 1
    request: Final = server.requests[0]
    assert not request.headers.get("user-agent", "").startswith("python-httpx")
    assert isinstance(request.body, dict)
    model: Final = request.body["model"]
    assert isinstance(model, str)
    return model


class Rewrite(CustomLogger):
    def __init__(self, **changes: object) -> None:  # kwargs-ok: the hook replays arbitrary kwarg edits
        super().__init__()
        self.changes: Final = changes

    async def async_pre_call_deployment_hook(self, kwargs: dict[str, object], call_type: object) -> dict[str, object]:
        return {**kwargs, **self.changes}


@pytest.mark.asyncio
async def test_bare_catalog_model_is_served_natively_under_its_resolved_provider(
    messages_server: RecordingServer,
) -> None:
    assert BARE_MODEL in litellm.anthropic_models

    await litellm.anthropic.messages.acreate(**arguments(messages_server, model=BARE_MODEL))

    assert served_model(messages_server) == BARE_MODEL


@pytest.mark.asyncio
async def test_prefixed_model_reaches_the_wire_without_its_routing_prefix(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert served_model(messages_server) == BARE_MODEL


@pytest.mark.asyncio
async def test_model_rewritten_by_a_pre_call_hook_is_what_gets_served(messages_server: RecordingServer) -> None:
    litellm.callbacks.append(Rewrite(model=f"anthropic/{OTHER_MODEL}"))

    await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert served_model(messages_server) == OTHER_MODEL


@pytest.mark.asyncio
async def test_provider_rewritten_to_one_rust_cannot_serve_fails_under_that_provider(
    messages_server: RecordingServer,
) -> None:
    litellm.callbacks.append(Rewrite(model="gpt-4o", custom_llm_provider="openai"))
    messages_server.expected_requests = 0

    with pytest.raises(litellm.APIConnectionError, match=r"OpenAIException - invalid provider: openai"):
        await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert messages_server.requests == []
