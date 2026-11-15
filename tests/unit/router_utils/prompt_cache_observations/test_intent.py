import pytest

from litellm.router_utils.prompt_cache_observations.intent import cache_intent
from litellm.router_utils.prompt_cache_observations.types import CacheIntent
from litellm.types.integrations.anthropic_cache_control_hook import (
    GATEWAY_INJECTED_CACHE_METADATA_KEY,
    GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT,
)

_MARKED_CHAT = {
    "messages": [{"role": "user", "content": [{"type": "text", "text": "doc", "cache_control": {"type": "ephemeral"}}]}]
}
_PLAIN_CHAT = {"messages": [{"role": "user", "content": "doc"}]}


def _params(body: object = None, injected_for: str | None = None) -> dict[str, object]:
    return {
        "proxy_server_request": {"body": body} if body is not None else None,
        "metadata": {GATEWAY_INJECTED_CACHE_METADATA_KEY: injected_for} if injected_for is not None else {},
    }


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (_params(_MARKED_CHAT), CacheIntent.CLIENT),
        (
            _params(
                {"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}], "messages": []}
            ),
            CacheIntent.CLIENT,
        ),
        (_params({**_PLAIN_CHAT, "cache_control": {"type": "ephemeral"}}), CacheIntent.CLIENT),
        (
            _params(
                {**_PLAIN_CHAT, "tools": [{"name": "t", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]}
            ),
            CacheIntent.CLIENT,
        ),
        (_params(_PLAIN_CHAT, injected_for="deployment-1"), CacheIntent.GATEWAY),
        (_params(_PLAIN_CHAT, injected_for=GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT), CacheIntent.GATEWAY),
        (_params(_MARKED_CHAT, injected_for="deployment-1"), CacheIntent.CLIENT_AND_GATEWAY),
        (_params(_PLAIN_CHAT, injected_for="deployment-on-an-earlier-attempt"), None),
        (_params(_PLAIN_CHAT), None),
        (_params(), None),
    ],
)
def test_intent_is_read_from_the_client_body_and_this_attempts_gateway_injection(
    params: dict[str, object], expected: CacheIntent | None
) -> None:
    assert cache_intent(params, "deployment-1") == expected


def test_gateway_marker_is_read_from_litellm_metadata_too() -> None:
    params = {"litellm_metadata": {GATEWAY_INJECTED_CACHE_METADATA_KEY: "deployment-1"}}
    assert cache_intent(params, "deployment-1") == CacheIntent.GATEWAY
