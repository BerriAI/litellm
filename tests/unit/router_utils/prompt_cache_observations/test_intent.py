import pytest

from litellm.router_utils.prompt_cache_observations.intent import cache_requested
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
        (_params(_MARKED_CHAT), True),
        (
            _params(
                {"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}], "messages": []}
            ),
            True,
        ),
        (_params({**_PLAIN_CHAT, "cache_control": {"type": "ephemeral"}}), True),
        (
            _params(
                {**_PLAIN_CHAT, "tools": [{"name": "t", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]}
            ),
            True,
        ),
        (_params(_PLAIN_CHAT, injected_for="deployment-1"), True),
        (_params(_PLAIN_CHAT, injected_for=GATEWAY_INJECTED_FOR_EVERY_DEPLOYMENT), True),
        (_params(_MARKED_CHAT, injected_for="deployment-1"), True),
        (_params(_PLAIN_CHAT, injected_for="deployment-on-an-earlier-attempt"), False),
        (_params(_PLAIN_CHAT), False),
        (_params(), False),
    ],
)
def test_caching_is_requested_by_the_client_body_or_this_attempts_gateway_injection(
    params: dict[str, object], expected: bool
) -> None:
    assert cache_requested(params, "deployment-1") is expected


def test_gateway_marker_is_read_from_litellm_metadata_too() -> None:
    params = {"litellm_metadata": {GATEWAY_INJECTED_CACHE_METADATA_KEY: "deployment-1"}}
    assert cache_requested(params, "deployment-1") is True
