import pytest

from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse


@pytest.mark.parametrize(
    ("body", "matches"),
    (
        (b'{"challenge": "abc"}', True),
        (b'{"challenge": "\\u00e9"}', False),
        (b'{"challenge": 7}', False),
        (b"nope", False),
    ),
)
def test_only_an_exact_echo_of_the_challenge_verifies_the_endpoint(body: bytes, matches: bool) -> None:
    from litellm.proxy.mcp_server.events.webhooks.verification import echoed_challenge

    assert echoed_challenge(CallbackResponse(status=200, body=body), "abc") is matches
