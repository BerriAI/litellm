"""
Characterization of a router core's Anthropic safeguard-refusal fallbacks on /v1/messages
(non-streaming): a ``stop_reason: "refusal"`` response carrying ``stop_details`` re-enters the
fallback chain only when a fallback would resolve for it.

The core recognises a /v1/messages call by its endpoint's name, so this module runs on
``NamedEndpointHost`` (a ``ScriptedHost`` whose ``anthropic_messages`` endpoint is named so).
"""

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.fallbacks_helpers import (
    MESSAGES,
    MakeCore,
    NamedEndpointHost,
    anthropic_message,
    anthropic_refusal,
    build,
)


@pytest.fixture
def host() -> NamedEndpointHost:
    return NamedEndpointHost()


async def messages(core: RouterCore, **request: object) -> object:  # kwargs-ok: request kwargs
    return await core.aanthropic_messages(model="a", messages=MESSAGES, max_tokens=10, **request)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"content_policy_fallbacks": [{"a": ["b"]}]}, ["a", "b"]),
        ({"fallbacks": [{"a": ["b"]}]}, ["a", "b"]),
        ({"default_fallbacks": ["c"]}, ["a", "c"]),
    ],
    ids=["content-policy-chain", "general-chain", "default-fallbacks"],
)
async def test_a_safeguard_refusal_falls_back_when_a_chain_resolves(
    make_core: MakeCore, host: NamedEndpointHost, arguments: dict[str, object], expected: list[str]
) -> None:
    core = build(make_core, **arguments)
    host.answer("openai/a", anthropic_refusal())
    host.answer(f"openai/{expected[-1]}", anthropic_message())

    response = await messages(core)

    assert host.deployments_called("anthropic_messages") == expected
    assert isinstance(response, dict)
    assert response["stop_reason"] == "end_turn"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "request_kwargs", "refusal"),
    [
        ({}, {}, anthropic_refusal()),
        ({"content_policy_fallbacks": [{"other": ["b"]}], "fallbacks": [{"a": ["b"]}]}, {}, anthropic_refusal()),
        ({"fallbacks": [{"a": ["b"]}]}, {"disable_fallbacks": True}, anthropic_refusal()),
        ({"fallbacks": [{"a": ["b"]}]}, {}, anthropic_refusal(with_stop_details=False)),
    ],
    ids=[
        "no-chain",
        "unmatched-content-policy-chain-is-authoritative",
        "fallbacks-disabled",
        "plain-refusal-without-stop-details",
    ],
)
async def test_a_refusal_is_returned_to_the_caller_when_no_fallback_would_resolve(
    make_core: MakeCore,
    host: NamedEndpointHost,
    arguments: dict[str, object],
    request_kwargs: dict[str, object],
    refusal: dict[str, object],
) -> None:
    core = build(make_core, **arguments)
    host.answer("openai/a", refusal)

    response = await messages(core, **request_kwargs)

    assert host.deployments_called("anthropic_messages") == ["a"]
    assert isinstance(response, dict)
    assert response["stop_reason"] == "refusal"
    assert response["_hidden_params"]["additional_headers"]["x-litellm-attempted-fallbacks"] == 0


@pytest.mark.asyncio
async def test_the_last_refusal_is_returned_when_every_fallback_refuses(
    make_core: MakeCore, host: NamedEndpointHost
) -> None:
    core = build(make_core, fallbacks=[{"a": ["b"]}])
    host.answer("*", anthropic_refusal())

    response = await messages(core)

    assert host.deployments_called("anthropic_messages") == ["a", "b"]
    assert isinstance(response, dict)
    assert response["stop_reason"] == "refusal"
