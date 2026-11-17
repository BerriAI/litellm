"""Model capacity (`dynamic_rate_limiter_v3`) on provider pass-through routes.

A model group's `rpm` (summed over its deployments) is one window every caller shares, enforced by the dynamic
limiter's pre-call hook against `registry().group(data["model"])`. Pass-through reaches that hook two ways. A
Bedrock call to `/bedrock/model/<group>/<action>` runs the common request path with the group as its model
(`allm_passthrough_route`), so the group's window must hold it exactly as it holds `/v1/chat/completions`. An
`/anthropic/*` call runs the pass-through pre-call hook with the caller's untouched body, whose `model` is
whatever the caller wrote; when that name is a router group the same window applies, and when it is not, there is
no group and no capacity to enforce.

The refusal is 429 with `Model capacity reached for <group>. Priority: None, Rate limit type: requests, Model
TPM: not configured, Model RPM: <n>, Remaining: 0` (`dynamic_rate_limiter_v3.py:1309`). The capacity limiter is
being replaced by a Rust callback (#51540), so these are the pass-through cases the sibling capacity agent left
out.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_passthru_support import (
    BEDROCK_CONVERSE_BODY,
    anthropic_body,
    anthropic_error,
    bearer,
    bedrock_deployment,
    bedrock_peer,
    marker,
    message_of,
    openai_error,
    posts,
    provider_environment,
    scripted_provider,
    send,
)
from integration.routing.dec1_capacity_support import capacity_message, capacity_proxy

pytestmark = pytest.mark.timeout(300)

_ANTHROPIC_GROUP: Final = f"dec1-passthru-cap-claude-{uuid.uuid4().hex[:8]}"
_BEDROCK_GROUP: Final = f"dec1-passthru-cap-bedrock-{uuid.uuid4().hex[:8]}"
_WIDE_BEDROCK_GROUP: Final = f"dec1-passthru-cap-wide-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def providers() -> Iterator[Wire]:
    with wire_server(scripted_provider) as served:
        yield served


@pytest.fixture(scope="module")
def bedrock() -> Iterator[Wire]:
    with wire_server(bedrock_peer) as served:
        yield served


@pytest.fixture(scope="module")
def proxy(providers: Wire, bedrock: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """One proxy with the dynamic limiter on and three groups: an Anthropic group with `rpm: 1`, a Bedrock group
    with `rpm: 1` and a Bedrock group with `rpm: 2`."""
    model_list: Final = [
        {
            "model_name": _ANTHROPIC_GROUP,
            "litellm_params": {"model": f"anthropic/{_ANTHROPIC_GROUP}", "api_key": "dec1-cap", "rpm": 1},
            "model_info": {"id": f"{_ANTHROPIC_GROUP}-0"},
        },
        {
            "model_name": _BEDROCK_GROUP,
            "litellm_params": bedrock_deployment(bedrock, rpm=1),
            "model_info": {"id": f"{_BEDROCK_GROUP}-0"},
        },
        {
            "model_name": _WIDE_BEDROCK_GROUP,
            "litellm_params": bedrock_deployment(bedrock, rpm=2),
            "model_info": {"id": f"{_WIDE_BEDROCK_GROUP}-0"},
        },
    ]
    with (
        gateway_from_environment() as shared,
        capacity_proxy(
            shared,
            tmp_path_factory.mktemp("dec1-passthru-capacity"),
            model_list=model_list,
            environment=provider_environment(providers),
        ) as candidate,
    ):
        yield candidate


@pytest.fixture(autouse=True)
def _drained(providers: Wire, bedrock: Wire) -> None:
    providers.drain()
    bedrock.drain()


def _converse(proxy: Gateway, group: str):  # noqa: ANN202
    return proxy.request("POST", f"/bedrock/model/{group}/converse", BEDROCK_CONVERSE_BODY)


def test_a_bedrock_router_group_at_capacity_refuses_the_next_pass_through_call(proxy: Gateway, bedrock: Wire) -> None:
    """The Bedrock group has `rpm: 1`. Two `/bedrock/model/<group>/converse` calls with the master key: the first is
    served and the second must be 429 with exactly `Model capacity reached for <group>. Priority: None, Rate limit
    type: requests, Model TPM: not configured, Model RPM: 1, Remaining: 0`, and the deployment must have received
    one call."""
    first: Final = _converse(proxy, _BEDROCK_GROUP)
    assert first.status_code == 200, first.text
    refused: Final = _converse(proxy, _BEDROCK_GROUP)
    assert message_of(openai_error(refused, 429)) == capacity_message(_BEDROCK_GROUP, None, "requests", rpm=1)
    assert len(posts(bedrock)) == 1


def test_a_served_bedrock_pass_through_reports_the_groups_capacity_headers(proxy: Gateway) -> None:
    """The second Bedrock group has `rpm: 2`. Its first Converse call must answer 200 with
    `x-ratelimit-model_saturation_check-limit-requests: 2` and `...-remaining-requests: 1`, as a chat call on the
    same group does. UNVERIFIED: that the router pass-through response copies the limiter's headers."""
    response: Final = _converse(proxy, _WIDE_BEDROCK_GROUP)
    assert response.status_code == 200, response.text
    assert response.headers.get("x-ratelimit-model_saturation_check-limit-requests") == "2", dict(response.headers)
    assert response.headers.get("x-ratelimit-model_saturation_check-remaining-requests") == "1", dict(response.headers)


def test_an_anthropic_pass_through_body_naming_a_group_at_capacity_is_refused(proxy: Gateway, providers: Wire) -> None:
    """The Anthropic group `dec1-passthru-cap-claude-<hex>` has `rpm: 1`. Two `/anthropic/v1/messages` calls whose
    body `model` is that group name: the first must be forwarded (to `ANTHROPIC_API_BASE`, with the body unchanged)
    and the second refused with 429 in the Anthropic envelope (`rate_limit_error`) carrying the capacity message.
    UNVERIFIED: assumes the pass-through pre-call hook hands the dynamic limiter the parsed body, so
    `data["model"]` is the group name."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        body: Final = anthropic_body(marker(), model=_ANTHROPIC_GROUP)
        first: Final = send(proxy, "/anthropic/v1/messages", body, headers=bearer(key))
        assert first.status_code == 200, first.text
        refused: Final = send(
            proxy, "/anthropic/v1/messages", anthropic_body(marker(), model=_ANTHROPIC_GROUP), headers=bearer(key)
        )
        error: Final = anthropic_error(refused, 429)
        assert error["type"] == "rate_limit_error", refused.text
        assert message_of(error) == capacity_message(_ANTHROPIC_GROUP, None, "requests", rpm=1), refused.text
        assert len(posts(providers)) == 1


def test_an_anthropic_pass_through_body_naming_no_group_is_not_held_by_capacity(
    proxy: Gateway, providers: Wire
) -> None:
    """Three `/anthropic/v1/messages` calls name `claude-sonnet-4-5`, which no group on this proxy serves. All three
    must be served: with no group there is no model capacity to enforce, so the dynamic limiter lets the call
    through (`registry().group(...)` returns None)."""
    with proxy.scenario() as scenario:
        key: Final = scenario.key()
        statuses: Final = [
            send(proxy, "/anthropic/v1/messages", anthropic_body(marker()), headers=bearer(key)).status_code
            for _ in range(3)
        ]
        assert statuses == [200, 200, 200]
        assert len(posts(providers)) == 3
