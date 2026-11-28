"""Rate limits on agents and agent sessions.

The contract, read from ``_create_agent_rate_limit_descriptors`` and ``_get_resolved_agent_id`` in
``litellm/proxy/hooks/parallel_request_limiter_v3.py``:

- An agent registered with ``rpm_limit`` / ``tpm_limit`` (``POST /v1/agents``) caps every request attributed to
  it, across keys and sessions: descriptor ``agent``, value the agent id.
- An agent with ``session_rpm_limit`` / ``session_tpm_limit`` also caps each session on its own: descriptor
  ``agent_session``, value ``<agent_id>:<session_id>``. The session id is the first of the top-level
  ``litellm_session_id``, ``metadata.session_id`` and ``litellm_metadata.session_id``. A request without a session
  id gets no session descriptor.
- The agent is the key's own ``agent_id`` when the key has one, else ``metadata.agent_id`` (or
  ``litellm_metadata.agent_id``) on the request.

The Rust port (#51537) builds these descriptors in ``builder.rs`` but leaves the agent registry lookup in Python,
so the registry-to-descriptor hand-off is new code. Agents are looked up in the in-memory registry; UNVERIFIED that
a freshly created agent is visible to the request path at once, so first requests poll.
"""

from __future__ import annotations

import uuid
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    posts,
    provider,
    scope_headers,
    statuses,
)
from tests.integration.authorization.dec1_ratelim_support import agent, chat_with


def _first_served(gateway: Gateway, model: str, key: str, agent_id: str) -> None:
    """Poll until the first request attributed to a new agent is served and reports the agent window. Requests
    made before the registry knows the agent carry no agent descriptor, so they do not use the agent window."""
    eventually(
        lambda: chat_with(gateway, model, key, metadata={"agent_id": agent_id}),
        lambda response: response.status_code == 200 and bool(scope_headers(response, "agent")),
    )


def test_an_agents_request_window_is_shared_across_keys(gateway: Gateway) -> None:
    """An agent has rpm_limit=1. Key one sends a request with ``metadata.agent_id``; key two then sends one for the
    same agent and must be refused with ``Rate limit exceeded for agent: <agent_id>. Limit type: requests. Current
    limit: 1, Remaining: 0``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, rpm_limit=1)
        first: Final = scenario.key(models=[model])
        second: Final = scenario.key(models=[model])
        _first_served(gateway, model, first, identity)
        assert_limited(
            chat_with(gateway, model, second, metadata={"agent_id": identity}), "agent", identity, "requests", 1
        )


def test_a_served_agent_request_reports_the_agent_window(gateway: Gateway) -> None:
    """An agent has rpm_limit=5 and tpm_limit=1000. A request for it asking max_tokens=20 (estimate 30) must carry
    ``x-ratelimit-agent-limit-requests: 5``, ``-remaining-requests: 4``, ``-limit-tokens: 1000`` and
    ``-remaining-tokens: 970``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, rpm_limit=5, tpm_limit=1000)
        key: Final = scenario.key(models=[model])
        served: Final = eventually(
            lambda: chat_with(gateway, model, key, metadata={"agent_id": identity}, max_tokens=20),
            lambda response: bool(scope_headers(response, "agent")),
        )
        assert scope_headers(served, "agent") == {
            "limit-requests": "5",
            "remaining-requests": "4",
            "limit-tokens": "1000",
            "remaining-tokens": "970",
        }


def test_an_agents_tpm_limit_refuses_a_request_too_large_for_it(gateway: Gateway) -> None:
    """An agent has tpm_limit=20. A request for it asking max_tokens=50 (estimate 60) must be refused with
    ``Rate limit exceeded for agent: <agent_id>. Limit type: tokens. Current limit: 20, Remaining: 20`` before the
    provider is called."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, tpm_limit=20)
        key: Final = scenario.key(models=[model])
        refused: Final = eventually(
            lambda: chat_with(gateway, model, key, metadata={"agent_id": identity}, max_tokens=50),
            lambda response: response.status_code == 429,
        )
        assert_limited(refused, "agent", identity, "tokens", 20, 20)
        assert posts(wire) == ()


def test_each_session_of_an_agent_has_its_own_window(gateway: Gateway) -> None:
    """An agent has session_rpm_limit=1. Session A sends one request (``litellm_session_id``); its second must be
    refused with ``Rate limit exceeded for agent_session: <agent_id>:<A>``, while session B's first request is
    still served."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, session_rpm_limit=1)
        key: Final = scenario.key(models=[model])
        session_a: Final = f"session-{uuid.uuid4().hex}"
        session_b: Final = f"session-{uuid.uuid4().hex}"
        eventually(
            lambda: chat_with(gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=session_a),
            lambda response: bool(scope_headers(response, "agent_session")),
        )
        assert_limited(
            chat_with(gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=session_a),
            "agent_session",
            f"{identity}:{session_a}",
            "requests",
            1,
        )
        assert (
            chat_with(gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=session_b).status_code
            == 200
        )


def test_metadata_session_id_identifies_the_session_too(gateway: Gateway) -> None:
    """An agent has session_rpm_limit=1. Two requests carry the session only as ``metadata.session_id``; the
    second must be refused for ``agent_session: <agent_id>:<session>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, session_rpm_limit=1)
        key: Final = scenario.key(models=[model])
        session: Final = f"session-{uuid.uuid4().hex}"
        metadata: Final = {"agent_id": identity, "session_id": session}
        eventually(
            lambda: chat_with(gateway, model, key, metadata=metadata),
            lambda response: bool(scope_headers(response, "agent_session")),
        )
        assert_limited(
            chat_with(gateway, model, key, metadata=metadata), "agent_session", f"{identity}:{session}", "requests", 1
        )


def test_a_request_without_a_session_is_not_held_to_the_session_limit(gateway: Gateway) -> None:
    """An agent has only session_rpm_limit=1. Three requests for it without any session id must all be served and
    carry no ``x-ratelimit-agent_session-*`` header: no session id, no session descriptor."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, session_rpm_limit=1)
        key: Final = scenario.key(models=[model])
        responses: Final = [chat_with(gateway, model, key, metadata={"agent_id": identity}) for _ in range(3)]
        assert statuses(responses) == [200, 200, 200]
        assert scope_headers(responses[-1], "agent_session") == {}


def test_the_session_window_and_the_agent_window_are_both_charged(gateway: Gateway) -> None:
    """An agent has rpm_limit=2 and session_rpm_limit=5. Two requests in two different sessions use the agent
    window; a third in a fresh session must be refused for ``agent: <agent_id>`` although its session is empty."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, rpm_limit=2, session_rpm_limit=5)
        key: Final = scenario.key(models=[model])
        eventually(
            lambda: chat_with(
                gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=f"s-{uuid.uuid4().hex}"
            ),
            lambda response: bool(scope_headers(response, "agent")),
        )
        second: Final = chat_with(
            gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=f"s-{uuid.uuid4().hex}"
        )
        assert second.status_code == 200, second.text
        third: Final = chat_with(
            gateway, model, key, metadata={"agent_id": identity}, litellm_session_id=f"s-{uuid.uuid4().hex}"
        )
        assert_limited(third, "agent", identity, "requests", 2)


def test_a_keys_own_agent_wins_over_the_agent_named_in_metadata(gateway: Gateway) -> None:
    """Agent X (rpm_limit=1) is bound to a key through ``agent_id``; agent Y (rpm_limit=100) is named in the
    request's ``metadata.agent_id``. The key's second request must be refused for ``agent: <X>``: a key-level
    agent takes precedence over a caller-supplied one, so a caller cannot escape its agent's limit."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        bound: Final = agent(gateway, scenario, wire.url, rpm_limit=1)
        named: Final = agent(gateway, scenario, wire.url, rpm_limit=100)
        key: Final = scenario.key(models=[model], agent_id=bound)
        first: Final = chat_with(gateway, model, key, metadata={"agent_id": named})
        assert first.status_code == 200, first.text
        assert scope_headers(first, "agent")["limit-requests"] == "1"
        assert_limited(chat_with(gateway, model, key, metadata={"agent_id": named}), "agent", bound, "requests", 1)


def test_an_unknown_agent_id_in_metadata_adds_no_agent_limit(gateway: Gateway) -> None:
    """A request names ``metadata.agent_id`` of an agent that was never registered. Three such requests must be
    served without any ``x-ratelimit-agent-*`` header: an agent missing from the registry contributes nothing."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        responses: Final = [
            chat_with(gateway, model, key, metadata={"agent_id": f"missing-{uuid.uuid4().hex}"}) for _ in range(3)
        ]
        assert statuses(responses) == [200, 200, 200]
        assert scope_headers(responses[-1], "agent") == {}
        assert len(posts(wire)) == 3


def test_an_agent_limit_raised_through_patch_admits_it_again(gateway: Gateway) -> None:
    """An agent with rpm_limit=1 has been used and refused. After ``PATCH /v1/agents/<id>`` sets rpm_limit=100 a
    request for it must eventually be served in the same window. UNVERIFIED how quickly the in-memory registry
    sees the patch on every worker; this polls for 10 seconds."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        identity: Final = agent(gateway, scenario, wire.url, rpm_limit=1)
        key: Final = scenario.key(models=[model])
        eventually(
            lambda: chat_with(gateway, model, key, metadata={"agent_id": identity}).status_code,
            lambda status: status == 429,
        )
        patched: Final = gateway.request("PATCH", f"/v1/agents/{identity}", {"rpm_limit": 100})
        assert patched.status_code == 200, patched.text
        eventually(
            lambda: chat_with(gateway, model, key, metadata={"agent_id": identity}).status_code,
            lambda status: status == 200,
        )
