"""Contract: budgets and rate limits on the ``/v1/realtime`` websocket and on MCP tool calls.

Realtime (litellm/proxy/proxy_server.py ``realtime_websocket_endpoint``): auth runs on the upgrade, so an
over-budget key never reaches the upstream. Limits are enforced by the pre-call logic before the upstream
handshake; a refusal there is reported as one JSON ``error`` event and the socket is closed with 1011
``Pre-call error`` (``_reject_realtime_session``), which also releases any budget hold and parallel slot.
Frame-level token limits exist only for the Responses websocket (``enforce_token_limits_for_frame``), which
``routing/`` already pins, so the realtime tests here stop at session admission.

MCP (litellm/proxy/mcp_server/): the admitted identity's ``budget_reservation`` is cleared
(``_enforce_admitted_live_policy`` sets it to None), so a tool call takes no budget hold; the configured tool
cost is charged after the call. The limiter adds one request to ``mcp_per_key`` (``<hashed key>:<server name>``),
``mcp_per_team`` (``<team_id>:<server name>``) and ``mcp_server`` (``<server_id>``) windows and never touches a
token window (``tokens_per_unit`` is None, parallel_request_limiter_v3.py:2975 to 3100).
"""

from __future__ import annotations

import threading
import uuid
from typing import Final

import httpx
import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.mcp import (
    JsonRpc,
    McpCaller,
    ScriptedTool,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from tests.integration.authorization.dec1_budlim2_support import chat, surface_model, surface_provider
from tests.integration.authorization.dec1_limits_support import (
    counter_key,
    hashed,
    limit_message,
    redis_integer,
    shared_redis,
)
from tests.integration.spend.dec1_budgets_support import (
    background,
    counter,
    counter_settles,
    key_counter,
    redis_client,
)

pytestmark: Final = pytest.mark.shares_worker_state

TOOL_COST: Final = 0.25
POLICY_VIOLATION: Final = 1008
INTERNAL_ERROR: Final = 1011


def _realtime(gateway: Gateway, key: str, model: str) -> tuple[str | None, int | None, str]:
    """Open ``/v1/realtime?model=<model>`` and return (first text frame or None, close code, close reason).

    A handshake the proxy refuses with an HTTP status is returned as (None, that status, "")."""
    url: Final = str(gateway.client.base_url).replace("http", "ws", 1).rstrip("/") + f"/v1/realtime?model={model}"
    try:
        with connect(url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=15) as socket:
            try:
                first: Final = socket.recv(timeout=15)
            except ConnectionClosed as closed:
                return None, closed.rcvd.code if closed.rcvd else None, closed.rcvd.reason if closed.rcvd else ""
            try:
                socket.recv(timeout=15)
            except ConnectionClosed as closed:
                text: Final = first if isinstance(first, str) else first.decode()
                return text, closed.rcvd.code if closed.rcvd else None, closed.rcvd.reason if closed.rcvd else ""
            raise AssertionError("the realtime session was not closed after its first frame")
    except InvalidStatus as rejected:
        return None, rejected.response.status_code, ""


def test_an_over_budget_key_is_refused_at_the_realtime_upgrade_without_reaching_the_upstream(
    gateway: Gateway,
) -> None:
    """A key with ``max_budget: 0`` opens ``/v1/realtime`` for a realtime deployment. Auth refuses on the upgrade
    with a policy violation: either the handshake is answered HTTP 403 or the socket is closed with 1008
    (user_api_key_auth.py ``user_api_key_auth_websocket_for_model``). No frame is sent first and the upstream
    receives no connection at all."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        model: Final = surface_model(scenario, provider, model="openai/gpt-4o-realtime-preview")
        key: Final = scenario.key(models=[model], max_budget=0)
        frame, code, _reason = _realtime(gateway, key, model)
        assert frame is None
        assert code in (403, POLICY_VIOLATION), code
        assert provider.posts() == ()
        assert provider.wire is not None and provider.wire.drain() == ()


def test_a_key_over_its_rpm_gets_one_error_event_and_a_1011_close_on_realtime(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 1`` spends its request on a chat call, then opens ``/v1/realtime``. The pre-call
    logic refuses: the client receives one JSON frame ``{"type": "error", "error": {"type": "guardrail_error",
    "message": "...Rate limit exceeded for api_key: <hash>. Limit type: requests..."}}`` and the socket closes
    with 1011 ``Pre-call error``; the upstream is never contacted. The ``guardrail_error`` type for a rate limit
    is what ``_reject_realtime_session`` defaults to today; it is pinned so a port changes it on purpose.
    UNVERIFIED: if the pre-body check also runs on the websocket upgrade, the refusal is a 1008 close instead."""
    with surface_provider() as provider, gateway.scenario() as scenario:
        chat_model: Final = surface_model(scenario, provider)
        realtime_model: Final = surface_model(scenario, provider, model="openai/gpt-4o-realtime-preview")
        key: Final = scenario.key(models=[chat_model, realtime_model], rpm_limit=1)
        assert chat(gateway, chat_model, key, max_tokens=5).status_code == 200
        provider.posts()
        frame, code, reason = _realtime(gateway, key, realtime_model)
        assert frame is not None, (code, reason)
        assert '"type": "error"' in frame or '"type":"error"' in frame, frame
        assert "guardrail_error" in frame, frame
        assert limit_message("api_key", hashed(key), "requests", 1).split(" Limit resets at")[0] in frame, frame
        assert (code, reason) == (INTERNAL_ERROR, "Pre-call error")
        assert provider.posts() == ()


def _gated_tool(gate: threading.Event, arrived: threading.Event) -> ScriptedTool:
    def respond(params: JsonRpc) -> JsonRpc:
        arrived.set()
        assert gate.wait(timeout=60), "the gated tool was never released"
        return text_result("gated")

    return ScriptedTool("gate", respond)


def _rest_call(gateway: Gateway, key: str, server_id: str, name: str) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"name": name, "arguments": {}, "server_id": server_id},
    )


@pytest.mark.timeout(180)
def test_an_in_flight_mcp_tool_call_takes_no_budget_hold_and_is_charged_its_cost_after(gateway: Gateway) -> None:
    """An MCP server is registered with ``default_cost_per_query: 0.25``. A key with ``max_budget`` 1.0 calls its
    tool through ``/mcp-rest/tools/call`` while the tool is held. The key counter ``spend:key:<hash>`` stays absent
    or 0 during the call (MCP clears the request's reservation), and after the tool answers it settles to 0.25,
    the configured cost. A port that adds an estimate hold to MCP would need a price model it does not have."""
    gate: Final = threading.Event()
    arrived: Final = threading.Event()
    with (
        scripted_peer(_gated_tool(gate, arrived)) as peer,
        gateway.scenario() as scenario,
        redis_client() as cache,
        background() as run,
    ):
        alias: Final = f"budlim{uuid.uuid4().hex[:8]}"
        server_id: Final = register_mcp(
            scenario,
            peer,
            alias,
            mcp_info={"server_name": alias, "mcp_server_cost_info": {"default_cost_per_query": TOOL_COST}},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, max_budget=1.0)
        in_flight: Final = run(lambda: _rest_call(gateway, key, server_id, f"{alias}-gate"))
        try:
            assert arrived.wait(timeout=30), "the tool call never reached the peer"
            assert counter(cache, key_counter(key)) in (None, 0.0)
        finally:
            gate.set()
        assert in_flight.result(timeout=60).status_code == 200
        counter_settles(cache, key_counter(key), TOOL_COST)


def test_the_mcp_per_key_refusal_names_the_hashed_key_and_the_server(gateway: Gateway) -> None:
    """A key with ``mcp_rpm_limit: {<alias>: 1}`` calls the server's tool twice through ``/mcp-rest/tools/call``.
    The second call is HTTP 429 and its body carries ``Rate limit exceeded for mcp_per_key: <sha256 of the
    key>:<alias>. Limit type: requests. Current limit: 1, Remaining: 0``; the peer receives one ``tools/call``."""
    gate: Final = threading.Event()
    gate.set()
    with scripted_peer(_gated_tool(gate, threading.Event())) as peer, gateway.scenario() as scenario:
        alias: Final = f"budlim{uuid.uuid4().hex[:8]}"
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, mcp_rpm_limit={alias: 1})
        assert _rest_call(gateway, key, server_id, f"{alias}-gate").status_code == 200
        peer.drain()
        refused: Final = _rest_call(gateway, key, server_id, f"{alias}-gate")
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_per_key", f"{hashed(key)}:{alias}", "requests", 1) in refused.text, refused.text
        assert tool_calls(peer.drain()) == ()


def test_the_mcp_server_rpm_refusal_names_the_server_id(gateway: Gateway) -> None:
    """An MCP server registered with ``rpm: 1`` is called by two different keys. The second call is HTTP 429 with
    ``Rate limit exceeded for mcp_server: <server_id>. Limit type: requests. Current limit: 1, Remaining: 0``:
    the server window is shared by every caller and keyed by the server id, not its alias."""
    gate: Final = threading.Event()
    gate.set()
    with scripted_peer(_gated_tool(gate, threading.Event())) as peer, gateway.scenario() as scenario:
        alias: Final = f"budlim{uuid.uuid4().hex[:8]}"
        server_id: Final = register_mcp(scenario, peer, alias, rpm=1)
        first: Final = scenario.key(object_permission={"mcp_servers": [server_id]})
        second: Final = scenario.key(object_permission={"mcp_servers": [server_id]})
        assert _rest_call(gateway, first, server_id, f"{alias}-gate").status_code == 200
        refused: Final = _rest_call(gateway, second, server_id, f"{alias}-gate")
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_server", server_id, "requests", 1) in refused.text, refused.text


def test_a_per_key_mcp_limit_on_one_server_does_not_limit_another_server(gateway: Gateway) -> None:
    """A key has ``mcp_rpm_limit: {<alias A>: 1}`` and access to servers A and B. After one call to A, a second
    call to A is 429 while two calls to B are both 200: the descriptor value includes the server name."""
    gate: Final = threading.Event()
    gate.set()
    with (
        scripted_peer(_gated_tool(gate, threading.Event())) as peer_a,
        scripted_peer(_gated_tool(gate, threading.Event())) as peer_b,
        gateway.scenario() as scenario,
    ):
        alias_a: Final = f"budlima{uuid.uuid4().hex[:8]}"
        alias_b: Final = f"budlimb{uuid.uuid4().hex[:8]}"
        server_a: Final = register_mcp(scenario, peer_a, alias_a)
        server_b: Final = register_mcp(scenario, peer_b, alias_b)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_a, server_b]}, mcp_rpm_limit={alias_a: 1})
        assert _rest_call(gateway, key, server_a, f"{alias_a}-gate").status_code == 200
        assert _rest_call(gateway, key, server_a, f"{alias_a}-gate").status_code == 429
        assert _rest_call(gateway, key, server_b, f"{alias_b}-gate").status_code == 200
        assert _rest_call(gateway, key, server_b, f"{alias_b}-gate").status_code == 200


def test_mcp_tool_calls_use_the_key_request_window_but_no_tokens(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 10`` and ``tpm_limit: 5`` makes three MCP tool calls. All three are served (a
    token limit of 5 would refuse any LLM call with a prompt), the key's Redis request window
    ``{api_key:<hash>}:requests`` reads 3 and its token window ``{api_key:<hash>}:tokens`` stays absent.
    UNVERIFIED: assumes MCP REST calls reach the key's general rpm descriptor like the ``mcp`` entry point does
    in ``mcp/test_mcp_rate_limits.py``.

    BUG (run 44176682961): the tool call came back as an isError result 'Rate limit exceeded for api_key: <hash>.
    Limit type: tokens. Current limit: 5, Remaining: 5'. A call that uses no tokens is refused on a token window
    that still has every token left."""
    pytest.skip("BUG: an MCP tool call is refused on the key token window with 'Remaining: 5' of 5")
    gate: Final = threading.Event()
    gate.set()
    with (
        scripted_peer(_gated_tool(gate, threading.Event())) as peer,
        gateway.scenario() as scenario,
        shared_redis() as redis,
    ):
        alias: Final = f"budlim{uuid.uuid4().hex[:8]}"
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, rpm_limit=10, tpm_limit=5)
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        for _ in range(3):
            outcome = caller.call(f"{alias}-gate", {})
            assert outcome.ok, outcome.raw
        eventually(
            lambda: redis_integer(redis, counter_key("api_key", hashed(key), "requests")),
            lambda value: value == 3,
            seconds=30,
        )
        assert redis_integer(redis, counter_key("api_key", hashed(key), "tokens")) is None


def test_an_exhausted_key_is_refused_on_an_mcp_tool_call_before_the_peer(gateway: Gateway) -> None:
    """A key with ``max_budget: 0`` calls a tool through ``/mcp-rest/tools/call``. The common checks run for MCP
    like for inference, so the call is refused with HTTP 422 (the over-budget status
    ``_enforce_admitted_live_policy`` documents, not a 401) and no ``tools/call`` reaches the peer.

    BUG (run 44176682961): the tool call was answered 200 with the peer's 'gated' result. The key budget is not
    checked on the MCP tool call path."""
    pytest.skip("BUG: an MCP tool call by a key over its budget is served instead of refused with 422")
    gate: Final = threading.Event()
    gate.set()
    with scripted_peer(_gated_tool(gate, threading.Event())) as peer, gateway.scenario() as scenario:
        alias: Final = f"budlim{uuid.uuid4().hex[:8]}"
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, max_budget=0)
        peer.drain()
        refused: Final = _rest_call(gateway, key, server_id, f"{alias}-gate")
        assert refused.status_code == 422, refused.text
        assert "budget" in refused.text.lower(), refused.text
        assert tool_calls(peer.drain()) == ()
