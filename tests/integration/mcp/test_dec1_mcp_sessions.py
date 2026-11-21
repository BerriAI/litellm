"""Streamable HTTP session ids, stale and foreign sessions, and aggregate resource reads.

The contract on the Python gateway (`litellm/proxy/mcp_server/server.py`): a POST without
`mcp-session-id` that carries `initialize` goes to the stateful session manager and the response carries
a fresh `mcp-session-id`; any other request without a session id is served statelessly. A session id is
bound to the caller that created it (`_owner_fingerprint_for`): another caller presenting it gets HTTP 403
`{"error": "Forbidden", "details": "mcp-session-id is bound to a different caller."}`. A session id this
worker does not know is stripped from a POST so the request is still served, and a DELETE of an unknown
session answers 200 `{"message": "Session terminated successfully"}` (issue 20992).

The Rust host differs today (`gateway-mcp/src/sessions.rs`): an unknown id is HTTP 404
`{"detail": "MCP session not found"}` and a foreign one is 403 `{"detail": "MCP session belongs to another
caller"}`. These tests pin the Python side so the switch is a deliberate, visible change.

Session ownership is per worker memory, so the session tests use one module-scoped single-worker proxy.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.mcp import INITIALIZE, McpPeer, mcp_peer, register_mcp, tool_calls
from integration._support.process import owned_proxy

from tests.integration.mcp.dec1_mcp_support import (
    first_text,
    listed_names,
    open_session,
    rpc,
    rpc_error,
    rpc_result,
)

pytestmark: Final = pytest.mark.shares_worker_state

_FOREIGN: Final = {"error": "Forbidden", "details": "mcp-session-id is bound to a different caller."}
_DELETED: Final = {"message": "Session terminated successfully"}


@pytest.fixture(scope="module")
def single_worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """One proxy with one worker, so a session created by one request is visible to the next."""
    directory: Final = tmp_path_factory.mktemp("dec1-mcp-sessions")
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, workers=1) as owned:
        yield owned


def _granted(scenario: Scenario, peer: McpPeer, stem: str) -> tuple[str, str, str]:
    alias: Final = stem + uuid.uuid4().hex[:8]
    identity: Final = register_mcp(scenario, peer, alias)
    return alias, identity, scenario.key(object_permission={"mcp_servers": [identity]})


def test_initialize_without_a_session_id_returns_a_new_session_id(single_worker: Gateway) -> None:
    """A granted key POSTs `initialize` to `/mcp` with no `mcp-session-id`. The request is routed to the
    stateful manager, so the response must be HTTP 200 with a non-empty `mcp-session-id` header and a
    JSON-RPC result carrying `serverInfo`. Clients such as editors depend on getting that id."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        _, _, key = _granted(scenario, peer, "dec1sess")
        session, response = open_session(single_worker, "/mcp", key=key)
        assert len(session) >= 8, session
        assert "serverInfo" in rpc_result(response), response.text


def test_two_initializes_by_the_same_caller_get_different_session_ids(single_worker: Gateway) -> None:
    """The same key initializes twice. Each initialize opens its own session, so the two
    `mcp-session-id` values must differ; a host that keyed sessions by caller would hand back one id."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        _, _, key = _granted(scenario, peer, "dec1twice")
        first, _ = open_session(single_worker, "/mcp", key=key)
        second, _ = open_session(single_worker, "/mcp", key=key)
        assert first != second


def test_a_request_without_initialize_or_session_is_served_statelessly(single_worker: Gateway) -> None:
    """A granted key sends `tools/list` straight to `/mcp` with no initialize and no session id (the curl
    or Inspector shape). It must be HTTP 200 and list `<alias>-add`, and the response carries no
    `mcp-session-id` because nothing stateful was created."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        alias, _, key = _granted(scenario, peer, "dec1stateless")
        response: Final = rpc(single_worker, "/mcp", "tools/list", key=key)
        assert f"{alias}-add" in listed_names(response), response.text
        assert "mcp-session-id" not in response.headers, dict(response.headers)


def test_the_owner_can_call_a_tool_inside_its_session(single_worker: Gateway) -> None:
    """A key initializes, sends `notifications/initialized`, then calls `<alias>-add` with
    `{a: 2, b: 3}` on the same `mcp-session-id`. The call must return `5` and reach the peer once."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        alias, _, key = _granted(scenario, peer, "dec1owner")
        session, _ = open_session(single_worker, "/mcp", key=key)
        headers: Final = {"mcp-session-id": session}
        single_worker.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={"x-litellm-api-key": key, "Accept": "application/json, text/event-stream", **headers},
        )
        peer.drain()
        result: Final = rpc_result(
            rpc(single_worker, "/mcp", "tools/call", {"name": f"{alias}-add", "arguments": {"a": 2, "b": 3}}, key=key, headers=headers)
        )
        assert first_text(result) == "5", result
        assert len(tool_calls(peer.drain())) == 1


def test_another_key_presenting_a_live_session_id_is_refused_with_403(single_worker: Gateway) -> None:
    """Key A initializes and gets a session id. Key B, also granted the server, sends `tools/list` with A's
    `mcp-session-id`. The owner check runs before anything else, so B must get HTTP 403 with exactly
    `{"error": "Forbidden", "details": "mcp-session-id is bound to a different caller."}`."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        _, identity, owner = _granted(scenario, peer, "dec1foreign")
        intruder: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        session, _ = open_session(single_worker, "/mcp", key=owner)
        response: Final = rpc(single_worker, "/mcp", "tools/list", key=intruder, headers={"mcp-session-id": session})
        assert response.status_code == 403, response.text
        assert response.json() == _FOREIGN


def test_a_foreign_delete_cannot_terminate_the_owners_session(single_worker: Gateway) -> None:
    """Key B sends `DELETE /mcp` with key A's live session id. It must be refused with the same 403 body,
    and afterwards A can still call a tool on that session and get `5`. The owner check runs before the
    stale-session cleanup, so a non-owner can never close someone else's session."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        alias, identity, owner = _granted(scenario, peer, "dec1fdel")
        intruder: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        session, _ = open_session(single_worker, "/mcp", key=owner)
        deleted: Final = single_worker.client.delete(
            "/mcp", headers={"x-litellm-api-key": intruder, "mcp-session-id": session}
        )
        assert deleted.status_code == 403, deleted.text
        assert deleted.json() == _FOREIGN
        still: Final = rpc(
            single_worker,
            "/mcp",
            "tools/call",
            {"name": f"{alias}-add", "arguments": {"a": 2, "b": 3}},
            key=owner,
            headers={"mcp-session-id": session},
        )
        assert first_text(rpc_result(still)) == "5", still.text


def test_an_unknown_session_id_on_a_post_is_stripped_and_the_request_is_served(single_worker: Gateway) -> None:
    """A granted key sends `tools/list` with a made-up `mcp-session-id`. The worker does not know it, so
    `_handle_stale_mcp_session` strips the header and the request is served like a sessionless one: HTTP
    200 listing `<alias>-add`. The Rust host answers 404 here; a client that reconnects after a proxy
    restart relies on the Python behaviour."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        alias, _, key = _granted(scenario, peer, "dec1stale")
        response: Final = rpc(
            single_worker, "/mcp", "tools/list", key=key, headers={"mcp-session-id": uuid.uuid4().hex}
        )
        assert f"{alias}-add" in listed_names(response), response.text


def test_delete_of_an_unknown_session_is_an_idempotent_200(single_worker: Gateway) -> None:
    """`DELETE /mcp` with a made-up `mcp-session-id` must answer HTTP 200 with exactly
    `{"message": "Session terminated successfully"}` and never a 404, so client shutdown after a restart
    does not log an error."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        _, _, key = _granted(scenario, peer, "dec1unknowndel")
        response: Final = single_worker.client.delete(
            "/mcp", headers={"x-litellm-api-key": key, "mcp-session-id": uuid.uuid4().hex}
        )
        assert response.status_code == 200, response.text
        assert response.json() == _DELETED


def test_after_the_owner_deletes_its_session_the_id_is_treated_as_unknown(single_worker: Gateway) -> None:
    """The owner initializes, then `DELETE /mcp` with its session id (HTTP 200 from the SDK session
    manager). A later `tools/list` with that id is no longer bound: it is stripped as stale and served with
    HTTP 200, and a second DELETE gets the idempotent `Session terminated successfully` body.
    UNVERIFIED: the first DELETE's body comes from the MCP SDK and is not asserted."""
    with mcp_peer() as peer, single_worker.scenario() as scenario:
        alias, _, key = _granted(scenario, peer, "dec1ownerdel")
        session, _ = open_session(single_worker, "/mcp", key=key)
        first: Final = single_worker.client.delete("/mcp", headers={"x-litellm-api-key": key, "mcp-session-id": session})
        assert first.status_code in (200, 204), first.text
        after: Final = rpc(single_worker, "/mcp", "tools/list", key=key, headers={"mcp-session-id": session})
        assert f"{alias}-add" in listed_names(after), after.text
        again: Final = single_worker.client.delete("/mcp", headers={"x-litellm-api-key": key, "mcp-session-id": session})
        assert again.status_code == 200 and again.json() == _DELETED, again.text


def test_a_missing_key_cannot_initialize_a_session(single_worker: Gateway) -> None:
    """`initialize` with no credential must be refused with HTTP 401 and no `mcp-session-id` header, so
    an anonymous caller can never obtain a session to replay later."""
    response: Final = single_worker.client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": INITIALIZE},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert response.status_code == 401, response.text
    assert "mcp-session-id" not in response.headers


def test_aggregate_resource_read_routes_to_the_one_server_that_lists_the_uri(gateway: Gateway) -> None:
    """A key is granted a rich peer (lists `status://ready`) and a plain peer (no resources). A
    `resources/read` for `status://ready` on the aggregate `/mcp` must be routed to the rich peer only
    (#51462, `resolve_owner` in `operations.py`) and return contents whose text is `ready`. The Rust host
    instead refuses any read with more than one server in scope (`AmbiguousResourceServer`)."""
    with mcp_peer(rich=True) as rich, mcp_peer() as plain, gateway.scenario() as scenario:
        a: Final = register_mcp(scenario, rich, "dec1res" + uuid.uuid4().hex[:8])
        b: Final = register_mcp(scenario, plain, "dec1plain" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        result: Final = rpc_result(rpc(gateway, "/mcp", "resources/read", {"uri": "status://ready"}, key=key))
        contents: Final = result["contents"]
        assert isinstance(contents, list) and contents, result
        assert contents[0]["text"] == "ready", result
        reads: Final = [
            item for item in plain.drain() if isinstance(item.get("body"), dict) and item["body"].get("method") == "resources/read"
        ]
        assert reads == []


def test_aggregate_resource_read_of_a_uri_two_servers_list_is_ambiguous(gateway: Gateway) -> None:
    """Two rich peers both list `status://ready`. The aggregate read must fail with a JSON-RPC error whose
    message contains `Resource URI 'status://ready' is ambiguous across MCP servers` and the hint
    `Use /{server_name}/mcp or x-mcp-servers to select one server.`, and neither peer may get a
    `resources/read`."""
    with mcp_peer(rich=True) as first, mcp_peer(rich=True) as second, gateway.scenario() as scenario:
        a: Final = register_mcp(scenario, first, "dec1amb" + uuid.uuid4().hex[:8])
        b: Final = register_mcp(scenario, second, "dec1amb" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        error: Final = rpc_error(rpc(gateway, "/mcp", "resources/read", {"uri": "status://ready"}, key=key))
        message: Final = str(error.get("message"))
        assert "Resource URI 'status://ready' is ambiguous across MCP servers" in message, error
        assert "Use /{server_name}/mcp or x-mcp-servers to select one server." in message, error
        for peer in (first, second):
            assert not any(
                isinstance(item.get("body"), dict) and item["body"].get("method") == "resources/read"
                for item in peer.drain()
            )


def test_aggregate_resource_read_of_a_uri_no_server_lists_is_not_found(gateway: Gateway) -> None:
    """With the same two servers, a read of `status://missing` must fail with a JSON-RPC error whose
    message contains `Resource 'status://missing' not found on any allowed MCP server.`"""
    with mcp_peer(rich=True) as rich, mcp_peer() as plain, gateway.scenario() as scenario:
        a: Final = register_mcp(scenario, rich, "dec1nf" + uuid.uuid4().hex[:8])
        b: Final = register_mcp(scenario, plain, "dec1nf" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        error: Final = rpc_error(rpc(gateway, "/mcp", "resources/read", {"uri": "status://missing"}, key=key))
        assert "Resource 'status://missing' not found on any allowed MCP server." in str(error.get("message")), error


def test_x_mcp_servers_header_resolves_an_otherwise_ambiguous_resource_read(gateway: Gateway) -> None:
    """Two rich peers list `status://ready`; the caller adds `x-mcp-servers: <alias of the first>`. With one
    server in scope no owner lookup runs, so the read must succeed with text `ready` and only the first
    peer may receive `resources/read`."""
    with mcp_peer(rich=True) as first, mcp_peer(rich=True) as second, gateway.scenario() as scenario:
        alias: Final = "dec1pick" + uuid.uuid4().hex[:8]
        a: Final = register_mcp(scenario, first, alias)
        b: Final = register_mcp(scenario, second, "dec1other" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        second.drain()
        result: Final = rpc_result(
            rpc(gateway, "/mcp", "resources/read", {"uri": "status://ready"}, key=key, headers={"x-mcp-servers": alias})
        )
        assert result["contents"][0]["text"] == "ready", result
        assert not any(
            isinstance(item.get("body"), dict) and item["body"].get("method") == "resources/read" for item in second.drain()
        )
