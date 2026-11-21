"""MCP tool-call rate limits: per key, per team and per server windows, and how they combine.

``enforce_mcp_server_rate_limits`` (``litellm/proxy/hooks/parallel_request_limiter_v3.py:3065``) runs before an MCP
server is called and builds up to three request-only descriptors:

- ``mcp_per_key`` with value ``<hashed key>:<server name>`` from the key's ``metadata.mcp_rpm_limit`` map;
- ``mcp_per_team`` with value ``<team_id>:<server name>`` from the team's ``metadata.mcp_rpm_limit`` map;
- ``mcp_server`` with value ``<server_id>`` from the server's own ``rpm``.

The server name is ``server.alias or server.server_name or server.name`` (:3071), never the server id. All
descriptors are checked and incremented in one atomic step (``atomic_check_and_increment_by_n``), so a call refused
by one scope adds nothing to the others. The refusal is the v3 limiter's ``Rate limit exceeded for <scope>: <value>.
Limit type: requests. Current limit: <n>, Remaining: 0. Limit resets at: <UTC>``. There is no per-tool limit on the
branch: the finest scope is the server.

The REST listing ``/mcp-rest/tools/list`` runs the same check for every server it lists
(``litellm/proxy/mcp_server/rest_endpoints.py:786``), so listing uses the same windows as calling.
``test_dec1_budlim2_realtime_and_mcp.py`` already pins the per key text, the server id text and per server
separation for keys; ``mcp/test_mcp_rate_limits.py`` pins the windows across two replicas.
"""

from __future__ import annotations

import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.mcp import echo_tool, register_mcp, scripted_peer, tool_calls
from tests.integration.authorization.dec1_limcap4_support import counter, counter_settles
from tests.integration.authorization.dec1_limits_support import hashed, limit_message


def _alias(label: str) -> str:
    return f"limcap4{label}{uuid.uuid4().hex[:8]}"


def _call(gateway: Gateway, key: str, server_id: str, alias: str) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"name": f"{alias}-echo", "arguments": {"probe": 1}, "server_id": server_id},
    )


def _list(gateway: Gateway, key: str, server_id: str) -> httpx.Response:
    return gateway.client.get(
        "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": server_id}
    )


def test_a_team_mcp_limit_refusal_names_the_team_and_the_server(gateway: Gateway) -> None:
    """A team has ``mcp_rpm_limit: {<alias>: 1}``. Its key calls the server's tool twice through
    ``/mcp-rest/tools/call``. The second call is HTTP 429 with ``Rate limit exceeded for mcp_per_team:
    <team_id>:<alias>. Limit type: requests. Current limit: 1, Remaining: 0`` and the peer receives no second
    ``tools/call``."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("team")
        server_id: Final = register_mcp(scenario, peer, alias)
        permission: Final = {"mcp_servers": [server_id]}
        team: Final = scenario.team(mcp_rpm_limit={alias: 1}, object_permission=permission)
        key: Final = scenario.key(team_id=team, object_permission=permission)
        assert _call(gateway, key, server_id, alias).status_code == 200
        peer.drain()
        refused: Final = _call(gateway, key, server_id, alias)
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_per_team", f"{team}:{alias}", "requests", 1) in refused.text, refused.text
        assert tool_calls(peer.drain()) == ()


def test_a_team_mcp_window_is_shared_by_two_keys_of_the_team(gateway: Gateway) -> None:
    """The team allows 2 calls a minute to the server. Key one calls twice and is served both times; key two's first
    call is refused 429 for ``mcp_per_team``: the window is the team's, keyed ``<team_id>:<alias>``, not each key's."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("shared")
        server_id: Final = register_mcp(scenario, peer, alias)
        permission: Final = {"mcp_servers": [server_id]}
        team: Final = scenario.team(mcp_rpm_limit={alias: 2}, object_permission=permission)
        first: Final = scenario.key(team_id=team, object_permission=permission)
        second: Final = scenario.key(team_id=team, object_permission=permission)
        assert [_call(gateway, first, server_id, alias).status_code for _ in range(2)] == [200, 200]
        refused: Final = _call(gateway, second, server_id, alias)
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_per_team", f"{team}:{alias}", "requests", 2) in refused.text, refused.text


def test_a_team_mcp_limit_on_one_server_does_not_limit_another_server(gateway: Gateway) -> None:
    """The team limits server A to one call a minute and has access to servers A and B. After one call to A, a second
    call to A is 429 while two calls to B are 200: only servers named in the team's map get a ``mcp_per_team``
    descriptor (:3050)."""
    with (
        scripted_peer(echo_tool("echo")) as peer_a,
        scripted_peer(echo_tool("echo")) as peer_b,
        gateway.scenario() as scenario,
    ):
        alias_a, alias_b = _alias("a"), _alias("b")
        server_a: Final = register_mcp(scenario, peer_a, alias_a)
        server_b: Final = register_mcp(scenario, peer_b, alias_b)
        permission: Final = {"mcp_servers": [server_a, server_b]}
        team: Final = scenario.team(mcp_rpm_limit={alias_a: 1}, object_permission=permission)
        key: Final = scenario.key(team_id=team, object_permission=permission)
        assert _call(gateway, key, server_a, alias_a).status_code == 200
        assert _call(gateway, key, server_a, alias_a).status_code == 429
        assert [_call(gateway, key, server_b, alias_b).status_code for _ in range(2)] == [200, 200]


def test_the_tighter_of_the_key_and_team_mcp_limits_refuses(gateway: Gateway) -> None:
    """The key allows 3 calls a minute to the server and its team allows 1. The second call is refused for
    ``mcp_per_team`` with ``Current limit: 1``: both descriptors are enforced together, so the team cap is not lifted
    by a looser key cap."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("tight")
        server_id: Final = register_mcp(scenario, peer, alias)
        permission: Final = {"mcp_servers": [server_id]}
        team: Final = scenario.team(mcp_rpm_limit={alias: 1}, object_permission=permission)
        key: Final = scenario.key(team_id=team, object_permission=permission, mcp_rpm_limit={alias: 3})
        assert _call(gateway, key, server_id, alias).status_code == 200
        refused: Final = _call(gateway, key, server_id, alias)
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_per_team", f"{team}:{alias}", "requests", 1) in refused.text, refused.text


def test_a_key_mcp_limit_keyed_by_the_server_id_does_not_apply(gateway: Gateway) -> None:
    """A key's ``mcp_rpm_limit`` names the server by its id, ``{<server_id>: 1}``, instead of its alias. Three calls
    are all served: the per key descriptor looks the limit up by ``server.alias or server.server_name or
    server.name`` (:2997), so an id never matches. A port that also accepts ids would start refusing these keys."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("byid")
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, mcp_rpm_limit={server_id: 1})
        assert [_call(gateway, key, server_id, alias).status_code for _ in range(3)] == [200, 200, 200]


def test_an_mcp_refusal_by_the_key_scope_does_not_use_up_the_server_window(gateway: Gateway) -> None:
    """The server has ``rpm: 2`` and key A allows itself one call. A is served once and refused once for
    ``mcp_per_key``; key B is then still served once before B is refused for ``mcp_server: <server_id>``. A refused
    call increments no descriptor (``atomic_check_and_increment_by_n`` is all or nothing), so A's refusal did not take
    the server's second request."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("atomic")
        server_id: Final = register_mcp(scenario, peer, alias, rpm=2)
        permission: Final = {"mcp_servers": [server_id]}
        first: Final = scenario.key(object_permission=permission, mcp_rpm_limit={alias: 1})
        second: Final = scenario.key(object_permission=permission)
        assert _call(gateway, first, server_id, alias).status_code == 200
        assert _call(gateway, first, server_id, alias).status_code == 429
        assert _call(gateway, second, server_id, alias).status_code == 200
        refused: Final = _call(gateway, second, server_id, alias)
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_server", server_id, "requests", 2) in refused.text, refused.text


def test_an_mcp_call_writes_one_request_to_each_of_its_windows(gateway: Gateway) -> None:
    """A key with ``mcp_rpm_limit: {<alias>: 5}`` in a team with ``mcp_rpm_limit: {<alias>: 5}`` calls a server with
    ``rpm: 5`` once. Redis then holds 1 on ``{mcp_per_key:<hash>:<alias>}:requests``,
    ``{mcp_per_team:<team_id>:<alias>}:requests`` and ``{mcp_server:<server_id>}:requests``, and no token counter for
    any of them (MCP descriptors carry no ``tokens_per_unit``)."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("counters")
        server_id: Final = register_mcp(scenario, peer, alias, rpm=5)
        permission: Final = {"mcp_servers": [server_id]}
        team: Final = scenario.team(mcp_rpm_limit={alias: 5}, object_permission=permission)
        key: Final = scenario.key(team_id=team, object_permission=permission, mcp_rpm_limit={alias: 5})
        assert _call(gateway, key, server_id, alias).status_code == 200
        assert counter_settles("mcp_per_key", f"{hashed(key)}:{alias}", "requests", 1) == 1
        assert counter("mcp_per_team", f"{team}:{alias}", "requests") == 1
        assert counter("mcp_server", server_id, "requests") == 1
        assert counter("mcp_server", server_id, "tokens") is None


def test_a_server_without_rpm_and_keys_without_mcp_limits_write_no_mcp_counters(gateway: Gateway) -> None:
    """Nothing limits the server: no ``rpm`` on it and no ``mcp_rpm_limit`` on the key or its team. After two calls
    Redis has no ``{mcp_server:<server_id>}:requests`` and no ``{mcp_per_key:<hash>:<alias>}:requests``: with no
    descriptors the check returns before touching Redis (:3095)."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("free")
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]})
        assert [_call(gateway, key, server_id, alias).status_code for _ in range(2)] == [200, 200]
        assert counter("mcp_server", server_id, "requests") is None
        assert counter("mcp_per_key", f"{hashed(key)}:{alias}", "requests") is None


def test_listing_a_servers_tools_uses_the_keys_per_server_window(gateway: Gateway) -> None:
    """A key allows itself one call a minute to the server. It lists the server's tools through
    ``/mcp-rest/tools/list?server_id=<id>`` (200), and its first tool call is then refused 429 for ``mcp_per_key``:
    the listing ran ``enforce_mcp_server_rate_limits`` and took the window's only request. UNVERIFIED: that the
    single-server REST listing goes through ``_get_tools_for_single_server`` with ``apply_tool_filters`` on."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("list")
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, mcp_rpm_limit={alias: 1})
        listed: Final = _list(gateway, key, server_id)
        assert listed.status_code == 200, listed.text
        refused: Final = _call(gateway, key, server_id, alias)
        assert refused.status_code == 429, refused.text
        assert limit_message("mcp_per_key", f"{hashed(key)}:{alias}", "requests", 1) in refused.text, refused.text


def test_raising_a_keys_mcp_limit_through_key_update_admits_it_again(gateway: Gateway) -> None:
    """A key with ``mcp_rpm_limit: {<alias>: 1}`` is refused on its second call. ``/key/update`` sets
    ``metadata.mcp_rpm_limit`` to ``{<alias>: 3}``, and within seconds, with no restart, the key is served again in
    the same minute: the cached key is evicted on update and the limiter reads the new map on the next call."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("raise")
        server_id: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [server_id]}, mcp_rpm_limit={alias: 1})
        assert _call(gateway, key, server_id, alias).status_code == 200
        assert _call(gateway, key, server_id, alias).status_code == 429
        gateway.post("/key/update", {"key": key, "metadata": {"mcp_rpm_limit": {alias: 3}}})
        served: Final = eventually(
            lambda: _call(gateway, key, server_id, alias),
            lambda response: response.status_code == 200,
            seconds=20,
        )
        assert served.status_code == 200, served.text


def test_lowering_a_teams_mcp_limit_through_team_update_refuses_its_key(gateway: Gateway) -> None:
    """A team allows 5 calls a minute to the server and its key has made one. ``/team/update`` lowers
    ``mcp_rpm_limit`` to ``{<alias>: 1}``, and the key's next call is refused 429 for ``mcp_per_team`` with
    ``Current limit: 1`` (the team window already holds one request)."""
    with scripted_peer(echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = _alias("lower")
        server_id: Final = register_mcp(scenario, peer, alias)
        permission: Final = {"mcp_servers": [server_id]}
        team: Final = scenario.team(mcp_rpm_limit={alias: 5}, object_permission=permission)
        key: Final = scenario.key(team_id=team, object_permission=permission)
        assert _call(gateway, key, server_id, alias).status_code == 200
        gateway.post("/team/update", {"team_id": team, "mcp_rpm_limit": {alias: 1}})
        refused: Final = eventually(
            lambda: _call(gateway, key, server_id, alias),
            lambda response: response.status_code == 429,
            seconds=20,
        )
        assert limit_message("mcp_per_team", f"{team}:{alias}", "requests", 1) in refused.text, refused.text
