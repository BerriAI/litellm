"""Slots F1 to F3: MCP credentials reach only the MCP peer they belong to.

Each scenario registers a scripted MCP peer (``_support/mcp.py``) that records every request,
wires one credential slot to it and calls a tool, either directly over the server's MCP
endpoint or through ``/v1/chat/completions`` with the provider double asking for the tool. The
``echo`` tool succeeds and the ``deny`` tool answers HTTP 401, so both the success and the
upstream-rejection logging paths run.

- F1: static ``auth_value`` registered through ``/v1/mcp/server``.
- F2: per-user OAuth access token, issued by the OAuth 2.1 double through the gateway's
  authorization-code flow with PKCE.
- F2E: per-user env var value, stored through ``/v1/mcp/server/{server_id}/user-env-vars`` and
  substituted into the server's ``Authorization`` header.
- F3: client ``x-mcp-<server>-authorization`` request header.

Positive control: the peer's ``tools/call`` request must carry ``Authorization: Bearer
<canary>``, or the test fails before sweeping. Sensitivity control: the marker sent as the tool
argument must be reported where stored prompts belong. Then no sweep may find the canary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Final, Literal
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.mcp import JsonRpc, McpCaller, McpPeer, ScriptedTool, echo_tool, register_mcp, scripted_peer
from integration._support.oauth_server import oauth_server
from integration._support.wire import Reply, Request
from integration.security._canary import MARKER, Canary, canary, find_canary
from integration.security._sinks import CONFIG_MODEL, GENERIC_SINK, Caller, Rig, canary_rig, chat_upstream, settle
from integration.security._sweeps import Hit, assert_marker_seen, assert_no_hits, record_route_sweep, sweep_all

Via = Literal["direct", "chat"]
Outcome = Literal["success", "upstream_401"]
TOOL: Final[Mapping[Outcome, str]] = {"success": "echo", "upstream_401": "deny"}
USER_TOKEN: Final = "USER_TOKEN"
CLIENT_REDIRECT: Final = "http://127.0.0.1:9/cb"
SLACK: Final = timedelta(seconds=5)


def _tool_call(request: Request) -> Reply:
    """Provider double: asks for the first offered tool with the user text, then echoes the tool result."""
    body: Final = json.loads(request.body or b"{}")
    tools: Final = body.get("tools") or []
    messages: Final = body.get("messages") or []
    if not tools or any(message.get("role") == "tool" for message in messages):
        return chat_upstream(request)
    call: Final = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": tools[0]["function"]["name"],
            "arguments": json.dumps({"text": str(messages[-1].get("content", ""))}),
        },
    }
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
        ).encode()
    )


def _deny(params: JsonRpc) -> Reply:
    return Reply(
        status=401,
        body=b'{"error":"invalid_token"}',
        headers={"www-authenticate": 'Bearer error="invalid_token"'},
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """One owned proxy per module: every F credential is registered at runtime with a fresh core."""
    with canary_rig(tmp_path_factory.mktemp("canary-mcp"), upstream=_tool_call) as value:
        yield value


@dataclass(frozen=True, slots=True)
class Wiring:
    server_id: str
    alias: str
    caller: Caller
    headers: Mapping[str, str] = field(default_factory=dict)
    responses: tuple[httpx.Response, ...] = ()


def _caller(scenario: Scenario, server_id: str) -> Caller:
    grant: Final[JsonRpc] = {"mcp_servers": [server_id]}
    team: Final = scenario.team(object_permission=dict(grant))
    user: Final = scenario.user(user_role="internal_user")
    scenario.gateway.post("/team/member_add", {"team_id": team, "member": {"user_id": user, "role": "user"}})
    key: Final = scenario.key(team_id=team, user_id=user, models=[CONFIG_MODEL], object_permission=dict(grant))
    return Caller(team, user, key)


def _pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def _authorize_and_redeem(rig: Rig, alias: str, key: str) -> None:
    """Run the gateway's authorization-code flow for the caller; the double mints the canary."""
    client: Final = rig.proxy.client
    base: Final = str(client.base_url).rstrip("/")
    registered: Final = client.post(f"/{alias}/register", json={"redirect_uris": [CLIENT_REDIRECT]})
    assert registered.status_code in (200, 201), registered.text
    client_id: Final = string_value(registered.json()["client_id"])
    verifier: Final = secrets.token_urlsafe(32)
    started: Final = client.get(
        f"/{alias}/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
            "response_type": "code",
            "state": "canary-state",
            "code_challenge": _pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "scope": "tools.call",
        },
        headers={"x-litellm-api-key": key},
    )
    assert started.status_code in (302, 307), started.text
    consent: Final = httpx.get(started.headers["location"], follow_redirects=False, trust_env=False)
    assert consent.status_code == 302, consent.text
    returned: Final = client.get(
        consent.headers["location"].removeprefix(base), headers={"x-litellm-api-key": key}, cookies=started.cookies
    )
    assert returned.status_code == 302, returned.text
    code: Final = parse_qs(urlsplit(returned.headers["location"]).query)["code"][0]
    redeemed: Final = client.post(
        f"/{alias}/token",
        headers={"x-litellm-api-key": key},
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
        },
    )
    assert redeemed.status_code == 200, redeemed.text


@contextmanager
def _wired(slot: str, rig: Rig, scenario: Scenario, peer: McpPeer, credential: Canary) -> Iterator[Wiring]:
    """Register the peer with ``credential`` in ``slot`` and return the caller that uses it."""
    alias: Final = "canary" + uuid.uuid4().hex[:8]
    if slot == "F1":
        server: Final = register_mcp(
            scenario, peer, alias, auth_type="bearer_token", credentials={"auth_value": credential.value}
        )
        yield Wiring(server, alias, _caller(scenario, server))
    elif slot == "F2":
        with oauth_server(mint=lambda grant: credential.value) as auth:
            server_f2: Final = register_mcp(
                scenario,
                peer,
                alias,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
                issuer=auth.issuer,
                authorization_url=auth.issuer + "/authorize",
                token_url=auth.issuer + "/token",
                registration_url=auth.issuer + "/register",
                credentials={"client_id": "canary-client", "client_secret": "canary-client-secret"},
            )
            caller_f2: Final = _caller(scenario, server_f2)
            _authorize_and_redeem(rig, alias, caller_f2.key)
            yield Wiring(server_f2, alias, caller_f2)
    elif slot == "F2E":
        server_f2e: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="none",
            env_vars=[{"name": USER_TOKEN, "scope": "user", "description": "per-user token"}],
            static_headers={"Authorization": f"Bearer ${{{USER_TOKEN}}}"},
        )
        caller_f2e: Final = _caller(scenario, server_f2e)
        stored: Final = rig.proxy.request(
            "POST",
            f"/v1/mcp/server/{server_f2e}/user-env-vars",
            {"values": {USER_TOKEN: credential.value}},
            key=caller_f2e.key,
        )
        assert stored.status_code == 200, stored.text
        yield Wiring(server_f2e, alias, caller_f2e, responses=(stored,))
    else:
        assert slot == "F3", slot
        server_f3: Final = register_mcp(scenario, peer, alias)
        yield Wiring(
            server_f3,
            alias,
            _caller(scenario, server_f3),
            headers={f"x-mcp-{alias}-authorization": f"Bearer {credential.value}"},
        )


def _send(rig: Rig, wiring: Wiring, via: Via, tool: str, text: str) -> httpx.Response:
    if via == "direct":
        return McpCaller(rig.proxy, wiring.caller.key, "server_mcp", wiring.alias, wiring.headers).rpc(
            "tools/call", {"name": f"{wiring.alias}-{tool}", "arguments": {"text": text}}
        )
    return rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": CONFIG_MODEL,
            "messages": [{"role": "user", "content": text}],
            "tools": [
                {
                    "type": "mcp",
                    "server_url": f"litellm_proxy/mcp/{wiring.alias}",
                    "server_label": "litellm",
                    "require_approval": "never",
                    "allowed_tools": [f"{wiring.alias}-{tool}"],
                }
            ],
        },
        key=wiring.caller.key,
        headers=wiring.headers,
    )


def _answer(response: httpx.Response, via: Via) -> str:
    """The text the caller got back: the tool result (direct) or the assistant message (chat)."""
    assert response.status_code == 200, response.text
    if via == "chat":
        return string_value(object_value(response.json()["choices"][0]["message"])["content"])
    data: Final = next(
        line.removeprefix("data:").strip() for line in response.text.splitlines() if line.startswith("data:")
    )
    result: Final = object_value(json.loads(data)["result"])
    assert isinstance(result["content"], list)
    return string_value(object_value(result["content"][0])["text"])


def _tool_call_authorizations(peer: McpPeer, seen: list[dict[str, object]]) -> tuple[object, ...]:
    seen.extend(peer.drain())
    return tuple(
        object_value(call["headers"]).get("authorization")
        for call in seen
        if isinstance(call["body"], dict) and call["body"].get("method") == "tools/call"
    )


def _spend_rows(marker: Canary, since: datetime, call_types: frozenset[str]) -> Sequence[Mapping[str, object]]:
    """Every spend row carrying ``marker``, once a row of each of ``call_types`` has been written."""
    return eventually(
        lambda: read_rows(
            'SELECT request_id, call_type FROM "LiteLLM_SpendLogs" '
            'WHERE "startTime" >= %s AND proxy_server_request::text LIKE %s',
            (since.astimezone(UTC).replace(tzinfo=None) - SLACK, f"%{marker.core}%"),
        ),
        lambda rows: call_types <= {row["call_type"] for row in rows},
        seconds=70,
    )


def _drawer_hits(
    rig: Rig, request_ids: Sequence[str], canaries: Sequence[Canary], callers: Mapping[str, str]
) -> tuple[Hit, ...]:
    """S2 for the Logs drawer of every extra spend row the scenario wrote."""
    found: Final[list[Hit]] = []  # mutable-ok: accumulated across rows and callers
    for request_id in request_ids:
        for label, key in callers.items():
            response = rig.proxy.request("GET", f"/spend/logs/ui/{request_id}", key=key)
            where = f"GET /spend/logs/ui/{request_id} as {label} -> {response.status_code}"
            found.extend(
                Hit("S2", where, match.slot, match.encoding) for match in find_canary(response.content, canaries)
            )
    return tuple(found)


@pytest.mark.timeout(240)  # full S1/S2 walk: every table and ~400 GET routes as two callers
@pytest.mark.parametrize("outcome", ["success", "upstream_401"])
@pytest.mark.parametrize("via", ["direct", "chat"])
@pytest.mark.parametrize("slot", ["F1", "F2", "F2E", "F3"])
def test_mcp_credential_reaches_only_its_peer(
    rig: Rig, slot: str, via: Via, outcome: Outcome, request: pytest.FixtureRequest
) -> None:
    credential: Final = canary(slot)
    marker: Final = canary(MARKER)
    started: Final = datetime.now(UTC)
    peer_calls: Final[list[dict[str, object]]] = []  # mutable-ok: accumulates the peer's recorded requests
    with (
        scripted_peer(echo_tool("echo"), ScriptedTool("deny", _deny)) as peer,
        rig.proxy.scenario() as scenario,
        _wired(slot, rig, scenario, peer, credential) as wiring,
    ):
        response: Final = _send(rig, wiring, via, TOOL[outcome], f"slot {slot} {marker.value}")
        answer: Final = _answer(response, via)
        assert (marker.value in answer) if outcome == "success" else ("401" in answer), answer
        assert _tool_call_authorizations(peer, peer_calls) == (f"Bearer {credential.value}",), (
            f"Positive control: the MCP peer never received the {slot} canary on tools/call"
        )
        rows: Final = _spend_rows(
            marker, started, frozenset({"call_mcp_tool", "acompletion"} if via == "chat" else {"call_mcp_tool"})
        )
        tool_row: Final = next(str(row["request_id"]) for row in rows if row["call_type"] == "call_mcp_tool")
        settle(rig, tool_row, marker)

        canaries: Final = (marker, credential)
        report: Final = sweep_all(
            rig.proxy,
            canaries,
            responses=(response, *wiring.responses),
            sinks={name: sink.requests() for name, sink in rig.sinks.items()},
            ids={
                "request_id": tool_row,
                "server_id": wiring.server_id,
                "mcp_server_name": wiring.alias,
                "team_id": wiring.caller.team_id,
                "user_id": wiring.caller.user_id,
                "model": CONFIG_MODEL,
                "model_id": CONFIG_MODEL,
            },
            callers=wiring.caller.callers(rig),
            own_headers=rig.own_headers,
            since=started,
        )
        record_route_sweep(report.routes, request.node.nodeid)
        assert_marker_seen(report, {"S2": f"GET /spend/logs?request_id={tool_row} as admin -> 200"})
        assert_marker_seen(
            report,
            {
                "S1": "LiteLLM_SpendLogs.proxy_server_request",
                "S2": f"GET /spend/logs/ui/{tool_row} as admin -> 200",
                "S4": f"{GENERIC_SINK}[",
                **({"S3": "response[0] POST"} if outcome == "success" else {}),
            },
        )
        other_rows: Final = tuple(str(row["request_id"]) for row in rows if str(row["request_id"]) != tool_row)
        assert_no_hits(
            (*report.credential_hits(), *_drawer_hits(rig, other_rows, (credential,), wiring.caller.callers(rig))),
            f"slot {slot}, {via}, {outcome}",
        )
