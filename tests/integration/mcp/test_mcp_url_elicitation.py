import asyncio
import itertools
import json
import threading
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.mcp import (
    INITIALIZE,
    EntryPoint,
    JsonRpc,
    JsonRpcFailure,
    McpCaller,
    McpPeer,
    PeerKind,
    ScriptedTool,
    _parse_rpc_body,
    peer_of,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.oauth_server import oauth_server
from integration._support.process import WorkerHealthcheck, graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Wire, wire_server
from mcp import ClientSession, MCPError
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent

pytestmark: Final = pytest.mark.shares_worker_state

MESSAGE: Final = "This request requires more information."
FLATTENED: Final = f"MCPError: {MESSAGE}"
MCP_ENTRIES: Final[tuple[EntryPoint, ...]] = ("mcp", "root", "server_mcp", "sse")
AuthMode = Literal["none", "token_exchange", "oauth_delegate"]
SPEND_ROWS: Final = (
    "SELECT status, metadata->'mcp_tool_call_metadata'->'arguments'->>'nonce' AS nonce"
    ' FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s'
)


def _elicitation(nonce: str) -> dict[str, object]:
    return {
        "code": -32042,
        "message": MESSAGE,
        "data": {
            "elicitations": [
                {
                    "mode": "url",
                    "elicitationId": nonce,
                    "url": f"https://consent.example.invalid/oauth2/authorize?request_uri={nonce}",
                    "message": "Authorize to continue.",
                }
            ]
        },
    }


def _nonce(params: JsonRpc) -> str:
    arguments: Final = params["arguments"]
    assert isinstance(arguments, Mapping), params
    return str(arguments["nonce"])


def _consent_tool(*, event_stream: bool = False) -> ScriptedTool:
    return ScriptedTool(
        "consent", lambda params: JsonRpcFailure(_elicitation(_nonce(params))), event_stream=event_stream
    )


def _echo_tool() -> ScriptedTool:
    return ScriptedTool("echo", lambda params: text_result(_nonce(params)))


def _failing_tool(error: JsonRpc) -> ScriptedTool:
    return ScriptedTool("consent", lambda _: JsonRpcFailure(error))


def _upstream_nonces(peer: McpPeer) -> list[str]:
    return sorted(_nonce(call["body"]["params"]) for call in tool_calls(peer.drain()))


def _spend_rows(key: str, count: int) -> list[tuple[str, str]]:
    rows: Final = eventually(
        lambda: read_rows(SPEND_ROWS, (sha256(key.encode()).hexdigest(), "call_mcp_tool")),
        lambda found: len(found) >= count,
        seconds=70,
        return_last_on_timeout=True,
    )
    return sorted((str(row["status"]), str(row["nonce"])) for row in rows)


def _server_arg(entry: EntryPoint, identity: str) -> str | None:
    return identity if entry == "rest" else None


@contextmanager
def _granted(gateway: Gateway, *tools: ScriptedTool, **fields: object) -> Iterator[tuple[McpPeer, str, str, str]]:
    with scripted_peer(*tools) as upstream, gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias, **fields)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        upstream.drain()
        yield upstream, alias, identity, key


@pytest.mark.parametrize("entry", MCP_ENTRIES)
def test_url_elicitation_required_reaches_every_mcp_entry_point_as_the_upstream_jsonrpc_error(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, _, key):
        caller: Final = McpCaller(gateway, key, entry, alias)
        assert caller.initialize().ok
        nonce: Final = uuid.uuid4().hex
        reply: Final = caller.rpc("tools/call", {"name": f"{alias}-consent", "arguments": {"nonce": nonce}})
        assert reply.status_code == 200, reply.text
        assert _parse_rpc_body(reply) == {"jsonrpc": "2.0", "id": 1, "error": _elicitation(nonce)}, reply.text
        control: Final = caller.call(f"{alias}-echo", {"nonce": "control"})
        assert control.ok and control.text == "control", control.raw
        assert _upstream_nonces(upstream) == sorted(["control", nonce])


def _official_outcomes(
    gateway: Gateway, key: str, path: str, alias: str, nonce: str, *, legacy_sse: bool
) -> tuple[Mapping[str, object] | None, str | None]:
    url: Final = str(gateway.client.base_url).rstrip("/") + path
    headers: Final = {"x-litellm-api-key": key}

    async def raised(session: ClientSession) -> Mapping[str, object] | None:
        try:
            await session.call_tool(f"{alias}-consent", {"nonce": nonce})
        except MCPError as error:
            return error.error.model_dump(mode="json", by_alias=True, exclude_none=True)
        return None

    async def run() -> tuple[Mapping[str, object] | None, str | None]:
        transport: Final = (
            sse_client(url, headers=headers)
            if legacy_sse
            else streamable_http_client(url, http_client=httpx.AsyncClient(headers=headers, timeout=30))
        )
        async with transport as streams, ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            error: Final = await raised(session)
            control: Final = await session.call_tool(f"{alias}-echo", {"nonce": "control"})
            content: Final = control.content[0] if control.content else None
            return error, content.text if isinstance(content, TextContent) and not control.is_error else None

    return asyncio.run(run())


@pytest.mark.parametrize(("path", "legacy_sse"), (("/mcp/", False), ("/mcp/sse", True)), ids=("streamable", "sse"))
def test_official_sdk_client_raises_the_upstream_url_elicitation_error_and_keeps_its_session(
    gateway: Gateway, path: str, legacy_sse: bool
) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, _, key):
        nonce: Final = uuid.uuid4().hex
        error, control = _official_outcomes(gateway, key, path, alias, nonce, legacy_sse=legacy_sse)
        assert error == _elicitation(nonce), error
        assert control == "control"
        assert _upstream_nonces(upstream) == sorted(["control", nonce])


@pytest.mark.parametrize("kind", ("http", "sse", "stdio"))
def test_sdk_server_raising_url_elicitation_required_is_relayed_over_every_upstream_transport(
    gateway: Gateway, kind: PeerKind
) -> None:
    with peer_of(kind, rich=True) as upstream, gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"elicitation_id": nonce})
        assert outcome.status == 200 and outcome.error, outcome.raw
        assert json.loads(outcome.error) == {
            "code": -32042,
            "message": "URL elicitation required",
            "data": {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorize to continue.",
                        "url": f"https://consent.example.invalid/authorize?id={nonce}",
                        "elicitationId": nonce,
                    }
                ]
            },
        }, outcome.raw
        control: Final = caller.call(f"{alias}-add", {"a": 3, "b": 5})
        assert control.ok and control.text == "8", control.raw


def test_url_elicitation_required_sent_as_an_event_stream_reply_is_relayed(gateway: Gateway) -> None:
    with _granted(gateway, _consent_tool(event_stream=True), _echo_tool()) as (upstream, alias, _, key):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce})
        assert outcome.status == 200 and outcome.error, outcome.raw
        assert json.loads(outcome.error) == _elicitation(nonce), outcome.raw
        assert _upstream_nonces(upstream) == [nonce]


def test_rest_tool_call_keeps_the_flattened_error_result_for_url_elicitation_required(gateway: Gateway) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, identity, key):
        nonce: Final = uuid.uuid4().hex
        response: Final = gateway.client.post(
            "/mcp-rest/tools/call",
            headers={"x-litellm-api-key": key},
            json={"server_id": identity, "name": f"{alias}-consent", "arguments": {"nonce": nonce}},
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["isError"] is True, response.text
        assert [(block["type"], block["text"]) for block in body["content"]] == [("text", FLATTENED)], response.text
        assert nonce not in response.text, response.text
        assert _upstream_nonces(upstream) == [nonce]


REDACTED: Final = "[KEYWORD_REDACTED]"


def _addressed_elicitation(nonce: str, shown: str, hinted: str, verb: str = "Authorize") -> dict[str, object]:
    return {
        "code": -32042,
        "message": f"{verb} to continue for {shown}.",
        "data": {
            "elicitations": [
                {
                    "mode": "url",
                    "elicitationId": nonce,
                    "url": f"https://consent.example.invalid/oauth2/authorize?request_uri={nonce}&login_hint={hinted}",
                    "message": f"Sign in as {shown}.",
                }
            ]
        },
    }


@contextmanager
def _output_guardrail(gateway: Gateway, keyword: str, action: Literal["MASK", "BLOCK"]) -> Iterator[None]:
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": "elicit" + uuid.uuid4().hex[:8],
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "post_mcp_call",
                    "default_on": True,
                    "blocked_words": [{"keyword": keyword, "action": action}],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    try:
        yield
    finally:
        deleted: Final = gateway.client.delete(
            f"/guardrails/{created.json()['guardrail_id']}", headers={"x-litellm-api-key": gateway.key}
        )
        assert deleted.status_code == 200, deleted.text


def _addressed_tools(
    email: str, hinted: str | None = None, verb: str = "Authorize"
) -> tuple[ScriptedTool, ScriptedTool]:
    return (
        ScriptedTool(
            "consent",
            lambda params: JsonRpcFailure(_addressed_elicitation(_nonce(params), email, hinted or email, verb)),
        ),
        ScriptedTool("whoami", lambda _: text_result(f"Signed in as {email}.")),
    )


@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_masking_output_guardrail_redacts_the_elicitation_messages_and_leaves_the_consent_url_alone(
    gateway: Gateway, entry: EntryPoint
) -> None:
    email: Final = f"john.doe.{uuid.uuid4().hex[:8]}@example.com"
    with (
        _output_guardrail(gateway, email, "MASK"),
        _granted(gateway, *_addressed_tools(email)) as (upstream, alias, identity, key),
    ):
        caller: Final = McpCaller(gateway, key, entry, alias)
        control: Final = eventually(
            lambda: caller.call(f"{alias}-whoami", {}, _server_arg(entry, identity)),
            lambda outcome: outcome.text == f"Signed in as {REDACTED}.",
            seconds=40,
            return_last_on_timeout=True,
        )
        assert control.text == f"Signed in as {REDACTED}.", control.raw
        upstream.drain()
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce}, _server_arg(entry, identity))
        assert outcome.status == 200 and outcome.error, outcome.raw
        redacted: Final = _addressed_elicitation(nonce, REDACTED, email)
        assert (json.loads(outcome.error) if entry == "mcp" else outcome.error) == (
            redacted if entry == "mcp" else f"MCPError: {redacted['message']}"
        ), outcome.raw
        assert _upstream_nonces(upstream) == [nonce]


@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_blocking_output_guardrail_withholds_the_elicitation_like_a_blocked_tool_result(
    gateway: Gateway, entry: EntryPoint
) -> None:
    email: Final = f"john.doe.{uuid.uuid4().hex[:8]}@example.com"
    with (
        _output_guardrail(gateway, email, "BLOCK"),
        _granted(gateway, *_addressed_tools(email)) as (upstream, alias, identity, key),
    ):
        caller: Final = McpCaller(gateway, key, entry, alias)
        blocked_result: Final = eventually(
            lambda: caller.call(f"{alias}-whoami", {}, _server_arg(entry, identity)),
            lambda outcome: outcome.error is not None,
            seconds=40,
            return_last_on_timeout=True,
        )
        assert blocked_result.error is not None, blocked_result.raw
        upstream.drain()
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce}, _server_arg(entry, identity))
        assert (outcome.status, outcome.text, outcome.error) == (
            blocked_result.status,
            blocked_result.text,
            blocked_result.error,
        ), outcome.raw
        assert "consent.example.invalid" not in str(outcome.raw) and nonce not in str(outcome.raw), outcome.raw
        assert _upstream_nonces(upstream) == [nonce]


@contextmanager
def _authenticated(
    scenario: Scenario, upstream: McpPeer, mode: AuthMode
) -> Iterator[tuple[str, str, Mapping[str, str], object]]:
    alias: Final = "elicit" + uuid.uuid4().hex[:8]
    bearer: Final = "user-" + uuid.uuid4().hex
    if mode == "token_exchange":
        with oauth_server() as auth:
            exchanged: Final = register_mcp(
                scenario,
                upstream,
                alias,
                auth_type="oauth2_token_exchange",
                token_exchange_endpoint=auth.issuer + "/token",
                audience="urn:integration:peer",
                credentials={"client_id": "te-client", "client_secret": "te-secret"},
            )
            auth.drain()
            yield alias, exchanged, {"Authorization": f"Bearer {bearer}"}, auth
        return
    if mode == "oauth_delegate":
        yield (
            alias,
            register_mcp(scenario, upstream, alias, auth_type="oauth_delegate"),
            {"Authorization": f"Bearer {bearer}"},
            None,
        )
        return
    yield alias, register_mcp(scenario, upstream, alias), {}, None


def _authorizations(peer: McpPeer, nonce: str) -> tuple[str | None, ...]:
    return tuple(
        next((str(value) for name, value in call["headers"].items() if str(name).lower() == "authorization"), None)
        for call in tool_calls(peer.drain())
        if _nonce(call["body"]["params"]) == nonce
    )


@pytest.mark.parametrize("mode", ("none", "token_exchange", "oauth_delegate"))
@pytest.mark.parametrize("entry", ("mcp", "server_mcp", "rest"))
def test_url_elicitation_required_on_each_upstream_auth_path_is_an_mcp_error_and_a_rest_error_result(
    gateway: Gateway, mode: AuthMode, entry: EntryPoint
) -> None:
    with (
        scripted_peer(_consent_tool(), _echo_tool()) as upstream,
        gateway.scenario() as scenario,
        _authenticated(scenario, upstream, mode) as (alias, identity, headers, auth),
    ):
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias, headers=headers)
        upstream.drain()
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce}, _server_arg(entry, identity))
        assert outcome.status == 200 and outcome.error, outcome.raw
        if entry == "rest":
            assert outcome.error == FLATTENED and nonce not in outcome.raw, outcome.raw
        else:
            assert json.loads(outcome.error) == _elicitation(nonce), outcome.raw
        sent: Final = _authorizations(upstream, nonce)
        assert len(sent) == 1, sent
        if mode == "oauth_delegate":
            assert sent == (headers["Authorization"],)
        if mode == "token_exchange":
            exchanges: Final = auth.token_requests()
            assert len(exchanges) == 1 and exchanges[0]["subject_token"] == headers["Authorization"][7:], exchanges
            assert sent[0] is not None and sent[0] != headers["Authorization"] and auth.is_live(sent[0][7:]), sent
        control: Final = caller.call(f"{alias}-echo", {"nonce": "control"}, _server_arg(entry, identity))
        assert control.ok and control.text == "control", control.raw


@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_url_elicitation_required_call_is_logged_once_as_a_failure(gateway: Gateway, entry: EntryPoint) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, identity, key):
        caller: Final = McpCaller(gateway, key, entry, alias)
        nonce: Final = uuid.uuid4().hex
        for _ in range(2):
            repeated = caller.call(f"{alias}-consent", {"nonce": nonce}, _server_arg(entry, identity))
            assert repeated.status == 200 and repeated.error, repeated.raw
        control: Final = caller.call(f"{alias}-echo", {"nonce": "control"}, _server_arg(entry, identity))
        assert control.ok, control.raw
        assert _spend_rows(key, 3) == [("failure", nonce), ("failure", nonce), ("success", "control")]
        assert _upstream_nonces(upstream) == sorted(["control", nonce, nonce])


@pytest.mark.parametrize(
    ("code", "message"),
    ((-32001, "control error"), (-32602, "invalid params"), (-32603, "internal error"), (-32000, "server error")),
)
@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_other_upstream_jsonrpc_error_codes_stay_error_results(
    gateway: Gateway, entry: EntryPoint, code: int, message: str
) -> None:
    error: Final = {"code": code, "message": message, "data": {"elicitations": [{"mode": "url", "url": "x"}]}}
    with _granted(gateway, _failing_tool(error), _echo_tool()) as (_, alias, identity, key):
        caller: Final = McpCaller(gateway, key, entry, alias)
        reply: Final = caller.call(f"{alias}-consent", {"nonce": "n"}, _server_arg(entry, identity))
        assert reply.status == 200 and reply.text == f"MCPError: {message}", reply.raw
        assert reply.error == reply.text, reply.raw


LONG_URL: Final = "https://consent.example.invalid/authorize?state=" + "s" * 5000
ODD_ERRORS: Final[Mapping[str, Mapping[str, object]]] = {
    "data-missing": {"code": -32042, "message": MESSAGE},
    "data-empty-object": {"code": -32042, "message": MESSAGE, "data": {}},
    "elicitations-empty": {"code": -32042, "message": MESSAGE, "data": {"elicitations": []}},
    "data-string": {"code": -32042, "message": MESSAGE, "data": "see the consent page"},
    "data-int": {"code": -32042, "message": MESSAGE, "data": 7},
    "data-list": {"code": -32042, "message": MESSAGE, "data": ["a", "b"]},
    "url-5kb": {"code": -32042, "message": MESSAGE, "data": {"elicitations": [{"mode": "url", "url": LONG_URL}]}},
    "message-empty": {"code": -32042, "message": "", "data": {"elicitations": []}},
    "two-elicitations": {
        "code": -32042,
        "message": MESSAGE,
        "data": {
            "elicitations": [{"mode": "url", "url": "https://a.invalid"}, {"mode": "url", "url": "https://a.invalid"}]
        },
    },
}


@pytest.mark.parametrize("shape", tuple(ODD_ERRORS))
def test_url_elicitation_required_with_an_unusual_payload_is_relayed_exactly_as_the_upstream_sent_it(
    gateway: Gateway, shape: str
) -> None:
    with _granted(gateway, _failing_tool(ODD_ERRORS[shape]), _echo_tool()) as (_, alias, identity, key):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": "n"})
        assert outcome.status == 200 and outcome.error, outcome.raw
        assert json.loads(outcome.error) == ODD_ERRORS[shape], outcome.raw
        rest: Final = McpCaller(gateway, key, "rest", alias).call(f"{alias}-consent", {"nonce": "n"}, identity)
        assert rest.status == 200 and rest.error == f"MCPError: {ODD_ERRORS[shape]['message']}", rest.raw
        control: Final = caller.call(f"{alias}-echo", {"nonce": "control"})
        assert control.ok and control.text == "control", control.raw


def test_tool_result_that_only_mentions_the_elicitation_code_stays_an_error_result(gateway: Gateway) -> None:
    text: Final = "upstream said -32042 URL elicitation required"
    mention: Final = ScriptedTool("consent", lambda _: {"content": [{"type": "text", "text": text}], "isError": True})
    with _granted(gateway, mention) as (_, alias, _, key):
        outcome: Final = McpCaller(gateway, key, "mcp", alias).call(f"{alias}-consent", {"nonce": "n"})
        assert outcome.status == 200 and outcome.text == text and outcome.error == text, outcome.raw


def test_callers_without_access_never_reach_the_upstream_that_asks_for_url_elicitation(gateway: Gateway) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, _, key), gateway.scenario() as other:
        anonymous: Final = McpCaller(gateway, None, "mcp", alias).call(f"{alias}-consent", {"nonce": "anonymous"})
        assert anonymous.status == 401, anonymous.raw
        ungranted: Final = McpCaller(gateway, other.key(), "mcp", alias).call(
            f"{alias}-consent", {"nonce": "ungranted"}
        )
        assert ungranted.error is not None and "-32042" not in ungranted.raw, ungranted.raw
        assert "consent.example.invalid" not in anonymous.raw + ungranted.raw
        granted: Final = McpCaller(gateway, key, "mcp", alias).call(f"{alias}-consent", {"nonce": "granted"})
        assert granted.error and json.loads(granted.error) == _elicitation("granted"), granted.raw
        assert _upstream_nonces(upstream) == ["granted"]


def test_retrying_after_consent_succeeds_and_the_gateway_never_retries_the_elicitation_itself(
    gateway: Gateway,
) -> None:
    attempts: Final = itertools.count(1)

    def respond(params: JsonRpc) -> JsonRpc | JsonRpcFailure:
        return JsonRpcFailure(_elicitation(_nonce(params))) if next(attempts) <= 2 else text_result("consented")

    with _granted(gateway, ScriptedTool("consent", respond)) as (upstream, alias, _, key):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        nonce: Final = uuid.uuid4().hex
        outcomes: Final = tuple(caller.call(f"{alias}-consent", {"nonce": nonce}) for _ in range(3))
        assert [json.loads(outcome.error) for outcome in outcomes[:2] if outcome.error] == [_elicitation(nonce)] * 2
        assert outcomes[2].ok and outcomes[2].text == "consented", outcomes[2].raw
        assert _upstream_nonces(upstream) == [nonce] * 3


def test_url_elicitation_error_echoes_the_callers_string_request_id(gateway: Gateway) -> None:
    with _granted(gateway, _consent_tool()) as (_, alias, _, key):
        headers: Final = {"x-litellm-api-key": key, "Accept": "application/json, text/event-stream"}
        opened: Final = gateway.client.post(
            "/mcp", headers=headers, json={"jsonrpc": "2.0", "id": "open", "method": "initialize", "params": INITIALIZE}
        )
        assert opened.status_code == 200, opened.text
        nonce: Final = uuid.uuid4().hex
        reply: Final = gateway.client.post(
            "/mcp",
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": "call-7",
                "method": "tools/call",
                "params": {"name": f"{alias}-consent", "arguments": {"nonce": nonce}},
            },
        )
        assert _parse_rpc_body(reply) == {"jsonrpc": "2.0", "id": "call-7", "error": _elicitation(nonce)}, reply.text


def _burst_call(
    caller: McpCaller, tool: str, nonce: str, identity: str
) -> tuple[str, str, str, "str | None", "str | None"]:
    outcome: Final = caller.call(tool, {"nonce": nonce}, _server_arg(caller.entry, identity))
    return caller.entry, tool.rsplit("-", 1)[1], nonce, outcome.error, outcome.text


def _expected(entry: str, tool: str, nonce: str, error: str | None, text: str | None) -> bool:
    if tool == "echo":
        return error is None and text == nonce
    if tool == "outage":
        return error is not None and "-32042" not in error and "consent.example.invalid" not in error
    if entry == "rest":
        return error == FLATTENED
    return error is not None and json.loads(error) == _elicitation(nonce)


BURST: Final = 30


def test_concurrent_burst_across_two_proxies_gives_each_caller_its_own_elicitation_and_logs_each_call_once(
    gateway: Gateway, peer: Gateway
) -> None:
    outage: Final = ScriptedTool("outage", lambda _: Reply(status=503, body=b'{"error": "scripted outage"}'))
    with _granted(gateway, _consent_tool(), _echo_tool(), outage) as (upstream, alias, identity, key):
        callers: Final = (
            McpCaller(gateway, key, "mcp", alias),
            McpCaller(peer, key, "server_mcp", alias),
            McpCaller(gateway, key, "rest", alias),
            McpCaller(peer, key, "mcp", alias),
            McpCaller(gateway, key, "server_mcp", alias),
            McpCaller(peer, key, "rest", alias),
            McpCaller(gateway, key, "root", alias),
        )
        tools: Final = (f"{alias}-consent", f"{alias}-echo", f"{alias}-consent", f"{alias}-outage", f"{alias}-consent")
        nonces: Final = tuple(uuid.uuid4().hex for _ in range(BURST))
        with ThreadPoolExecutor(max_workers=BURST) as pool:
            results: Final = tuple(
                pool.map(
                    _burst_call, itertools.cycle(callers), itertools.cycle(tools), nonces, itertools.repeat(identity)
                )
            )
        assert [result for result in results if not _expected(*result)] == [], results
        assert {(entry, tool) for entry, tool, *_ in results} >= {
            ("mcp", "consent"),
            ("server_mcp", "consent"),
            ("rest", "consent"),
            ("root", "consent"),
        }
        assert _upstream_nonces(upstream) == sorted(nonces)
        assert _spend_rows(key, BURST) == sorted(
            ("success" if tool == "echo" else "failure", nonce) for _, tool, nonce, _, _ in results
        )


def _fresh_connection_call(proxy_url: str, key: str, name: str, nonce: str) -> Mapping[str, object] | None:
    try:
        with httpx.Client(base_url=proxy_url, trust_env=False, timeout=60) as client:
            reply: Final = client.post(
                "/mcp",
                headers={"x-litellm-api-key": key, "Accept": "application/json, text/event-stream"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": {"nonce": nonce}},
                },
            )
            return _parse_rpc_body(reply)
    except httpx.TransportError:
        return None


HELD_PER_WORKER: Final = 10


@pytest.mark.timeout(2 * graceful_stop_seconds() + 240)
def test_worker_killed_mid_burst_leaves_the_sibling_and_the_replacement_relaying_url_elicitation(
    gateway: Gateway, tmp_path: Path
) -> None:
    gate: Final = threading.Event()
    arrived: Final[SimpleQueue[str]] = SimpleQueue()

    def held(params: JsonRpc) -> JsonRpcFailure:
        arrived.put(_nonce(params))
        assert gate.wait(120), "the burst was never released"
        return JsonRpcFailure(_elicitation(_nonce(params)))

    with (
        _granted(gateway, ScriptedTool("consent", held)) as (_, alias, _, key),
        owned_proxy_process(gateway, tmp_path, {}, workers=2, healthcheck=WorkerHealthcheck.LENIENT) as owned,
        ThreadPoolExecutor(max_workers=2 * HELD_PER_WORKER + 1) as pool,
    ):
        try:
            victim, survivor = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=30))
            proxy_url: Final = str(owned.gateway.client.base_url)
            name: Final = f"{alias}-consent"
            victim_nonces: Final = tuple(uuid.uuid4().hex for _ in range(HELD_PER_WORKER))
            survivor_nonces: Final = tuple(uuid.uuid4().hex for _ in range(HELD_PER_WORKER))
            with owned.suspended(survivor):
                on_victim: Final = tuple(
                    pool.submit(_fresh_connection_call, proxy_url, key, name, nonce) for nonce in victim_nonces
                )
                eventually(arrived.qsize, lambda count: count == HELD_PER_WORKER, seconds=120)
            with owned.suspended(victim):
                on_survivor: Final = tuple(
                    pool.submit(_fresh_connection_call, proxy_url, key, name, nonce) for nonce in survivor_nonces
                )
                eventually(arrived.qsize, lambda count: count == 2 * HELD_PER_WORKER, seconds=120)
            with owned.replacing_worker(victim) as kill:
                kill()
                gate.set()
                dropped: Final = tuple(future.result(timeout=120) for future in on_victim)
                relayed: Final = tuple(future.result(timeout=120) for future in on_survivor)
            assert dropped == (None,) * HELD_PER_WORKER, dropped
            assert list(relayed) == [
                {"jsonrpc": "2.0", "id": 1, "error": _elicitation(nonce)} for nonce in survivor_nonces
            ], relayed
            replacement: Final = next(iter(owned.worker_pids() - {survivor}))
            assert replacement != victim
            with owned.suspended(survivor):
                after: Final = pool.submit(_fresh_connection_call, proxy_url, key, name, "replacement").result(120)
            assert after == {"jsonrpc": "2.0", "id": 1, "error": _elicitation("replacement")}, after
        finally:
            gate.set()


def test_upstream_outage_is_an_error_result_and_url_elicitation_is_relayed_again_once_it_is_back(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        with scripted_peer(_consent_tool()) as first:
            identity: Final = register_mcp(scenario, first, alias)
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            caller: Final = McpCaller(gateway, key, "mcp", alias)
            before: Final = caller.call(f"{alias}-consent", {"nonce": "before"})
            assert before.error and json.loads(before.error) == _elicitation("before"), before.raw
        down: Final = _parse_rpc_body(
            caller.rpc("tools/call", {"name": f"{alias}-consent", "arguments": {"nonce": "x"}})
        )
        assert down is not None and "error" not in down and down["result"]["isError"] is True, down
        with scripted_peer(_consent_tool()) as replacement:
            edited: Final = gateway.request(
                "PUT",
                "/v1/mcp/server",
                {"server_id": identity, "server_name": alias, "alias": alias, **replacement.registration()},
            )
            assert edited.status_code in (200, 202), edited.text
            back: Final = eventually(
                lambda: _parse_rpc_body(
                    caller.rpc("tools/call", {"name": f"{alias}-consent", "arguments": {"nonce": "after"}})
                ),
                lambda reply: reply is not None and "error" in reply,
                seconds=40,
                return_last_on_timeout=True,
            )
            assert back == {"jsonrpc": "2.0", "id": 1, "error": _elicitation("after")}, back


def test_held_url_elicitation_reply_does_not_block_other_calls_and_is_relayed_when_released(
    gateway: Gateway,
) -> None:
    gate: Final = threading.Event()
    arrived: Final = threading.Event()

    def held(params: JsonRpc) -> JsonRpcFailure:
        arrived.set()
        assert gate.wait(120), "the held reply was never released"
        return JsonRpcFailure(_elicitation(_nonce(params)))

    with (
        _granted(gateway, ScriptedTool("consent", held), _echo_tool()) as (_, alias, _, key),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        try:
            caller: Final = McpCaller(gateway, key, "mcp", alias)
            slow: Final = pool.submit(caller.call, f"{alias}-consent", {"nonce": "held"})
            assert arrived.wait(60), "the held call never reached the upstream"
            fast: Final = tuple(caller.call(f"{alias}-echo", {"nonce": f"fast-{index}"}) for index in range(5))
            assert [outcome.text for outcome in fast] == [f"fast-{index}" for index in range(5)]
            assert not slow.done()
            gate.set()
            released: Final = slow.result(timeout=60)
            assert released.error and json.loads(released.error) == _elicitation("held"), released.raw
        finally:
            gate.set()


def _proxy_call(gateway: Gateway, key: str, name: str, arguments: JsonRpc) -> Mapping[str, object]:
    reply: Final = gateway.client.post(
        "/mcp/proxy",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        headers={"x-litellm-api-key": key, "Accept": "application/json, text/event-stream"},
    )
    assert reply.status_code == 200, reply.text
    body: Final = _parse_rpc_body(reply)
    assert body is not None, reply.text
    return body


def _proxy_tool_id(gateway: Gateway, key: str, name: str) -> str:
    found: Final = _proxy_call(gateway, key, "search_tools", {"query": name})
    hits: Final = json.loads(found["result"]["content"][0]["text"])
    return next(str(hit["tool_id"]) for hit in hits if hit["name"] == name)


def test_url_elicitation_through_the_proxy_call_tool_is_relayed_and_logged_as_one_failure(gateway: Gateway) -> None:
    with _granted(gateway, _consent_tool(), _echo_tool()) as (upstream, alias, _, key):
        consent: Final = _proxy_tool_id(gateway, key, f"{alias}-consent")
        echo: Final = _proxy_tool_id(gateway, key, f"{alias}-echo")
        nonce: Final = uuid.uuid4().hex
        relayed: Final = _proxy_call(gateway, key, "call_tool", {"tool_id": consent, "arguments": {"nonce": nonce}})
        assert relayed == {"jsonrpc": "2.0", "id": 1, "error": _elicitation(nonce)}, relayed
        control: Final = _proxy_call(gateway, key, "call_tool", {"tool_id": echo, "arguments": {"nonce": "control"}})
        assert control["result"]["content"][0]["text"] == "control", control
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                (sha256(key.encode()).hexdigest(), "call_mcp_tool"),
            ),
            lambda found: "success" in [row["status"] for row in found],
            seconds=70,
            return_last_on_timeout=True,
        )
        assert sorted(str(row["status"]) for row in rows) == ["failure", "success"], rows
        assert _upstream_nonces(upstream) == sorted(["control", nonce])


def test_failure_log_of_a_masked_elicitation_never_stores_the_unmasked_message(gateway: Gateway) -> None:
    email: Final = f"john.doe.{uuid.uuid4().hex[:8]}@example.com"
    with (
        _output_guardrail(gateway, email, "MASK"),
        _granted(gateway, *_addressed_tools(email, "none")) as (_, alias, _identity, key),
    ):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        control: Final = eventually(
            lambda: caller.call(f"{alias}-whoami", {}),
            lambda outcome: outcome.text == f"Signed in as {REDACTED}.",
            seconds=40,
            return_last_on_timeout=True,
        )
        assert control.text == f"Signed in as {REDACTED}.", control.raw
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce})
        assert outcome.error is not None and json.loads(outcome.error) == _addressed_elicitation(
            nonce, REDACTED, "none"
        ), outcome.raw
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT metadata::text AS logged FROM "LiteLLM_SpendLogs"'
                " WHERE api_key = %s AND call_type = %s AND status = %s",
                (sha256(key.encode()).hexdigest(), "call_mcp_tool", "failure"),
            ),
            lambda found: len(found) >= 1,
            seconds=70,
            return_last_on_timeout=True,
        )
        assert [(REDACTED in str(row["logged"]), email in str(row["logged"])) for row in rows] == [(True, False)], rows


BLOCKED_NONCE: Final = "refuse-this-call"
REWRITTEN: Final = "[REWRITTEN]"
_HOOKS_CODE: Final = """\
import asyncio

import httpx
from fastapi import HTTPException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from mcp.types import TextContent


class SlowRefusal(CustomGuardrail):
    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        await asyncio.sleep(0.3)
        if "{blocked}" in str(data.get("mcp_arguments")):
            raise HTTPException(status_code=400, detail="refused by the during-call guardrail")


class OutputCounter(CustomGuardrail):
    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post("{sink}/guardrail", json={{"texts": inputs.get("texts")}})
        return inputs


class Rewriter(CustomLogger):
    async def async_post_mcp_tool_call_hook(self, kwargs, response_obj, start_time, end_time):
        texts = [block.text for block in response_obj.mcp_tool_call_response]
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post("{sink}/hook", json={{"texts": texts}})
        response_obj.mcp_tool_call_response = [
            TextContent(type="text", text=text.replace("{rewrite}", "{rewritten}")) for text in texts
        ]
        return response_obj


rewriter = Rewriter()
"""
REWRITE: Final = "rewrite.me@example.com"


@pytest.fixture(scope="module")
def hooked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire]]:
    directory: Final = tmp_path_factory.mktemp("elicitation-hooks")
    with wire_server(lambda _: Reply()) as sink:
        (directory / "elicitation_hooks.py").write_text(
            _HOOKS_CODE.format(blocked=BLOCKED_NONCE, sink=sink.url, rewrite=REWRITE, rewritten=REWRITTEN)
        )
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        path: Final = directory / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "guardrails": [
                        {
                            "guardrail_name": "slow-refusal",
                            "litellm_params": {
                                "guardrail": "elicitation_hooks.SlowRefusal",
                                "mode": "during_mcp_call",
                                "default_on": True,
                            },
                        },
                        {
                            "guardrail_name": "output-counter",
                            "litellm_params": {
                                "guardrail": "elicitation_hooks.OutputCounter",
                                "mode": "post_mcp_call",
                                "default_on": True,
                            },
                        },
                    ],
                    "litellm_settings": {**config["litellm_settings"], "callbacks": ["elicitation_hooks.rewriter"]},
                }
            )
        )
        with (
            gateway_from_environment() as gateway,
            owned_proxy(gateway, directory, {"MAX_STRING_LENGTH_PROMPT_IN_DB": "100000"}, config=path) as candidate,
        ):
            yield candidate, sink


def _sunk_texts(sink: Wire, marker: str) -> dict[str, list[list[str]]]:
    seen: Final = [(request.target, json.loads(request.body)["texts"]) for request in sink.drain()]
    return {
        stage: [texts for target, texts in seen if target == f"/{stage}" and marker in "".join(texts)]
        for stage in ("guardrail", "hook")
    }


def test_blocking_during_call_guardrail_slower_than_the_elicitation_still_blocks_the_caller(
    hooked: tuple[Gateway, Wire],
) -> None:
    gateway: Final = hooked[0]
    with _granted(gateway, _consent_tool()) as (upstream, alias, _, key):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        blocked: Final = caller.call(f"{alias}-consent", {"nonce": BLOCKED_NONCE})
        assert (blocked.status, blocked.text, blocked.error) == (
            200,
            "Error: refused by the during-call guardrail",
            "Error: refused by the during-call guardrail",
        ), blocked.raw
        assert "consent.example.invalid" not in blocked.raw, blocked.raw
        nonce: Final = uuid.uuid4().hex
        relayed: Final = caller.call(f"{alias}-consent", {"nonce": nonce})
        assert relayed.error is not None and json.loads(relayed.error) == _elicitation(nonce), relayed.raw
        assert _upstream_nonces(upstream) == sorted([BLOCKED_NONCE, nonce])


def test_post_tool_call_logger_hook_rewrites_the_relayed_elicitation_messages_once_and_not_the_url(
    hooked: tuple[Gateway, Wire],
) -> None:
    gateway, sink = hooked
    with _granted(gateway, *_addressed_tools(REWRITE)) as (_, alias, _identity, key):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        nonce: Final = uuid.uuid4().hex
        sink.drain()
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce})
        assert outcome.error is not None and json.loads(outcome.error) == _addressed_elicitation(
            nonce, REWRITTEN, REWRITE
        ), outcome.raw
        assert _sunk_texts(sink, "Authorize to continue")["hook"] == [
            [f"Authorize to continue for {REWRITE}.", f"Sign in as {REWRITE}."]
        ]


SurfaceCall = Literal["rest", "rest_virtual", "mcp", "mcp_virtual"]


@pytest.mark.parametrize("surface", ("rest", "rest_virtual", "mcp", "mcp_virtual"))
def test_output_guardrail_and_post_tool_call_hook_each_see_one_url_elicitation_once_on_every_surface(
    hooked: tuple[Gateway, Wire], surface: SurfaceCall
) -> None:
    gateway, sink = hooked
    email: Final = f"once.{uuid.uuid4().hex[:8]}@example.com"
    with scripted_peer(*_addressed_tools(email)) as upstream, gateway.scenario() as scenario:
        alias: Final = "elicit" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity], "mcp_tool_search_enabled": True})
        entry: Final[EntryPoint] = "rest" if surface.startswith("rest") else "mcp"
        caller: Final = McpCaller(gateway, key, entry, alias)
        nonce: Final = uuid.uuid4().hex
        sink.drain()
        outcome: Final = (
            caller.call("mcp_tool_call", {"tool_name": f"{alias}-consent", "arguments": {"nonce": nonce}})
            if surface.endswith("virtual")
            else caller.call(f"{alias}-consent", {"nonce": nonce}, _server_arg(entry, identity))
        )
        elicitation: Final = _addressed_elicitation(nonce, email, email)
        flattened: Final = [f"MCPError: {elicitation['message']}"]
        relayed: Final = [str(elicitation["message"]), f"Sign in as {email}."]
        assert outcome.status == 200 and outcome.error is not None, outcome.raw
        assert (
            (outcome.text, outcome.error) == (flattened[0], flattened[0])
            if entry == "rest"
            else json.loads(outcome.error) == elicitation
        ), outcome.raw
        assert _sunk_texts(sink, email) == {
            "guardrail": [flattened if entry == "rest" else relayed],
            "hook": [flattened if entry == "rest" else relayed],
        }


def test_output_guardrail_and_post_tool_call_hook_each_see_one_url_elicitation_once_through_the_proxy_call_tool(
    hooked: tuple[Gateway, Wire],
) -> None:
    gateway, sink = hooked
    email: Final = f"once.{uuid.uuid4().hex[:8]}@example.com"
    with _granted(gateway, *_addressed_tools(email)) as (_, alias, _identity, key):
        consent: Final = _proxy_tool_id(gateway, key, f"{alias}-consent")
        nonce: Final = uuid.uuid4().hex
        sink.drain()
        relayed: Final = _proxy_call(gateway, key, "call_tool", {"tool_id": consent, "arguments": {"nonce": nonce}})
        elicitation: Final = _addressed_elicitation(nonce, email, email)
        assert relayed == {"jsonrpc": "2.0", "id": 1, "error": elicitation}, relayed
        texts: Final = [str(elicitation["message"]), f"Sign in as {email}."]
        assert _sunk_texts(sink, email) == {"guardrail": [texts], "hook": [texts]}


def test_failure_log_of_an_elicitation_refused_by_an_output_guardrail_never_stores_the_upstream_message(
    hooked: tuple[Gateway, Wire],
) -> None:
    gateway: Final = hooked[0]
    email: Final = f"john.doe.{uuid.uuid4().hex[:8]}@example.com"
    verb: Final = "Authorize" + uuid.uuid4().hex[:8]
    with (
        _output_guardrail(gateway, verb, "MASK"),
        _output_guardrail(gateway, email, "BLOCK"),
        _granted(gateway, *_addressed_tools(email, "none", verb)) as (upstream, alias, _, key),
    ):
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        refused: Final = f"Error: Content blocked: keyword '{email}' detected"
        control: Final = eventually(
            lambda: caller.call(f"{alias}-whoami", {}),
            lambda outcome: outcome.error == refused,
            seconds=40,
            return_last_on_timeout=True,
        )
        assert control.error == refused, control.raw
        upstream.drain()
        since: Final = len(gateway.request("GET", "/spend/logs", params={"api_key": key}).json())
        nonce: Final = uuid.uuid4().hex
        outcome: Final = caller.call(f"{alias}-consent", {"nonce": nonce})
        assert (outcome.text, outcome.error) == (refused, refused), outcome.raw
        assert _upstream_nonces(upstream) == [nonce]
        logged: Final = eventually(
            lambda: gateway.request("GET", "/spend/logs", params={"api_key": key}),
            lambda listed: nonce in listed.text and len(listed.json()) > since,
            seconds=70,
            return_last_on_timeout=True,
        )
        rows: Final = [row for row in logged.json() if nonce in json.dumps(row)]
        assert [(row["status"], f"{verb} to continue for" in json.dumps(row)) for row in rows] == [
            ("failure", False)
        ], rows
