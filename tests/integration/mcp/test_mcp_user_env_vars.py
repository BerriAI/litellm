import os
import signal
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import psycopg
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import advisory_lock_key, advisory_waiters, legacy_advisory_lock_key, read_rows
from integration._support.mcp import McpPeer, call_tool, mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy_process
from pydantic import JsonValue, TypeAdapter

TOKEN: Final = "USER_TOKEN"
WORKSPACE: Final = "WORKSPACE"
METHODS: Final = ("GET", "POST", "DELETE")


@dataclass(frozen=True, slots=True)
class UpstreamCall:
    body: dict[str, JsonValue]
    headers: dict[bytes, bytes]


UPSTREAM_CALLS: Final = TypeAdapter(tuple[UpstreamCall, ...])
STATUS_LISTING: Final = TypeAdapter(list[dict[str, JsonValue]])
JSON_BODY: Final = TypeAdapter(dict[str, JsonValue])


def body(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_BODY.validate_json(response.content)


def register_user_var_server(scenario: Scenario, peer: McpPeer, *names: str) -> str:
    return register_mcp(
        scenario,
        peer,
        "integration" + uuid.uuid4().hex,
        auth_type="none",
        env_vars=[{"name": name, "scope": "user", "description": f"per-user {name}"} for name in names],
        static_headers={
            "Authorization": f"Bearer ${{{TOKEN}}}",
            **({"X-Workspace": f"${{{WORKSPACE}}}"} if WORKSPACE in names else {}),
        },
    )


def grants(*identities: str) -> JsonValue:
    return {"mcp_servers": list(identities)}


def user_key(scenario: Scenario, identity: str) -> str:
    return scenario.key(user_id=scenario.user(), object_permission=grants(identity))


def env_status(gateway: Gateway, key: str, identity: str) -> httpx.Response:
    return gateway.request("GET", f"/v1/mcp/server/{identity}/user-env-vars", key=key)


def store(gateway: Gateway, key: str, identity: str, values: Mapping[str, str]) -> httpx.Response:
    return gateway.request("POST", f"/v1/mcp/server/{identity}/user-env-vars", {"values": dict(values)}, key=key)


def clear(gateway: Gateway, key: str, identity: str) -> httpx.Response:
    return gateway.request("DELETE", f"/v1/mcp/server/{identity}/user-env-vars", key=key)


def set_names(response: httpx.Response) -> dict[str, bool]:
    assert response.status_code == 200, response.text
    status: Final = body(response)
    assert isinstance(status["required"], list)
    return {
        string_value(object_value(spec)["name"]): object_value(spec)["is_set"] is True for spec in status["required"]
    }


def tool_calls(peer: McpPeer) -> tuple[UpstreamCall, ...]:
    return tuple(
        call for call in UPSTREAM_CALLS.validate_python(peer.drain()) if call.body.get("method") == "tools/call"
    )


def add_upstream_headers(gateway: Gateway, peer: McpPeer, key: str, identity: str, a: int = 2) -> dict[bytes, bytes]:
    peer.drain()
    response: Final = call_tool(gateway, key, identity, tool_names(gateway, key, identity)["add"], {"a": a, "b": 3})
    assert response.status_code == 200, response.text
    calls: Final = tool_calls(peer)
    assert len(calls) == 1, calls
    return calls[0].headers


def add_upstream_authorization(gateway: Gateway, peer: McpPeer, key: str, identity: str) -> bytes:
    return add_upstream_headers(gateway, peer, key, identity)[b"authorization"]


def list_tools_status(target: Gateway, key: str, identity: str) -> int:
    return target.client.get(
        "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": identity}
    ).status_code


def wait_for_tools(target: Gateway, key: str, identity: str) -> dict[str, str]:
    eventually(lambda: list_tools_status(target, key, identity), lambda status: status == 200, seconds=60)
    return eventually(lambda: tool_names(target, key, identity), lambda names: "add" in names, seconds=60)


def assert_forwarded_eventually(target: Gateway, upstream: McpPeer, key: str, identity: str, expected: bytes) -> None:
    observed: Final = eventually(
        lambda: add_upstream_authorization(target, upstream, key, identity), lambda value: value == expected, seconds=75
    )
    assert observed == expected


def assert_precondition_failed(gateway: Gateway, key: str, identity: str, *missing: str) -> None:
    response: Final = call_tool(gateway, key, identity, tool_names(gateway, key, identity)["add"], {"a": 2, "b": 3})
    assert response.status_code == 412, response.text
    detail: Final = object_value(body(response)["detail"])
    assert detail["error"] == "missing_user_env_vars"
    assert detail["server_id"] == identity
    assert isinstance(detail["missing"], list)
    assert sorted(string_value(name) for name in detail["missing"]) == sorted(missing)
    assert string_value(detail["setup_url"]).endswith(f"fill_env_vars={identity}")


def stored_user_ids(identity: str) -> tuple[JsonValue, ...]:
    return tuple(
        row["user_id"]
        for row in read_rows('SELECT user_id FROM "LiteLLM_MCPUserEnvVars" WHERE server_id = %s', (identity,))
    )


def missing_count(response: httpx.Response) -> JsonValue:
    return body(response)["missing_count"]


def test_stored_value_is_forwarded_rotated_and_cleared(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, peer, TOKEN)
        key: Final = user_key(scenario, identity)
        before: Final = env_status(gateway, key, identity)
        assert set_names(before) == {TOKEN: False}
        assert missing_count(before) == 1
        assert string_value(body(before)["setup_url"]).endswith(f"fill_env_vars={identity}")
        assert_precondition_failed(gateway, key, identity, TOKEN)
        first: Final = store(gateway, key, identity, {TOKEN: "first-secret"})
        assert set_names(first) == {TOKEN: True}
        assert missing_count(first) == 0
        assert add_upstream_authorization(gateway, peer, key, identity) == b"Bearer first-secret"
        rotated: Final = store(gateway, key, identity, {TOKEN: "second-secret"})
        assert set_names(rotated) == {TOKEN: True}
        assert add_upstream_authorization(gateway, peer, key, identity) == b"Bearer second-secret"
        assert len(stored_user_ids(identity)) == 1
        cleared: Final = clear(gateway, key, identity)
        assert set_names(cleared) == {TOKEN: False}
        assert missing_count(cleared) == 1
        assert stored_user_ids(identity) == ()
        assert set_names(env_status(gateway, key, identity)) == {TOKEN: False}
        assert_precondition_failed(gateway, key, identity, TOKEN)
        assert set_names(clear(gateway, key, identity)) == {TOKEN: False}


def test_store_merges_per_variable_and_drops_undeclared_or_empty_values(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, peer, TOKEN, WORKSPACE)
        key: Final = user_key(scenario, identity)
        assert set_names(env_status(gateway, key, identity)) == {TOKEN: False, WORKSPACE: False}
        assert_precondition_failed(gateway, key, identity, TOKEN, WORKSPACE)
        partial: Final = store(gateway, key, identity, {TOKEN: "tok", "NOT_DECLARED": "x", "": "y"})
        assert set_names(partial) == {TOKEN: True, WORKSPACE: False}
        assert missing_count(partial) == 1
        assert_precondition_failed(gateway, key, identity, WORKSPACE)
        long_value: Final = "w" * 5120
        complete: Final = store(gateway, key, identity, {WORKSPACE: long_value})
        assert set_names(complete) == {TOKEN: True, WORKSPACE: True}
        forwarded: Final = add_upstream_headers(gateway, peer, key, identity)
        assert forwarded[b"authorization"] == b"Bearer tok"
        assert forwarded[b"x-workspace"] == long_value.encode()
        kept: Final = store(gateway, key, identity, {TOKEN: "", WORKSPACE: ""})
        assert set_names(kept) == {TOKEN: True, WORKSPACE: True}
        assert add_upstream_authorization(gateway, peer, key, identity) == b"Bearer tok"
        assert set_names(store(gateway, key, identity, {TOKEN: "tok"})) == {TOKEN: True, WORKSPACE: True}
        assert len(stored_user_ids(identity)) == 1


def test_malformed_bodies_missing_users_and_foreign_servers_are_rejected(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, peer, TOKEN)
        other: Final = register_user_var_server(scenario, peer, TOKEN)
        key: Final = user_key(scenario, identity)
        userless: Final = scenario.key(object_permission=grants(identity))
        path: Final = f"/v1/mcp/server/{identity}/user-env-vars"
        payload: Final[dict[str, JsonValue]] = {"values": {TOKEN: "x"}}
        malformed: Final[tuple[dict[str, JsonValue], ...]] = ({"values": {TOKEN: 7}}, {"values": ["a"]}, {})
        assert [gateway.request("POST", path, bad, key=key).status_code for bad in malformed] == [422, 422, 422]
        assert set_names(env_status(gateway, key, identity)) == {TOKEN: False}
        assert [gateway.client.request(method, path, json=payload).status_code for method in METHODS] == [401, 401, 401]
        no_user: Final = tuple(gateway.request(method, path, payload, key=userless) for method in METHODS)
        assert [response.status_code for response in no_user] == [400, 400, 400], [r.text for r in no_user]
        assert [object_value(body(r)["detail"])["error"] for r in no_user] == ["User ID not found in token"] * 3
        foreign: Final = f"/v1/mcp/server/{other}/user-env-vars"
        assert [gateway.request(method, foreign, payload, key=key).status_code for method in METHODS] == [403, 403, 403]
        unknown: Final = f"/v1/mcp/server/{uuid.uuid4()}/user-env-vars"
        assert [gateway.request(method, unknown, payload).status_code for method in METHODS] == [404, 404, 404]
        assert stored_user_ids(identity) == () and stored_user_ids(other) == ()


def test_status_list_keeps_fully_set_servers_and_is_scoped_to_the_caller(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        per_user: Final = register_user_var_server(scenario, peer, TOKEN)
        global_only: Final = register_mcp(
            scenario,
            peer,
            "integration" + uuid.uuid4().hex,
            env_vars=[{"name": "GLOBAL_TOKEN", "scope": "global", "description": "shared"}],
        )
        plain: Final = register_mcp(scenario, peer, "integration" + uuid.uuid4().hex)
        first_user: Final = scenario.key(
            user_id=scenario.user(), object_permission=grants(per_user, global_only, plain)
        )
        second_user: Final = scenario.key(
            user_id=scenario.user(), object_permission=grants(per_user, global_only, plain)
        )

        def listing(key: str) -> dict[str, JsonValue]:
            response: Final = gateway.request("GET", "/v1/mcp/user-env-vars/status", key=key)
            assert response.status_code == 200, response.text
            return {
                string_value(entry["server_id"]): entry["missing_count"]
                for entry in STATUS_LISTING.validate_json(response.content)
                if entry["server_id"] in {per_user, global_only, plain}
            }

        assert listing(first_user) == {per_user: 1}
        assert set_names(store(gateway, first_user, per_user, {TOKEN: "mine"})) == {TOKEN: True}
        assert listing(first_user) == {per_user: 0}
        assert listing(second_user) == {per_user: 1}
        assert set_names(env_status(gateway, second_user, per_user)) == {TOKEN: False}
        assert add_upstream_authorization(gateway, peer, first_user, per_user) == b"Bearer mine"
        assert_precondition_failed(gateway, second_user, per_user, TOKEN)
        assert set_names(clear(gateway, second_user, per_user)) == {TOKEN: False}
        assert listing(first_user) == {per_user: 0}
        assert add_upstream_authorization(gateway, peer, first_user, per_user) == b"Bearer mine"


def test_store_and_clear_on_one_process_are_honored_by_the_other(gateway: Gateway, peer: Gateway) -> None:
    with mcp_peer() as upstream, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, upstream, TOKEN)
        key: Final = user_key(scenario, identity)
        assert_precondition_failed(gateway, key, identity, TOKEN)
        wait_for_tools(peer, key, identity)
        assert_precondition_failed(peer, key, identity, TOKEN)
        assert set_names(store(gateway, key, identity, {TOKEN: "from-a"})) == {TOKEN: True}
        assert set_names(env_status(peer, key, identity)) == {TOKEN: True}
        assert add_upstream_authorization(peer, upstream, key, identity) == b"Bearer from-a"
        assert set_names(store(peer, key, identity, {TOKEN: "from-b"})) == {TOKEN: True}
        assert_forwarded_eventually(gateway, upstream, key, identity, b"Bearer from-b")
        assert set_names(clear(gateway, key, identity)) == {TOKEN: False}
        assert set_names(env_status(peer, key, identity)) == {TOKEN: False}
        assert stored_user_ids(identity) == ()

        def peer_status() -> int:
            names: Final = tool_names(peer, key, identity)
            return call_tool(peer, key, identity, names["add"], {"a": 1, "b": 1}).status_code

        assert eventually(peer_status, lambda code: code == 412, seconds=75) == 412
        assert_precondition_failed(gateway, key, identity, TOKEN)


@pytest.mark.timeout(240)
def test_concurrent_users_across_processes_never_leak_and_survive_a_killed_process(
    gateway: Gateway, peer: Gateway, tmp_path: Path
) -> None:
    with mcp_peer() as upstream, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, upstream, TOKEN)
        users: Final = tuple(scenario.user() for _ in range(4))
        keys: Final = {user: scenario.key(user_id=user, object_permission=grants(identity)) for user in users}
        assert [set_names(store(gateway, keys[user], identity, {TOKEN: f"seed-{user}"})) for user in users] == [
            {TOKEN: True}
        ] * len(users)
        names: Final = wait_for_tools(gateway, keys[users[0]], identity)
        wait_for_tools(peer, keys[users[0]], identity)

        def operation(target: Gateway, user: str, index: int) -> httpx.Response:
            if index % 4 == 1:
                return env_status(target, keys[user], identity)
            if index % 4 == 2:
                return call_tool(target, keys[user], identity, names["add"], {"a": users.index(user), "b": 0})
            return store(target, keys[user], identity, {TOKEN: f"{user}-{index}"})

        def outcome(targets: tuple[Gateway, ...], job: tuple[str, int]) -> tuple[int, int]:
            return job[1], operation(targets[job[1] % len(targets)], job[0], job[1]).status_code

        def burst(pool: ThreadPoolExecutor, targets: tuple[Gateway, ...]) -> tuple[tuple[int, int], ...]:
            jobs: Final = tuple((user, index) for user in users for index in range(6))
            return tuple(pool.map(partial(outcome, targets), jobs))

        def allowed_authorizations(item: UpstreamCall) -> tuple[str, frozenset[bytes]]:
            arguments: Final = object_value(object_value(item.body["params"])["arguments"])
            owner: Final = users[int(string_value(str(arguments["a"])))]
            return owner, frozenset(
                {f"Bearer seed-{owner}".encode()} | {f"Bearer {owner}-{i}".encode() for i in range(6)}
            )

        with owned_proxy_process(gateway, tmp_path, {}) as doomed, ThreadPoolExecutor(max_workers=8) as pool:
            wait_for_tools(doomed.gateway, keys[users[0]], identity)
            upstream.drain()
            outcomes: Final = burst(pool, (gateway, peer, doomed.gateway))
            assert all(code in {200, 412} for _, code in outcomes), outcomes
            assert all(code == 200 for index, code in outcomes if index % 4 != 2), outcomes
            doomed.process.send_signal(signal.SIGKILL)
            doomed.process.wait(timeout=10)
            after_kill: Final = burst(pool, (gateway, peer))
            assert all(code in {200, 412} for _, code in after_kill), after_kill
            assert all(code == 200 for index, code in after_kill if index % 4 != 2), after_kill
        forwarded: Final = tool_calls(upstream)
        assert forwarded
        leaked: Final = tuple(
            (owner, item.headers[b"authorization"])
            for item in forwarded
            for owner, allowed in (allowed_authorizations(item),)
            if item.headers[b"authorization"] not in allowed
        )
        assert leaked == ()
        assert [set_names(env_status(gateway, keys[user], identity)) for user in users] == [{TOKEN: True}] * len(users)
        assert [set_names(env_status(peer, keys[user], identity)) for user in users] == [{TOKEN: True}] * len(users)
        assert [set_names(store(gateway, keys[user], identity, {TOKEN: f"final-{user}"})) for user in users] == [
            {TOKEN: True}
        ] * len(users)
        for user in users:
            assert_forwarded_eventually(peer, upstream, keys[user], identity, f"Bearer final-{user}".encode())
        assert sorted(string_value(user_id) for user_id in stored_user_ids(identity)) == sorted(users)


def test_concurrent_stores_of_different_variables_do_not_lose_an_update(gateway: Gateway, peer: Gateway) -> None:
    with mcp_peer() as upstream, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, upstream, TOKEN, WORKSPACE)
        key: Final = user_key(scenario, identity)
        wait_for_tools(peer, key, identity)

        def race_once(pool: ThreadPoolExecutor) -> None:
            assert set_names(clear(gateway, key, identity)) == {TOKEN: False, WORKSPACE: False}
            first: Final = pool.submit(store, gateway, key, identity, {TOKEN: "racing-token"})
            second: Final = pool.submit(store, peer, key, identity, {WORKSPACE: "racing-workspace"})
            assert first.result().status_code == 200, first.result().text
            assert second.result().status_code == 200, second.result().text
            assert set_names(env_status(gateway, key, identity)) == {TOKEN: True, WORKSPACE: True}
            assert set_names(env_status(peer, key, identity)) == {TOKEN: True, WORKSPACE: True}
            assert len(stored_user_ids(identity)) == 1
            forwarded: Final = add_upstream_headers(gateway, upstream, key, identity, a=1)
            assert forwarded[b"authorization"] == b"Bearer racing-token"
            assert forwarded[b"x-workspace"] == b"racing-workspace"

        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(5):
                race_once(pool)


def test_store_waits_on_the_sha256_advisory_lock_for_the_user_and_server(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, peer, TOKEN)
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user, object_permission=grants(identity))
        wait_for_tools(gateway, key, identity)
        lock_key: Final = advisory_lock_key(user, identity)
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as holder,
        ):
            holder.execute("SELECT pg_advisory_lock(%s::bigint)", (lock_key,))
            pending: Final = pool.submit(store, gateway, key, identity, {TOKEN: "held-value"})
            eventually(lambda: advisory_waiters(lock_key), lambda rows: len(rows) == 1, seconds=20)
            assert not pending.done()
            holder.execute("SELECT pg_advisory_unlock(%s::bigint)", (lock_key,))
            response: Final = pending.result(timeout=30)
        assert response.status_code == 200, response.text
        assert set_names(env_status(gateway, key, identity)) == {TOKEN: True}


def test_store_does_not_wait_on_the_legacy_blake2b_lock_id(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_user_var_server(scenario, peer, TOKEN)
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user, object_permission=grants(identity))
        lock_key: Final = legacy_advisory_lock_key(f"{user}:{identity}")
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as holder:
            holder.execute("SELECT pg_advisory_lock(%s::bigint)", (lock_key,))
            response: Final = store(gateway, key, identity, {TOKEN: "free-value"})
        assert response.status_code == 200, response.text
        assert set_names(env_status(gateway, key, identity)) == {TOKEN: True}
