"""Per-agent ``litellm_params.a2a_protocol_version`` pins the A2A transport the proxy speaks to the agent.

langgraph-api 0.15 serves cards declaring ``JSONRPC``/``1.0`` while answering in the 0.3 ``kind`` dialect, so
a2a-sdk picks its strict v1 transport and rejects every reply. Pinning "0.3" routes the SDK onto its v0.3
compat transport; pinning "1.0" keeps v1 even where the mis-cased-binding fingerprint would downgrade.
Every cell asserts the caller's response, the exact request the scripted agent received, and Postgres.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration.compatibility._a2a_peers import (
    HINT,
    KIND_ERROR,
    legacy_send,
    peer,
    posts,
    register,
    spend_rows,
    sse_frames,
)
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

PROXY_WORKERS: Final = int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1"))
WORKER_SYNC_SECONDS: Final = 0.0 if PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + 5.0
FRESH_CONNECTION: Final = {"Connection": "close"}
PINNED: Final = {"a2a_protocol_version": "0.3"}
ROUTES: Final = ("/a2a/{id}", "/a2a/{id}/message/send", "/v1/a2a/{id}/message/send")


def _artifact_text(result: dict[str, JsonValue]) -> str:
    task: Final = result.get("task", result)
    assert isinstance(task, dict), result
    artifacts: Final = task["artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 1, result
    parts: Final = artifacts[0]["parts"]  # type: ignore[index]
    return str(parts[0]["text"])  # type: ignore[index]


def _assert_legacy_wire(sent: tuple[dict[str, JsonValue], ...], marker: str, method: str = "message/send") -> None:
    assert len(sent) == 1, sent
    assert sent[0]["method"] == method, sent
    message: Final = sent[0]["params"]["message"]  # type: ignore[index]
    assert message["kind"] == "message" and message["role"] == "user", sent
    assert message["parts"] == [{"kind": "text", "text": "synthetic ping"}], sent
    assert message["messageId"] == marker + "-in", sent


def _assert_v1_wire(sent: tuple[dict[str, JsonValue], ...], marker: str, method: str = "SendMessage") -> None:
    assert len(sent) == 1, sent
    assert sent[0]["method"] == method, sent
    message: Final = sent[0]["params"]["message"]  # type: ignore[index]
    assert "kind" not in message and message["role"] == "ROLE_USER", sent
    assert message["parts"] == [{"text": "synthetic ping"}], sent
    assert message["messageId"] == marker + "-in", sent


def _one_spend_row(request_id: str, call_type: str, status: str) -> None:
    rows: Final = eventually(lambda: spend_rows(request_id), lambda values: len(values) == 1, seconds=70)
    assert rows[0]["call_type"] == call_type and rows[0]["status"] == status, rows


def _send_ok(gateway: Gateway, identity: str, marker: str, path: str = "/a2a/{id}") -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", path.format(id=identity), legacy_send(marker))
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["id"] == marker and "error" not in body, response.text
    assert _artifact_text(body["result"]) == "echo: synthetic ping", response.text
    return body


def _send_kind_error(gateway: Gateway, identity: str, marker: str) -> str:
    response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(marker))
    assert response.status_code == 500, response.text
    message: Final = str(response.json()["error"]["message"])
    assert KIND_ERROR in message, response.text
    return message


@pytest.mark.parametrize("route", ROUTES)
def test_pinned_langgraph_agent_answers_message_send_in_the_legacy_dialect(gateway: Gateway, route: str) -> None:
    with peer("lgsend", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        _send_ok(gateway, identity, agent.marker, route)
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_pinned_langgraph_agent_streams_message_stream_to_the_final_frame(gateway: Gateway) -> None:
    with peer("lgstream", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        response: Final = gateway.request(
            "POST", f"/a2a/{identity}", legacy_send(agent.marker, method="message/stream")
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream"), response.headers
        frames: Final = sse_frames(response.text)
        assert all(frame["id"] == agent.marker and "error" not in frame for frame in frames), response.text
        results: Final = [frame["result"] for frame in frames]
        assert len(results) == 3, response.text
        assert results[0]["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING", response.text  # type: ignore[index]
        assert results[0]["statusUpdate"]["status"]["message"]["parts"] == [{"text": "echo: synthetic ping"}]  # type: ignore[index]
        assert results[1]["artifactUpdate"]["artifact"]["parts"] == [{"text": "echo: synthetic ping"}]  # type: ignore[index]
        assert results[2]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED", response.text  # type: ignore[index]
        _assert_legacy_wire(posts(agent.wire), agent.marker, "message/stream")  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message_streaming", "success")


def test_v1_caller_reaches_pinned_legacy_agent_in_its_own_envelope(gateway: Gateway) -> None:
    with peer("lgv1caller", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        response: Final = gateway.client.post(
            f"/a2a/{identity}",
            headers={"Authorization": f"Bearer {gateway.key}", "a2a-version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": agent.marker,
                "method": "SendMessage",
                "params": {
                    "message": {
                        "role": "ROLE_USER",
                        "messageId": agent.marker + "-in",
                        "parts": [{"text": "synthetic ping"}],
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["id"] == agent.marker and "error" not in body, response.text
        task: Final = body["result"]["task"]
        assert "kind" not in task and task["status"]["state"] == "TASK_STATE_COMPLETED", response.text
        assert task["artifacts"][0]["parts"] == [{"text": "echo: synthetic ping"}], response.text
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


@pytest.mark.parametrize("value", ("0.3.0", 0.3, " 0.3 "), ids=("dotted", "number", "spaced"))
def test_protocol_version_aliases_pin_the_legacy_transport(gateway: Gateway, value: JsonValue) -> None:
    with peer("lgalias", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": value}
        )
        _send_ok(gateway, identity, agent.marker)
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_genuine_v1_agent_pinned_one_point_zero_is_unchanged(gateway: Gateway) -> None:
    with peer("v1pinned", "v1") as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "1.0"}
        )
        _send_ok(gateway, identity, agent.marker)
        _assert_v1_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_genuine_v1_agent_unpinned_is_unchanged(gateway: Gateway) -> None:
    with peer("v1plain", "v1") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, None)
        _send_ok(gateway, identity, agent.marker)
        _assert_v1_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


@pytest.mark.parametrize("pin", (None, "0.3"), ids=("unpinned", "pinned-legacy"))
def test_lowercase_binding_still_downgrades_to_the_legacy_transport(gateway: Gateway, pin: str | None) -> None:
    with (
        peer("lglower", "langgraph", interfaces=[{"protocolBinding": "jsonrpc"}]) as agent,
        gateway.scenario() as scenario,
    ):
        params: Final = None if pin is None else {"a2a_protocol_version": pin}
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, params)
        _send_ok(gateway, identity, agent.marker)
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_explicit_one_point_zero_opts_out_of_the_lowercase_downgrade(gateway: Gateway) -> None:
    with peer("v1lower", "v1", interfaces=[{"protocolBinding": "jsonrpc"}]) as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "1.0"}
        )
        _send_ok(gateway, identity, agent.marker)
        _assert_v1_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_one_point_zero_pin_on_a_legacy_card_sends_v1(gateway: Gateway) -> None:
    legacy_card: Final = [{"protocolBinding": "JSONRPC", "protocolVersion": "0.3"}]
    with peer("v1oncard03", "v1", interfaces=legacy_card) as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "1.0"}
        )
        _send_ok(gateway, identity, agent.marker)
        _assert_v1_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_override_rewrites_only_canonical_interfaces(gateway: Gateway) -> None:
    with peer("lgmulti", "langgraph") as agent, peer("lgdecoy", "langgraph") as decoy, gateway.scenario() as scenario:
        agent.interfaces = [
            {"protocolBinding": "JSONRPC", "url": agent.url},
            {"protocolBinding": "custom-ws", "url": decoy.url},
            {"protocolBinding": "HTTP+JSON", "url": decoy.url},
        ]
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        _send_ok(gateway, identity, agent.marker)
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        assert decoy.wire.drain() == (), "the HTTP+JSON and custom bindings must not be selected"  # type: ignore[union-attr]
        _one_spend_row(agent.marker, "asend_message", "success")


def test_override_added_and_removed_by_patch_takes_effect_on_both_workers(gateway: Gateway) -> None:
    with peer("lgpatch", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, None)
        assert HINT in _send_kind_error(gateway, identity, agent.marker + "-before")
        before: Final = posts(agent.wire)  # type: ignore[arg-type]
        assert [item["method"] for item in before] == ["SendMessage"], before

        def send(suffix: str) -> int:
            response: Final = gateway.request(
                "POST", f"/a2a/{identity}", legacy_send(agent.marker + suffix), headers=FRESH_CONNECTION
            )
            return response.status_code

        def settle(params: dict[str, JsonValue], expected: int) -> None:
            patched: Final = gateway.request("PATCH", f"/v1/agents/{identity}", {"litellm_params": params})
            assert patched.status_code == 200, patched.text
            assert patched.json()["litellm_params"].get("a2a_protocol_version") == params.get("a2a_protocol_version")
            written_at: Final = time.monotonic()
            counter = iter(range(10_000))
            eventually(
                lambda: (time.monotonic() - written_at, tuple(send(f"-{next(counter)}") for _ in range(4))),
                lambda stamped: stamped[0] >= WORKER_SYNC_SECONDS and set(stamped[1]) == {expected},
                seconds=WORKER_SYNC_SECONDS + 30,
            )
            agent.wire.drain()  # type: ignore[union-attr]

        settle(dict(PINNED), 200)
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
        assert response.status_code == 200, response.text
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]
        settle({}, 500)
        assert HINT in _send_kind_error(gateway, identity, agent.marker + "-after")


def test_repeated_pinned_sends_log_one_row_each(gateway: Gateway) -> None:
    with peer("lgrepeat", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        markers: Final = tuple(f"{agent.marker}-{index}" for index in range(10))
        for marker in markers:
            response = gateway.request("POST", f"/a2a/{identity}", legacy_send(marker))
            assert response.status_code == 200 and response.json()["id"] == marker, response.text
        sent: Final = posts(agent.wire)  # type: ignore[arg-type]
        assert [item["params"]["message"]["messageId"] for item in sent] == [m + "-in" for m in markers], sent  # type: ignore[index]
        assert len({item["id"] for item in sent}) == len(markers), sent
        assert {item["method"] for item in sent} == {"message/send"}, sent
        for marker in markers:
            _one_spend_row(marker, "asend_message", "success")


def test_registered_agent_persists_protocol_version(gateway: Gateway) -> None:
    with peer("lgstored", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        assert gateway.get(f"/v1/agents/{identity}")["litellm_params"]["a2a_protocol_version"] == "0.3"  # type: ignore[index]
        rows: Final = read_rows('SELECT litellm_params FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,))
        assert len(rows) == 1, rows
        stored: Final = rows[0]["litellm_params"]
        stored_params: Final = json.loads(stored) if isinstance(stored, str) else stored
        assert stored_params["a2a_protocol_version"] == "0.3", rows  # type: ignore[index]
        _send_ok(gateway, identity, agent.marker)
        _assert_legacy_wire(posts(agent.wire), agent.marker)  # type: ignore[arg-type]


def _closed_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_pinned_agent_survives_localhost_retry(gateway: Gateway) -> None:
    dead: Final = f"http://localhost:{_closed_local_port()}/a2a/unreachable"
    with peer("lgretry", "langgraph", card_url=dead) as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, PINNED)
        _send_ok(gateway, identity, agent.marker)
        traffic: Final = agent.wire.drain()  # type: ignore[union-attr]
        sent: Final = tuple(json.loads(item.body) for item in traffic if item.method == "POST")
        assert tuple(item.target for item in traffic if item.method == "POST") == ("/a2a/" + agent.marker + "/",), (
            traffic
        )
        _assert_legacy_wire(sent, agent.marker)
        _one_spend_row(agent.marker, "asend_message", "success")
