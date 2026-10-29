"""Failure shapes around ``litellm_params.a2a_protocol_version``: the hint naming the setting, values the proxy
must ignore, and agent failures that must reach the caller unchanged while the proxy keeps serving."""

from __future__ import annotations

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
    sse_frames,
)
from pydantic import JsonValue

INVALID: Final[tuple[JsonValue, ...]] = ("2.0", "abc", True, False, 1, [], {"v": "0.3"}, "", "0" * 5120)
INVALID_IDS: Final = ("future", "word", "true", "false", "int", "list", "object", "empty", "five-kb")


def _neighbour_answers(gateway: Gateway, identity: str, marker: str) -> None:
    response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(marker))
    assert response.status_code == 200, response.text
    assert "error" not in response.json(), response.text


@pytest.mark.parametrize("method", ("message/send", "message/stream"))
def test_unpinned_langgraph_agent_error_names_the_setting(gateway: Gateway, method: str) -> None:
    with peer("lghint", "langgraph") as agent, gateway.scenario() as scenario:
        identity: Final = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, None)
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker, method=method))
        if method == "message/send":
            assert response.status_code == 500, response.text
            error = response.json()["error"]
        else:
            assert response.status_code == 200, response.text
            frames: Final = sse_frames(response.text)
            assert len(frames) == 1, response.text
            error = frames[0]["error"]
        assert error["code"] == -32603, response.text  # type: ignore[index]
        assert KIND_ERROR in error["message"] and error["message"].endswith(HINT), response.text  # type: ignore[index]
        sent: Final = posts(agent.wire)  # type: ignore[arg-type]
        expected_method: Final = "SendMessage" if method == "message/send" else "SendStreamingMessage"
        assert [item["method"] for item in sent] == [expected_method], sent


@pytest.mark.parametrize("value", INVALID, ids=INVALID_IDS)
def test_invalid_protocol_version_is_ignored(gateway: Gateway, value: JsonValue) -> None:
    with (
        peer("lginvalid", "langgraph") as agent,
        peer("lgneighbour", "langgraph") as neighbour,
        gateway.scenario() as scenario,
    ):
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": value}
        )
        pinned: Final = register(
            gateway, scenario.cleanups.callback, neighbour.marker, neighbour.url, {"a2a_protocol_version": "0.3"}
        )
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
        assert response.status_code == 500, response.text
        message: Final = response.json()["error"]["message"]
        assert KIND_ERROR in message and message.endswith(HINT), response.text
        assert [item["method"] for item in posts(agent.wire)] == ["SendMessage"]  # type: ignore[arg-type]
        _neighbour_answers(gateway, pinned, neighbour.marker)
        assert [item["method"] for item in posts(neighbour.wire)] == ["message/send"]  # type: ignore[arg-type]
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_missing_and_null_protocol_version_keep_card_version(gateway: Gateway) -> None:
    for params in (None, {"a2a_protocol_version": None}):
        with peer("lgabsent", "langgraph") as agent, gateway.scenario() as scenario:
            identity = register(gateway, scenario.cleanups.callback, agent.marker, agent.url, params)
            response = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
            assert response.status_code == 500, response.text
            assert response.json()["error"]["message"].endswith(HINT), response.text
            assert [item["method"] for item in posts(agent.wire)] == ["SendMessage"]  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fields", "status"),
    (
        ({"error": {"code": -32603, "message": "scripted agent failure"}}, 500),
        ({"post_status": 500}, 500),
        ({"post_status": 404}, 500),
    ),
    ids=("jsonrpc-error", "http-500", "http-404"),
)
def test_pinned_agent_upstream_failures_reach_caller_without_hint(
    gateway: Gateway, fields: dict[str, object], status: int
) -> None:
    with (
        peer("lgfail", "langgraph", **fields) as agent,
        peer("lgok", "langgraph") as healthy,
        gateway.scenario() as scenario,
    ):
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "0.3"}
        )
        other: Final = register(
            gateway, scenario.cleanups.callback, healthy.marker, healthy.url, {"a2a_protocol_version": "0.3"}
        )
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
        assert response.status_code == status, response.text
        body: Final = response.json()
        assert body["id"] == agent.marker and "result" not in body, response.text
        message: Final = str(body["error"]["message"])
        assert HINT not in message and KIND_ERROR not in message, response.text
        if "error" in fields:
            assert "scripted agent failure" in message, response.text
        else:
            assert str(fields["post_status"]) in message, response.text
        assert [item["method"] for item in posts(agent.wire)] == ["message/send"]  # type: ignore[arg-type]
        failure: Final = eventually(
            lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE model=%s', ("a2a_agent/" + agent.marker,)),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert failure[0]["status"] == "failure", failure
        _neighbour_answers(gateway, other, healthy.marker)


def test_pinned_agent_card_unreachable_reaches_caller(gateway: Gateway) -> None:
    with peer("lgnocard", "langgraph", card_status=404) as agent, gateway.scenario() as scenario:
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "0.3"}
        )
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
        assert response.status_code == 500, response.text
        body: Final = response.json()
        assert body["id"] == agent.marker and "result" not in body, response.text
        assert "404" in body["error"]["message"] and HINT not in response.text, response.text
        traffic: Final = agent.wire.drain()  # type: ignore[union-attr]
        assert traffic and all(item.method == "GET" for item in traffic), traffic
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_upstream_authored_kind_message_gets_hint(gateway: Gateway) -> None:
    """User ruling 2026-10-10: an agent-authored error that quotes the SDK's ``kind`` parse failure gets the hint
    too, since the substring check cannot tell the two apart and the hint is still the right next step."""
    authored: Final = 'upstream says: Message type "x" has no field named "kind"'
    with (
        peer("lgauthored", "langgraph", error={"code": -32000, "message": authored}) as agent,
        gateway.scenario() as scenario,
    ):
        identity: Final = register(
            gateway, scenario.cleanups.callback, agent.marker, agent.url, {"a2a_protocol_version": "0.3"}
        )
        response: Final = gateway.request("POST", f"/a2a/{identity}", legacy_send(agent.marker))
        assert response.status_code == 500, response.text
        message: Final = str(response.json()["error"]["message"])
        assert authored in message and message.endswith(HINT), response.text
        assert [item["method"] for item in posts(agent.wire)] == ["message/send"]  # type: ignore[arg-type]
