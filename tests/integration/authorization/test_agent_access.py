import json
import uuid
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows, write_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue


class _JsonRpcResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: str
    id: str
    result: dict[str, JsonValue]


_STREAM_EVENTS: Final = (
    {"kind": "status-update", "taskId": "t1", "contextId": "c1", "status": {"state": "working"}, "final": False},
    {
        "kind": "artifact-update",
        "taskId": "t1",
        "contextId": "c1",
        "artifact": {"artifactId": "a1", "parts": [{"kind": "text", "text": "chunk"}]},
    },
    {"kind": "status-update", "taskId": "t1", "contextId": "c1", "status": {"state": "completed"}, "final": True},
)
_STREAM_RESULTS: Final = (
    {"contextId": "c1", "final": False, "kind": "status-update", "status": {"state": "working"}, "taskId": "t1"},
    {
        "append": False,
        "artifact": {"artifactId": "a1", "parts": [{"kind": "text", "text": "chunk"}]},
        "contextId": "c1",
        "kind": "artifact-update",
        "lastChunk": False,
        "taskId": "t1",
    },
    {"contextId": "c1", "final": True, "kind": "status-update", "status": {"state": "completed"}, "taskId": "t1"},
)


def _peer_card(url: str, name: str) -> dict[str, JsonValue]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _register_agent(gateway: Gateway, scenario: Scenario, name: str, url: str) -> str:
    created: Final = gateway.request(
        "POST", "/v1/agents", {"agent_name": name, "agent_card_params": _peer_card(url, name)}
    )
    assert created.status_code == 200, created.text
    identity: Final = created.json()["agent_id"]

    def cleanup() -> None:
        deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
        assert deleted.status_code == 200, deleted.text
        assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

    scenario.cleanups.callback(cleanup)
    return identity


def test_agent_permissions_on_keys_teams_and_access_groups_gate_a2a_send_and_card_before_the_upstream(
    gateway: Gateway,
) -> None:
    marker: Final = "a2aacl" + uuid.uuid4().hex[:12]

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        body: Final = json.loads(request.body)
        if body["method"] == "message/stream":
            frames: Final = tuple(
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': event})}\n\n".encode()
                for event in _STREAM_EVENTS
            )
            return Reply(content_type="text/event-stream", chunks=frames)
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "kind": "message",
                        "role": "agent",
                        "messageId": "peer-message",
                        "parts": [{"kind": "text", "text": "granted"}],
                    },
                }
            ).encode()
        )

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        allowed_name: Final = marker + "-allowed"
        denied_name: Final = marker + "-denied"
        allowed: Final = _register_agent(gateway, scenario, allowed_name, wire.url)
        denied: Final = _register_agent(gateway, scenario, denied_name, wire.url)
        group: Final = gateway.request(
            "POST", "/v1/access_group", {"access_group_name": marker + "-group", "access_agent_ids": [allowed]}
        )
        assert group.status_code == 201, group.text
        group_id: Final = group.json()["access_group_id"]

        def delete_group() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/access_group/{group_id}")
            assert deleted.status_code == 204, deleted.text
            assert (
                read_rows(
                    'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s', (group_id,)
                )
                == []
            )

        scenario.cleanups.callback(delete_group)
        tag: Final = marker + "-tag"
        write_rows(
            'UPDATE "LiteLLM_AgentsTable" SET agent_access_groups = ARRAY[%s] WHERE agent_id = %s', (tag, allowed)
        )
        callers: Final = {
            "key grant": scenario.key(object_permission={"agents": [allowed]}),
            "team grant": scenario.key(team_id=scenario.team(object_permission={"agents": [allowed]})),
            "team grant narrowed by key": scenario.key(
                team_id=scenario.team(object_permission={"agents": [allowed, denied]}),
                object_permission={"agents": [allowed]},
            ),
            "access group": scenario.key(access_group_ids=[group_id]),
            "declared agent access group": scenario.key(object_permission={"agent_access_groups": [tag]}),
        }
        message: Final = {
            "kind": "message",
            "role": "user",
            "messageId": marker + "-in",
            "parts": [{"kind": "text", "text": "ping"}],
        }
        send_payload: Final = {
            "jsonrpc": "2.0",
            "id": marker,
            "method": "message/send",
            "params": {"message": message},
        }
        stream_payload: Final = {
            "jsonrpc": "2.0",
            "id": marker,
            "method": "message/stream",
            "params": {"message": message},
        }

        def refusal(identity: str) -> str:
            return json.dumps(
                {"detail": f"Agent '{identity}' is not allowed for your key/team. Contact proxy admin for access."},
                separators=(",", ":"),
            )

        spellings: Final = (
            ("send by id", lambda agent, name: f"/a2a/{agent}", send_payload, "message/send"),
            ("send by name", lambda agent, name: f"/a2a/{name}", send_payload, "message/send"),
            ("send v1 alias", lambda agent, name: f"/v1/a2a/{agent}/message/send", send_payload, "message/send"),
            ("stream", lambda agent, name: f"/a2a/{agent}", stream_payload, "message/stream"),
        )

        def attempt(key: str, agent: str, name: str) -> tuple[tuple[str, int, str], ...]:
            def call(
                spelling: str, path_of: Callable[[str, str], str], payload: dict[str, JsonValue]
            ) -> tuple[str, int, str]:
                headers: Final = {
                    "Authorization": f"Bearer {key}",
                    **({"Accept": "text/event-stream"} if payload is stream_payload else {}),
                }
                response = gateway.client.post(path_of(agent, name), headers=headers, json=payload)
                return spelling, response.status_code, response.read().decode()

            return tuple(call(label, path_of, payload) for label, path_of, payload, _ in spellings)

        def card(key: str, identity: str) -> tuple[int, str]:
            response = gateway.client.get(
                f"/a2a/{identity}/.well-known/agent-card.json", headers={"Authorization": f"Bearer {key}"}
            )
            return response.status_code, response.text

        for label, key in callers.items():
            for spelling, status, text in attempt(key, denied, denied_name):
                refused: Final = refusal(denied_name if "by name" in spelling else denied)
                assert (status, text) == (403, refused), f"{label} {spelling}: {status} {text}"
            assert card(key, denied) == (403, refusal(denied)), label
            assert wire.drain() == (), label
            for spelling, status, text in attempt(key, allowed, allowed_name):
                assert status == 200, f"{label} {spelling}: {status} {text}"
                if "stream" in spelling:
                    frames: Final = text.split("\n\n")
                    assert frames[-1] == "" and all(frame.startswith("data: ") for frame in frames[:-1]), (
                        f"{label} {spelling}: {text}"
                    )
                    parsed: Final = tuple(
                        _JsonRpcResult.model_validate_json(frame.removeprefix("data: ")) for frame in frames[:-1]
                    )
                    assert tuple(frame.result for frame in parsed) == _STREAM_RESULTS, f"{label} {spelling}: {text}"
                    assert {(frame.jsonrpc, frame.id) for frame in parsed} == {("2.0", marker)}, (
                        f"{label} {spelling}: {text}"
                    )
                    continue
                assert _JsonRpcResult.model_validate_json(text) == _JsonRpcResult(
                    jsonrpc="2.0",
                    id=marker,
                    result={
                        "kind": "message",
                        "role": "agent",
                        "messageId": "peer-message",
                        "parts": [{"kind": "text", "text": "granted"}],
                    },
                ), f"{label} {spelling}: {text}"
            status, text = card(key, allowed)
            assert status == 200 and json.loads(text)["url"].endswith(f"/a2a/{allowed}"), f"{label}: {text}"
            forwarded = tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")
            wire_methods: Final = tuple(name for _, _, _, name in spellings)
            assert forwarded == tuple(
                {
                    "jsonrpc": "2.0",
                    "id": forwarded[index]["id"] if index < len(forwarded) else "<none>",
                    "method": wire_method,
                    "params": {"configuration": {"blocking": True}, "message": message},
                }
                for index, wire_method in enumerate(wire_methods)
            ), f"{label}: {forwarded}"
            assert all(
                item["id"] != marker and str(uuid.UUID(item["id"])) == item["id"] for item in forwarded
            ), label

        emptied: Final = gateway.request("PUT", f"/v1/access_group/{group_id}", {"access_agent_ids": []})
        assert emptied.status_code == 200, emptied.text
        emptied_outcomes: Final = attempt(callers["access group"], allowed, allowed_name)
        assert emptied_outcomes == tuple(
            (spelling, 403, refusal(allowed_name if "by name" in spelling else allowed))
            for spelling, _, _ in emptied_outcomes
        ), emptied_outcomes
        assert wire.drain() == (), emptied_outcomes
