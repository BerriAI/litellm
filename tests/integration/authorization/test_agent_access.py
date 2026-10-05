import json
import uuid
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue


class _JsonRpcResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: str
    id: str
    result: dict[str, JsonValue]


def _peer_card(url: str, name: str) -> dict[str, JsonValue]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": False},
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
        allowed: Final = _register_agent(gateway, scenario, marker + "-allowed", wire.url)
        denied: Final = _register_agent(gateway, scenario, marker + "-denied", wire.url)
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
        callers: Final = {
            "key grant": scenario.key(object_permission={"agents": [allowed]}),
            "team grant": scenario.key(team_id=scenario.team(object_permission={"agents": [allowed]})),
            "team grant narrowed by key": scenario.key(
                team_id=scenario.team(object_permission={"agents": [allowed, denied]}),
                object_permission={"agents": [allowed]},
            ),
            "access group": scenario.key(access_group_ids=[group_id]),
        }

        def send(key: str, identity: str) -> tuple[int, str]:
            response = gateway.client.post(
                f"/a2a/{identity}",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "jsonrpc": "2.0",
                    "id": marker,
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "role": "user",
                            "messageId": marker + "-in",
                            "parts": [{"kind": "text", "text": "ping"}],
                        }
                    },
                },
            )
            return response.status_code, response.text

        def card(key: str, identity: str) -> tuple[int, str]:
            response = gateway.client.get(
                f"/a2a/{identity}/.well-known/agent-card.json", headers={"Authorization": f"Bearer {key}"}
            )
            return response.status_code, response.text

        refusal: Final = json.dumps(
            {"detail": f"Agent '{denied}' is not allowed for your key/team. Contact proxy admin for access."},
            separators=(",", ":"),
        )
        for label, key in callers.items():
            assert send(key, denied) == (403, refusal), label
            assert card(key, denied) == (403, refusal), label
            assert wire.drain() == (), label
            status, text = send(key, allowed)
            assert status == 200, f"{label}: {text}"
            assert _JsonRpcResult.model_validate_json(text).result["parts"] == [{"kind": "text", "text": "granted"}], (
                f"{label}: {text}"
            )
            status, text = card(key, allowed)
            assert status == 200 and json.loads(text)["url"].endswith(f"/a2a/{allowed}"), f"{label}: {text}"
            assert len(tuple(item for item in wire.drain() if item.method == "POST")) == 1, label

        emptied: Final = gateway.request("PUT", f"/v1/access_group/{group_id}", {"access_agent_ids": []})
        assert emptied.status_code == 200, emptied.text
        assert send(callers["access group"], allowed) == (
            403,
            json.dumps(
                {"detail": f"Agent '{allowed}' is not allowed for your key/team. Contact proxy admin for access."},
                separators=(",", ":"),
            ),
        )
        assert tuple(item for item in wire.drain() if item.method == "POST") == ()
