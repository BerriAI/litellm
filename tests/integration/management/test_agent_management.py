import json
import uuid
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

_REDACTED: Final = "REDACTED_BY_LITELM"


class _AgentRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_name: str
    litellm_params: dict[str, JsonValue] | None
    static_headers: dict[str, str] | None


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


def _register_agent(gateway: Gateway, scenario: Scenario, name: str, url: str, secret: str) -> str:
    created: Final = gateway.request(
        "POST",
        "/v1/agents",
        {
            "agent_name": name,
            "agent_card_params": _peer_card(url, name),
            "litellm_params": {"api_key": secret, "cost_per_query": 0.1},
            "static_headers": {"Authorization": f"Bearer {secret}"},
        },
    )
    assert created.status_code == 200, created.text
    assert secret not in json.dumps(created.json()["litellm_params"]), created.text
    identity: Final = created.json()["agent_id"]

    def cleanup() -> None:
        deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
        assert deleted.status_code == 200, deleted.text
        assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

    scenario.cleanups.callback(cleanup)
    return identity


def test_agent_reads_redact_credentials_filter_by_key_permission_and_keep_the_stored_credential_for_calls(
    gateway: Gateway,
) -> None:
    marker: Final = "agentread" + uuid.uuid4().hex[:12]
    secret: Final = "sk-agent-" + uuid.uuid4().hex

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
                        "parts": [{"kind": "text", "text": "authorized"}],
                    },
                }
            ).encode()
        )

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        visible: Final = _register_agent(gateway, scenario, marker + "-visible", wire.url, secret)
        hidden: Final = _register_agent(gateway, scenario, marker + "-hidden", wire.url, secret)
        restricted: Final = scenario.key(object_permission={"agents": [visible]})
        stored_params: Final = {"api_key": _REDACTED, "cost_per_query": 0.1}
        redacted_params: Final = {**stored_params, "is_public": False}

        listed: Final = gateway.client.get("/v1/agents", headers={"Authorization": f"Bearer {restricted}"})
        assert listed.status_code == 200, listed.text
        assert secret not in listed.text, listed.text
        assert [
            (agent.agent_id, agent.litellm_params, agent.static_headers)
            for agent in map(_AgentRead.model_validate, listed.json())
        ] == [(visible, redacted_params, None)], listed.text

        fetched: Final = gateway.client.get(f"/v1/agents/{visible}", headers={"Authorization": f"Bearer {restricted}"})
        assert fetched.status_code == 200, fetched.text
        assert secret not in fetched.text, fetched.text
        read: Final = _AgentRead.model_validate_json(fetched.content)
        assert (read.agent_id, read.static_headers) == (visible, None), fetched.text
        assert read.litellm_params in (stored_params, redacted_params), fetched.text

        refused: Final = gateway.client.get(f"/v1/agents/{hidden}", headers={"Authorization": f"Bearer {restricted}"})
        assert (refused.status_code, refused.json()) == (
            403,
            {"detail": f"Agent '{hidden}' is not allowed for your key/team. Contact proxy admin for access."},
        ), refused.text

        admin_list: Final = gateway.request("GET", "/v1/agents")
        assert admin_list.status_code == 200, admin_list.text
        admin_rows: Final = {
            agent.agent_id: agent.litellm_params
            for agent in map(_AgentRead.model_validate, admin_list.json())
            if agent.agent_id in (visible, hidden)
        }
        assert admin_rows == {visible: redacted_params, hidden: redacted_params}, admin_list.text

        stored: Final = read_rows(
            "SELECT litellm_params->>'api_key' AS api_key FROM \"LiteLLM_AgentsTable\" WHERE agent_id=%s", (visible,)
        )
        assert stored == [{"api_key": secret}], stored

        invoked: Final = gateway.client.post(
            f"/a2a/{visible}",
            headers={"Authorization": f"Bearer {restricted}"},
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
        assert invoked.status_code == 200, invoked.text
        assert invoked.json()["result"]["parts"] == [{"kind": "text", "text": "authorized"}], invoked.text
        posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert [post.headers["authorization"] for post in posts] == [f"Bearer {secret}"], posts


def test_only_admins_read_an_agents_token_exchange_block(gateway: Gateway) -> None:
    marker: Final = "agentobo" + uuid.uuid4().hex[:12]
    block: Final = {
        "token_exchange_endpoint": "https://user:pass@idp.example.com/token",
        "client_id": "agent-client",
        "client_secret": "obo-secret-" + uuid.uuid4().hex,
        "audience": "api://agent",
    }
    with gateway.scenario() as scenario:
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": _peer_card("https://agent.example.com", marker),
                "litellm_params": {"token_exchange": block},
            },
        )
        assert created.status_code == 200, created.text
        identity: Final = str(created.json()["agent_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/agents/{identity}")
        reader: Final = scenario.key(object_permission={"agents": [identity]})

        fetched: Final = gateway.client.get(f"/v1/agents/{identity}", headers={"Authorization": f"Bearer {reader}"})
        listed: Final = gateway.client.get("/v1/agents", headers={"Authorization": f"Bearer {reader}"})
        for response in (fetched, listed):
            assert response.status_code == 200, response.text
            assert "user:pass" not in response.text and block["client_secret"] not in response.text, response.text
        assert (_AgentRead.model_validate_json(fetched.content).litellm_params or {})["token_exchange"] == _REDACTED
        assert [
            (agent.litellm_params or {})["token_exchange"]
            for agent in map(_AgentRead.model_validate, listed.json())
            if agent.agent_id == identity
        ] == [_REDACTED], listed.text

        admin: Final = gateway.request("GET", f"/v1/agents/{identity}")
        assert admin.status_code == 200, admin.text
        assert (_AgentRead.model_validate_json(admin.content).litellm_params or {})["token_exchange"] == {
            **block,
            "client_secret": _REDACTED,
        }, admin.text
