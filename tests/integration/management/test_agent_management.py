import json
import uuid
from typing import Final

from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, JsonValue

_REDACTED: Final = "REDACTED_BY_LITELM"


class _AgentRead(BaseModel):
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
        redacted_params: Final = {"api_key": _REDACTED, "cost_per_query": 0.1, "is_public": False}

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
        assert (read.agent_id, read.litellm_params, read.static_headers) == (visible, redacted_params, None), (
            fetched.text
        )

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


def test_admin_agent_reads_mask_static_header_values(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        agent_name: Final = "agent-headers-" + uuid.uuid4().hex[:12]
        secret: Final = "Bearer agent-header-" + uuid.uuid4().hex
        expected_headers: Final = {"Authorization": _REDACTED, "X-Tenant": _REDACTED}
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": agent_name,
                "agent_card_params": _peer_card(gateway.upstream_url, agent_name),
                "litellm_params": {"cost_per_query": 0.1},
                "static_headers": {"Authorization": secret, "X-Tenant": "tenant-a"},
            },
        )
        assert created.status_code == 200, created.text
        created_agent: Final = _AgentRead.model_validate(created.json())
        assert created_agent.static_headers == expected_headers, created.text
        assert secret not in created.text, created.text

        def cleanup() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{created_agent.agent_id}")
            assert deleted.status_code == 200, deleted.text
            assert read_rows(
                'SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
                (created_agent.agent_id,),
            ) == []

        scenario.cleanups.callback(cleanup)

        listed: Final = gateway.request("GET", "/v1/agents")
        assert listed.status_code == 200, listed.text
        listed_agents: Final = tuple(
            agent for agent in map(_AgentRead.model_validate, listed.json()) if agent.agent_id == created_agent.agent_id
        )
        assert len(listed_agents) == 1, listed.text
        assert listed_agents[0].static_headers == expected_headers, listed.text
        assert secret not in listed.text, listed.text

        fetched: Final = gateway.request("GET", f"/v1/agents/{created_agent.agent_id}")
        assert fetched.status_code == 200, fetched.text
        fetched_agent: Final = _AgentRead.model_validate(fetched.json())
        assert fetched_agent.static_headers == expected_headers, fetched.text
        assert secret not in fetched.text, fetched.text


def test_agent_updates_restore_echoed_static_header_markers_for_upstream_calls(gateway: Gateway) -> None:
    marker: Final = "agent-header-update-" + uuid.uuid4().hex[:12]
    secret: Final = "Bearer agent-header-" + uuid.uuid4().hex

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
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker),
                "litellm_params": {"cost_per_query": 0.1},
                "static_headers": {"Authorization": secret, "X-Tenant": "tenant-a"},
            },
        )
        assert created.status_code == 200, created.text
        created_agent: Final = _AgentRead.model_validate(created.json())

        def cleanup() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{created_agent.agent_id}")
            assert deleted.status_code == 200, deleted.text
            assert read_rows(
                'SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
                (created_agent.agent_id,),
            ) == []

        scenario.cleanups.callback(cleanup)

        def send_message(message_id: str) -> None:
            response: Final = gateway.client.post(
                f"/a2a/{created_agent.agent_id}",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "method": "message/send",
                    "params": {
                        "message": {
                            "kind": "message",
                            "role": "user",
                            "messageId": message_id + "-in",
                            "parts": [{"kind": "text", "text": "ping"}],
                        }
                    },
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["result"]["parts"] == [{"kind": "text", "text": "authorized"}], response.text

        patched: Final = gateway.request(
            "PATCH",
            f"/v1/agents/{created_agent.agent_id}",
            {
                "static_headers": {
                    "Authorization": _REDACTED,
                    "X-Tenant": "tenant-b",
                    "X-Ghost": _REDACTED,
                }
            },
        )
        assert patched.status_code == 200, patched.text
        assert patched.json()["static_headers"] == {
            "Authorization": _REDACTED,
            "X-Tenant": _REDACTED,
        }, patched.text
        assert secret not in patched.text, patched.text
        assert read_rows(
            'SELECT static_headers FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
            (created_agent.agent_id,),
        ) == [{"static_headers": {"Authorization": secret, "X-Tenant": "tenant-b"}}]

        send_message(marker + "-patch")
        patched_posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert [
            (post.headers["authorization"], post.headers["x-tenant"]) for post in patched_posts
        ] == [(secret, "tenant-b")], patched_posts
        assert not any("x-ghost" in post.headers for post in patched_posts), patched_posts

        updated: Final = gateway.request(
            "PUT",
            f"/v1/agents/{created_agent.agent_id}",
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker),
                "litellm_params": {"cost_per_query": 0.1},
                "static_headers": {
                    "Authorization": _REDACTED,
                    "X-Tenant": _REDACTED,
                },
            },
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["static_headers"] == {
            "Authorization": _REDACTED,
            "X-Tenant": _REDACTED,
        }, updated.text
        assert secret not in updated.text, updated.text
        assert read_rows(
            'SELECT static_headers FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
            (created_agent.agent_id,),
        ) == [{"static_headers": {"Authorization": secret, "X-Tenant": "tenant-b"}}]

        send_message(marker + "-put")
        updated_posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert [
            (post.headers["authorization"], post.headers["x-tenant"]) for post in updated_posts
        ] == [(secret, "tenant-b")], updated_posts
        assert not any("x-ghost" in post.headers for post in updated_posts), updated_posts


def test_agent_patch_replaces_real_static_header_values_for_upstream_calls(gateway: Gateway) -> None:
    marker: Final = "agent-header-rotation-" + uuid.uuid4().hex[:12]
    replacement: Final = "Bearer agent-header-replacement-" + uuid.uuid4().hex

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
        agent_id: Final = _register_agent(gateway, scenario, marker, wire.url, marker)
        patched: Final = gateway.request(
            "PATCH",
            f"/v1/agents/{agent_id}",
            {"static_headers": {"Authorization": replacement}},
        )
        assert patched.status_code == 200, patched.text
        assert read_rows(
            'SELECT static_headers FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
            (agent_id,),
        ) == [{"static_headers": {"Authorization": replacement}}]

        invoked: Final = gateway.client.post(
            f"/a2a/{agent_id}",
            headers={"Authorization": f"Bearer {gateway.key}"},
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
        posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert [post.headers["authorization"] for post in posts] == [replacement], posts


def test_agent_put_omitting_static_headers_clears_them(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        agent_name: Final = "agent-header-clear-" + uuid.uuid4().hex[:12]
        agent_id: Final = _register_agent(
            gateway,
            scenario,
            agent_name,
            gateway.upstream_url,
            "agent-header-clear-" + uuid.uuid4().hex,
        )
        updated: Final = gateway.request(
            "PUT",
            f"/v1/agents/{agent_id}",
            {
                "agent_name": agent_name,
                "agent_card_params": _peer_card(gateway.upstream_url, agent_name),
                "litellm_params": {"cost_per_query": 0.1},
            },
        )
        assert updated.status_code == 200, updated.text
        assert read_rows(
            'SELECT static_headers FROM "LiteLLM_AgentsTable" WHERE agent_id=%s',
            (agent_id,),
        ) == [{"static_headers": {}}]
