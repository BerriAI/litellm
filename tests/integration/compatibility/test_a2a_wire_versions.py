import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.compatibility.a2a.supported_versions_preserve_literal_envelopes")
def test_a2a_versions_and_legacy_casing_preserve_real_wire_and_response(gateway: Gateway) -> None:
    for version, legacy in (("0.3", False), ("1.0", False), ("0.3", True)):
        marker: Final = "a2a" + uuid.uuid4().hex

        def upstream(request: Request, marker: str = marker, legacy: bool = legacy) -> Reply:
            if request.method == "GET":
                assert request.target in ("/.well-known/agent-card.json", "/.well-known/agent.json")
                card: Final = {
                    "protocolVersion": "0.3",
                    "name": marker,
                    "description": "Synthetic arithmetic peer",
                    "version": "1.0.0",
                    "url": wire.url + "/",
                    "capabilities": {"streaming": False},
                    "defaultInputModes": ["text"],
                    "defaultOutputModes": ["text"],
                    "skills": [],
                }
                if legacy:
                    card["supportedInterfaces"] = [
                        {"url": wire.url + "/", "protocolBinding": "jsonrpc", "protocolVersion": "1.0"}
                    ]
                return Reply(body=json.dumps(card).encode())
            assert request.method == "POST" and request.target == "/"
            body: Final = json.loads(request.body)
            assert body["jsonrpc"] == "2.0" and body["method"] == "message/send"
            message: Final = body["params"]["message"]
            assert message["role"] == "user" and message["messageId"] == marker + "-in"
            assert message["parts"] == [{"kind": "text", "text": "synthetic ping"}]
            assert "message_id" not in message
            return Reply(
                body=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "kind": "message",
                            "role": "agent",
                            "messageId": marker + "-out",
                            "parts": [{"kind": "text", "text": "synthetic pong"}],
                        },
                    }
                ).encode()
            )

        with wire_server(upstream) as wire, gateway.scenario() as scenario:
            card: Final = {
                "protocolVersion": version,
                "name": marker,
                "description": "Synthetic arithmetic peer",
                "version": "1.0.0",
                "url": wire.url + "/",
                "capabilities": {"streaming": False},
                "defaultInputModes": ["text"],
                "defaultOutputModes": ["text"],
                "skills": [],
            }
            created: Final = gateway.request("POST", "/v1/agents", {"agent_name": marker, "agent_card_params": card})
            identity: Final = created.json()["agent_id"]

            def cleanup(identity: str = identity) -> None:
                deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
                assert deleted.status_code == 200, deleted.text
                assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

            scenario.cleanups.callback(cleanup)
            assert created.status_code == 200, created.text
            assert gateway.get(f"/v1/agents/{identity}")["agent_card_params"]["protocolVersion"] == version
            discovered: Final = gateway.request("GET", f"/a2a/{identity}/.well-known/agent-card.json")
            assert discovered.status_code == 200, discovered.text
            parameters: Final = {
                "message": {
                    "role": "ROLE_USER" if version == "1.0" else "user",
                    "messageId": marker + "-in",
                    "parts": [{"text": "synthetic ping"}]
                    if version == "1.0"
                    else [{"kind": "text", "text": "synthetic ping"}],
                }
            }
            response: Final = gateway.client.post(
                f"/a2a/{identity}",
                headers={"Authorization": f"Bearer {gateway.key}", "a2a-version": version},
                json={
                    "jsonrpc": "2.0",
                    "id": marker,
                    "method": "SendMessage" if version == "1.0" else "message/send",
                    "params": parameters,
                },
            )
            assert response.status_code == 200, response.text
            body: Final = response.json()
            assert body["jsonrpc"] == "2.0" and body["id"] == marker and "error" not in body
            result: Final = body["result"]
            message: Final = result["message"] if version == "1.0" else result
            assert message["messageId"] == marker + "-out"
            assert message["role"] == ("ROLE_AGENT" if version == "1.0" else "agent")
            assert message["parts"][0]["text"] == "synthetic pong"
            assert (
                ("kind" not in result and "message" in result)
                if version == "1.0"
                else (result["kind"] == "message" and "message" not in result)
            )
            actual: Final = wire.drain()
            assert len(tuple(item for item in actual if item.method == "POST")) == 1
            assert any(item.method == "GET" for item in actual)


@pytest.mark.covers("compatibility.a2a.versioned_card_path_agent_is_reached_with_bearer_and_blocking_send")
def test_agent_serving_its_card_only_at_versioned_path_is_reached_with_bearer_and_answers(gateway: Gateway) -> None:
    marker: Final = "foundry" + uuid.uuid4().hex
    bearer: Final = "Bearer synthetic-entra-" + marker

    def upstream(request: Request) -> Reply:
        assert request.headers.get("authorization") == bearer, request.headers
        if request.method == "GET":
            if request.target != "/agentCard/v1.0":
                return Reply(status=404, body=json.dumps({"error": "not found"}).encode())
            return Reply(
                body=json.dumps(
                    {
                        "protocolVersion": "0.3",
                        "name": marker,
                        "description": "Synthetic prompt agent",
                        "version": "1.0.0",
                        "url": wire.url + "/",
                        "capabilities": {"streaming": False},
                        "defaultInputModes": ["text"],
                        "defaultOutputModes": ["text"],
                        "skills": [],
                    }
                ).encode()
            )
        assert request.method == "POST" and request.target == "/", request.target
        body: Final = json.loads(request.body)
        assert body["jsonrpc"] == "2.0" and body["method"] == "message/send", body
        message: Final = body["params"]["message"]
        assert message["kind"] == "message" and message["role"] == "user", message
        assert message["parts"] == [{"kind": "text", "text": "synthetic ping"}], message
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "kind": "message",
                        "role": "agent",
                        "messageId": marker + "-out",
                        "parts": [{"kind": "text", "text": "synthetic pong"}],
                    },
                }
            ).encode()
        )

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        card: Final = {
            "protocolVersion": "0.3",
            "name": marker,
            "description": "Synthetic prompt agent",
            "version": "1.0.0",
            "url": wire.url + "/",
            "capabilities": {"streaming": False},
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [],
        }
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {"agent_name": marker, "agent_card_params": card, "static_headers": {"Authorization": bearer}},
        )
        assert created.status_code == 200, created.text
        identity: Final = created.json()["agent_id"]

        def cleanup() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
            assert deleted.status_code == 200, deleted.text
            assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

        scenario.cleanups.callback(cleanup)
        response: Final = gateway.request(
            "POST",
            f"/a2a/{identity}",
            {
                "jsonrpc": "2.0",
                "id": marker,
                "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message",
                        "role": "user",
                        "messageId": marker + "-in",
                        "parts": [{"kind": "text", "text": "synthetic ping"}],
                    }
                },
            },
        )
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["jsonrpc"] == "2.0" and body["id"] == marker and "error" not in body, response.text
        assert body["result"]["kind"] == "message", response.text
        assert body["result"]["messageId"] == marker + "-out", response.text
        assert body["result"]["parts"] == [{"kind": "text", "text": "synthetic pong"}], response.text
        actual: Final = wire.drain()
        assert tuple(item.target for item in actual if item.method == "GET")[-1] == "/agentCard/v1.0", actual
        assert tuple(item.target for item in actual if item.method == "POST") == ("/",), actual
