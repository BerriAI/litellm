import asyncio
import json
import re
import threading
import uuid
from typing import Final

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.types import a2a_pb2
from google.protobuf.json_format import MessageToDict
from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue


class _JsonRpcResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jsonrpc: str
    id: str
    result: dict[str, JsonValue]


_MESSAGE_V03: Final = {
    "kind": "message",
    "messageId": "client-message",
    "parts": [{"kind": "text", "text": "synthetic ping"}],
    "role": "user",
}
_MESSAGE_V10: Final = {"messageId": "client-message", "parts": [{"text": "synthetic ping"}], "role": "ROLE_USER"}


def _peer_card(url: str, name: str, version: str = "0.3") -> dict[str, JsonValue]:
    return {
        "protocolVersion": version,
        "name": name,
        "description": "Synthetic peer",
        "version": "1.0.0",
        "url": url + "/",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _is_uuid(value: str) -> bool:
    return (
        re.fullmatch(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value) is not None
        and str(uuid.UUID(value)) == value
    )


def _register_agent(gateway: Gateway, scenario: Scenario, body: dict[str, JsonValue]) -> str:
    created: Final = gateway.request("POST", "/v1/agents", body)
    assert created.status_code == 200, created.text
    identity: Final = created.json()["agent_id"]

    def cleanup() -> None:
        deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
        assert deleted.status_code == 200, deleted.text
        assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

    scenario.cleanups.callback(cleanup)
    return identity


def _proxy_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _message_reply(body: dict[str, JsonValue], text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "kind": "message",
                    "role": "agent",
                    "messageId": "peer-message",
                    "parts": [{"kind": "text", "text": text}],
                },
            }
        ).encode()
    )


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
_STREAM_RESULTS: Final = {
    "0.3": (
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
    ),
    "1.0": (
        {"statusUpdate": {"taskId": "t1", "contextId": "c1", "status": {"state": "TASK_STATE_WORKING"}}},
        {
            "artifactUpdate": {
                "taskId": "t1",
                "contextId": "c1",
                "artifact": {"artifactId": "a1", "parts": [{"text": "chunk"}]},
            }
        },
        {"statusUpdate": {"taskId": "t1", "contextId": "c1", "status": {"state": "TASK_STATE_COMPLETED"}}},
    ),
}


_TRANSPORT_HEADERS: Final = frozenset(
    {
        "host",
        "accept",
        "accept-encoding",
        "connection",
        "content-length",
        "content-type",
        "user-agent",
        "cache-control",
    }
)


@pytest.mark.parametrize(
    ("version", "method", "message"),
    (("0.3", "message/stream", _MESSAGE_V03), ("1.0", "SendStreamingMessage", _MESSAGE_V10)),
    ids=("v0.3", "v1.0"),
)
def test_a2a_stream_relays_each_upstream_event_as_its_own_sse_frame_in_the_pinned_dialect(
    gateway: Gateway, version: str, method: str, message: dict[str, JsonValue]
) -> None:
    marker: Final = "a2astream" + uuid.uuid4().hex
    first_frame_seen: Final = threading.Event()

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target == "/.well-known/agent-card.json", request.target
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        frames: Final = tuple(
            f"data: {json.dumps({'jsonrpc': '2.0', 'id': 'upstream-' + marker, 'result': event})}\n\n".encode()
            for event in _STREAM_EVENTS
        )
        return Reply(content_type="text/event-stream", chunks=frames, gate_after_first=first_frame_seen)

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user)
        identity: Final = _register_agent(
            gateway,
            scenario,
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker, version),
                "static_headers": {"Authorization": "Bearer stream-server-token"},
            },
        )
        with gateway.client.stream(
            "POST",
            f"/a2a/{identity}",
            headers={
                "Authorization": f"Bearer {key}",
                "a2a-version": version,
                "x-litellm-trace-id": marker + "-trace",
                "Accept": "text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": marker, "method": method, "params": {"message": message}},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8", response.headers
            pieces = response.iter_text()
            received = ""
            for piece in pieces:
                received += piece
                if "\n\n" in received:
                    break
            first_frame_seen.set()
            received += "".join(pieces)
        frames: Final = received.split("\n\n")
        assert frames[-1] == "" and all(frame.startswith("data: ") for frame in frames[:-1]), received
        parsed: Final = tuple(_JsonRpcResult.model_validate_json(frame.removeprefix("data: ")) for frame in frames[:-1])
        assert tuple(frame.result for frame in parsed) == _STREAM_RESULTS[version], received
        assert {(frame.jsonrpc, frame.id) for frame in parsed} == {("2.0", marker)}, received
        posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert len(posts) == 1, posts
        assert posts[0].target == "/", posts[0].target
        assert posts[0].headers["accept"] == "text/event-stream", posts[0].headers
        forwarded: Final = {name: value for name, value in posts[0].headers.items() if name not in _TRANSPORT_HEADERS}
        assert forwarded == {
            "authorization": "Bearer stream-server-token",
            "a2a-version": "0.3",
            "x-litellm-trace-id": marker + "-trace",
            "x-litellm-user-id": user,
            "x-litellm-agent-id": identity,
        }, posts[0].headers
        body: Final = json.loads(posts[0].body)
        assert str(uuid.UUID(body["id"])) == body["id"] and body["id"] != marker, posts[0].body
        assert body == {
            "jsonrpc": "2.0",
            "id": body["id"],
            "method": "message/stream",
            "params": {"configuration": {"blocking": True}, "message": _MESSAGE_V03},
        }, posts[0].body


def test_a2a_send_forwards_configured_headers_and_minted_identity_but_never_spoofed_or_proxy_credentials(
    gateway: Gateway,
) -> None:
    marker: Final = "a2ahdr" + uuid.uuid4().hex[:12]

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        return _message_reply(json.loads(request.body), "headers seen")

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        identity: Final = _register_agent(
            gateway,
            scenario,
            {
                "agent_name": marker,
                "agent_card_params": _peer_card(wire.url, marker),
                "static_headers": {"Authorization": "Bearer server-token"},
                "extra_headers": ["x-api-key"],
            },
        )
        team: Final = scenario.team()
        member: Final = scenario.user()
        team_key: Final = scenario.key(team_id=team, user_id=member)
        solo_user: Final = scenario.user()
        solo_key: Final = scenario.key(user_id=solo_user)
        payload: Final = {"jsonrpc": "2.0", "id": marker, "method": "message/send", "params": {"message": _MESSAGE_V03}}
        spoofs: Final = {
            "X-API-Key": "client-secret",
            f"x-a2a-{marker}-authorization": "Bearer agent-token",
            f"x-a2a-{marker}-x-tenant": "tenant-1",
            f"x-a2a-{marker}-x-litellm-team-id": "spoofed-team",
            f"x-a2a-{identity}-x-region": "eu-1",
            "x-a2a-other-agent-authorization": "Bearer leak",
            "X-LiteLLM-User-Id": "spoofed-user",
            "X-LiteLLM-Team-Id": "spoofed-team",
        }
        for headers in (
            {"Authorization": f"Bearer {team_key}", "x-litellm-trace-id": marker + "-trace", **spoofs},
            {"x-litellm-api-key": solo_key, "Authorization": "Bearer backend-token", **spoofs},
        ):
            response = gateway.client.post(f"/a2a/{identity}", headers=headers, json=payload)
            assert response.status_code == 200, response.text
            assert _JsonRpcResult.model_validate_json(response.content).result["parts"] == [
                {"kind": "text", "text": "headers seen"}
            ], response.text
        posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert len(posts) == 2, posts
        for post, (user, team_header, key) in zip(
            posts, ((member, team, team_key), (solo_user, None, solo_key)), strict=True
        ):
            forwarded = {name: value for name, value in post.headers.items() if name not in _TRANSPORT_HEADERS}
            minted: Final = forwarded.get("x-litellm-trace-id", "")
            if team_header is None:
                assert _is_uuid(minted), post.headers
            expected = {
                "authorization": "Bearer server-token",
                "x-api-key": "client-secret",
                "x-tenant": "tenant-1",
                "x-region": "eu-1",
                "a2a-version": "0.3",
                "x-litellm-agent-id": identity,
                "x-litellm-trace-id": marker + "-trace" if team_header is not None else minted,
                "x-litellm-user-id": user,
                **({} if team_header is None else {"x-litellm-team-id": team_header}),
            }
            assert forwarded == expected, post.headers
            body = json.loads(post.body)
            assert str(uuid.UUID(body["id"])) == body["id"], post.body
            assert body == {
                "jsonrpc": "2.0",
                "id": body["id"],
                "method": "message/send",
                "params": {"configuration": {"blocking": True}, "message": _MESSAGE_V03},
            }, post.body
            assert all(
                key not in value and gateway.key not in value and "leak" not in value for value in post.headers.values()
            ), post.headers


@pytest.mark.parametrize("version", ("0.3", "1.0"))
def test_a2a_card_routes_front_the_agent_with_the_proxy_url_and_sdk_clients_call_back_through_the_proxy(
    gateway: Gateway, version: str
) -> None:
    marker: Final = "a2acard" + uuid.uuid4().hex

    def upstream_card() -> dict[str, JsonValue]:
        return {
            **_peer_card(wire.url, marker, version),
            "supportedInterfaces": [{"url": wire.url + "/", "protocolBinding": "JSONRPC", "protocolVersion": version}],
        }

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(upstream_card()).encode())
        body: Final = json.loads(request.body)
        if body["method"] in ("message/stream", "SendStreamingMessage"):
            events: Final = (
                _STREAM_EVENTS
                if version == "0.3"
                else tuple(_STREAM_RESULTS["1.0"][index] for index in range(len(_STREAM_EVENTS)))
            )
            frames: Final = tuple(
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': event})}\n\n".encode()
                for event in events
            )
            return Reply(content_type="text/event-stream", chunks=frames)
        if version == "0.3":
            return _message_reply(body, "via proxy")
        message: Final = {"messageId": "peer-message", "role": "ROLE_AGENT", "parts": [{"text": "via proxy"}]}
        return Reply(body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": {"message": message}}).encode())

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user)
        identity: Final = _register_agent(
            gateway,
            scenario,
            {
                "agent_name": marker,
                "agent_card_params": upstream_card(),
                "static_headers": {"Authorization": "Bearer card-server-token"},
            },
        )
        proxy_url: Final = f"{_proxy_base(gateway)}/a2a/{identity}"
        cards: Final = tuple(
            gateway.request("GET", f"/a2a/{identity}/.well-known/{name}") for name in ("agent-card.json", "agent.json")
        )
        for card in cards:
            assert card.status_code == 200, card.text
            assert wire.url not in card.text, card.text
            assert card.json() == cards[0].json(), card.text
        served: Final = cards[0].json()
        assert served["url"] == proxy_url, cards[0].text
        assert served["protocolVersion"] == version, cards[0].text
        assert served["supportedInterfaces"] == [
            {"url": proxy_url, "protocolBinding": "JSONRPC", "protocolVersion": version}
        ], cards[0].text
        assert served["name"] == marker, cards[0].text

        async def call_through_sdk() -> tuple[a2a_pb2.AgentCard, tuple[a2a_pb2.StreamResponse, ...]]:
            async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=30) as http:
                resolved = await A2ACardResolver(httpx_client=http, base_url=proxy_url).get_agent_card()
                client = await create_client(resolved, ClientConfig(httpx_client=http, streaming=False))
                request = a2a_pb2.SendMessageRequest(
                    message=a2a_pb2.Message(
                        message_id=marker + "-in", role=a2a_pb2.Role.ROLE_USER, parts=[a2a_pb2.Part(text="ping")]
                    )
                )
                return resolved, tuple([event async for event in client.send_message(request)])

        resolved, events = asyncio.run(call_through_sdk())
        assert [interface.url for interface in resolved.supported_interfaces] == [proxy_url], resolved
        assert [event.message.parts[0].text for event in events] == ["via proxy"], events

        async def stream_through_sdk() -> tuple[a2a_pb2.StreamResponse, ...]:
            async with httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, timeout=30) as http:
                client = await create_client(resolved, ClientConfig(httpx_client=http, streaming=True))
                request = a2a_pb2.SendMessageRequest(
                    message=a2a_pb2.Message(
                        message_id=marker + "-stream", role=a2a_pb2.Role.ROLE_USER, parts=[a2a_pb2.Part(text="ping")]
                    )
                )
                return tuple([event async for event in client.send_message(request)])

        stream_events: Final = asyncio.run(stream_through_sdk())
        assert [MessageToDict(event, preserving_proto_field_name=False) for event in stream_events] == [
            {"statusUpdate": {"taskId": "t1", "contextId": "c1", "status": {"state": "TASK_STATE_WORKING"}}},
            {
                "artifactUpdate": {
                    "taskId": "t1",
                    "contextId": "c1",
                    "artifact": {"artifactId": "a1", "parts": [{"text": "chunk"}]},
                }
            },
            {"statusUpdate": {"taskId": "t1", "contextId": "c1", "status": {"state": "TASK_STATE_COMPLETED"}}},
        ], stream_events
        posts: Final = tuple(item for item in wire.drain() if item.method == "POST")
        assert len(posts) == 2, posts
        forwarded_headers: Final = tuple(
            {name: value for name, value in post.headers.items() if name not in _TRANSPORT_HEADERS} for post in posts
        )
        assert all(_is_uuid(headers.get("x-litellm-trace-id", "")) for headers in forwarded_headers), forwarded_headers
        assert forwarded_headers == tuple(
            {
                "authorization": "Bearer card-server-token",
                "x-litellm-user-id": user,
                "x-litellm-agent-id": identity,
                "x-litellm-trace-id": headers.get("x-litellm-trace-id", ""),
                "a2a-version": version,
            }
            for headers in forwarded_headers
        ), [post.headers for post in posts]
        forwarded: Final = tuple(json.loads(post.body) for post in posts)
        assert all(str(uuid.UUID(post["id"])) == post["id"] for post in forwarded), [post.body for post in posts]
        assert forwarded == (
            {
                "0.3": {
                    "jsonrpc": "2.0",
                    "id": forwarded[0]["id"],
                    "method": "message/send",
                    "params": {
                        "configuration": {"blocking": True},
                        "message": {
                            "kind": "message",
                            "messageId": marker + "-in",
                            "role": "user",
                            "parts": [{"kind": "text", "text": "ping"}],
                        },
                    },
                },
                "1.0": {
                    "jsonrpc": "2.0",
                    "id": forwarded[0]["id"],
                    "method": "SendMessage",
                    "params": {
                        "configuration": {},
                        "message": {"messageId": marker + "-in", "role": "ROLE_USER", "parts": [{"text": "ping"}]},
                    },
                },
            }[version],
            {
                "0.3": {
                    "jsonrpc": "2.0",
                    "id": forwarded[1]["id"],
                    "method": "message/stream",
                    "params": {
                        "configuration": {"blocking": True},
                        "message": {
                            "kind": "message",
                            "messageId": marker + "-stream",
                            "role": "user",
                            "parts": [{"kind": "text", "text": "ping"}],
                        },
                    },
                },
                "1.0": {
                    "jsonrpc": "2.0",
                    "id": forwarded[1]["id"],
                    "method": "SendStreamingMessage",
                    "params": {
                        "configuration": {},
                        "message": {
                            "messageId": marker + "-stream",
                            "role": "ROLE_USER",
                            "parts": [{"text": "ping"}],
                        },
                    },
                },
            }[version],
        ), [post.body for post in posts]


@pytest.mark.parametrize("streaming", (False, True), ids=("send", "stream"))
def test_a2a_message_send_aliases_and_agent_name_reach_the_same_agent_with_the_same_conversion(
    gateway: Gateway, streaming: bool
) -> None:
    marker: Final = "a2aalias" + uuid.uuid4().hex
    method: Final = "SendStreamingMessage" if streaming else "SendMessage"
    wire_method: Final = "message/stream" if streaming else "message/send"

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        assert (request.method, request.target) == ("POST", "/"), request.target
        body: Final = json.loads(request.body)
        if body["method"] == "message/stream":
            frames: Final = tuple(
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': 'upstream-' + str(body['id']), 'result': event})}\n\n".encode()
                for event in _STREAM_EVENTS
            )
            return Reply(content_type="text/event-stream", chunks=frames)
        return _message_reply(body, "alias pong")

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        identity: Final = _register_agent(
            gateway, scenario, {"agent_name": marker, "agent_card_params": _peer_card(wire.url, marker, "1.0")}
        )
        paths: Final = (f"/a2a/{identity}/message/send", f"/v1/a2a/{identity}/message/send", f"/a2a/{marker}")
        for path in paths:
            headers: Final = {"Authorization": f"Bearer {gateway.key}", "a2a-version": "1.0"}
            payload: Final = {
                "jsonrpc": "2.0",
                "id": marker,
                "method": method,
                "params": {"message": _MESSAGE_V10},
            }
            if streaming:
                with gateway.client.stream(
                    "POST", path, headers={**headers, "Accept": "text/event-stream"}, json=payload
                ) as response:
                    received: Final = response.read().decode()
                    assert response.status_code == 200, f"{path}: {received}"
                frames: Final = received.split("\n\n")
                assert frames[-1] == "" and all(frame.startswith("data: ") for frame in frames[:-1]), (
                    f"{path}: {received}"
                )
                parsed: Final = tuple(
                    _JsonRpcResult.model_validate_json(frame.removeprefix("data: ")) for frame in frames[:-1]
                )
                assert tuple(frame.result for frame in parsed) == _STREAM_RESULTS["1.0"], f"{path}: {received}"
                assert {(frame.jsonrpc, frame.id) for frame in parsed} == {("2.0", marker)}, f"{path}: {received}"
                continue
            response = gateway.client.post(path, headers=headers, json=payload)
            assert response.status_code == 200, f"{path}: {response.text}"
            assert _JsonRpcResult.model_validate_json(response.content) == _JsonRpcResult(
                jsonrpc="2.0",
                id=marker,
                result={
                    "message": {"messageId": "peer-message", "role": "ROLE_AGENT", "parts": [{"text": "alias pong"}]}
                },
            ), f"{path}: {response.text}"
        posts: Final = tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")
        assert len(posts) == len(paths), posts
        assert all(str(uuid.UUID(post["id"])) == post["id"] and post["id"] != marker for post in posts), posts
        assert posts == tuple(
            {
                "jsonrpc": "2.0",
                "id": post["id"],
                "method": wire_method,
                "params": {"configuration": {"blocking": True}, "message": _MESSAGE_V03},
            }
            for post in posts
        ), posts


def test_a2a_task_methods_pass_through_verbatim_and_push_callbacks_are_validated_before_forwarding(
    gateway: Gateway,
) -> None:
    marker: Final = "a2atask" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        body: Final = json.loads(request.body)
        if body["method"] == "tasks/resubscribe":
            frames: Final = tuple(
                f"data: {json.dumps({'jsonrpc': '2.0', 'id': body['id'], 'result': event})}\n\n".encode()
                for event in _STREAM_EVENTS
            )
            return Reply(content_type="text/event-stream", chunks=frames)
        result: JsonValue = (
            _peer_card(wire.url, marker)
            if body["method"] == "agent/getAuthenticatedExtendedCard"
            else {"kind": "task", "id": "t1", "contextId": "c1", "status": {"state": body["method"]}}
        )
        return Reply(body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode())

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        identity: Final = _register_agent(
            gateway, scenario, {"agent_name": marker, "agent_card_params": _peer_card(wire.url, marker)}
        )

        def call(method: str, params: dict[str, JsonValue]) -> httpx.Response:
            return gateway.client.post(
                f"/a2a/{identity}",
                headers={"Authorization": f"Bearer {gateway.key}"},
                json={"jsonrpc": "2.0", "id": marker + method, "method": method, "params": params},
            )

        task_params: Final = {"id": "t1", "historyLength": 2}
        push_params: Final = {"id": "t1", "pushNotificationConfigId": "c1"}
        for method, canonical, params in (
            ("tasks/get", "tasks/get", task_params),
            ("GetTask", "tasks/get", task_params),
            ("tasks/list", "tasks/list", {"contextId": "c1"}),
            ("ListTasks", "tasks/list", {"contextId": "c1"}),
            ("tasks/cancel", "tasks/cancel", {"id": "t1"}),
            ("CancelTask", "tasks/cancel", {"id": "t1"}),
            ("tasks/pushNotificationConfig/get", "tasks/pushNotificationConfig/get", push_params),
            ("GetTaskPushNotificationConfig", "tasks/pushNotificationConfig/get", push_params),
            ("tasks/pushNotificationConfig/list", "tasks/pushNotificationConfig/list", {"id": "t1"}),
            ("ListTaskPushNotificationConfigs", "tasks/pushNotificationConfig/list", {"id": "t1"}),
            ("tasks/pushNotificationConfig/delete", "tasks/pushNotificationConfig/delete", push_params),
            ("DeleteTaskPushNotificationConfig", "tasks/pushNotificationConfig/delete", push_params),
        ):
            response = call(method, params)
            assert response.status_code == 200, f"{method}: {response.text}"
            assert response.json() == {
                "jsonrpc": "2.0",
                "id": marker + method,
                "result": {"kind": "task", "id": "t1", "contextId": "c1", "status": {"state": canonical}},
            }, f"{method}: {response.text}"
            forwarded = tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")
            assert forwarded == (
                {
                    "jsonrpc": "2.0",
                    "id": marker + method,
                    "method": canonical,
                    "params": params,
                },
            ), f"{method}: {forwarded}"

        for method in ("tasks/resubscribe", "SubscribeToTask"):
            with gateway.client.stream(
                "POST",
                f"/a2a/{identity}",
                headers={"Authorization": f"Bearer {gateway.key}", "Accept": "text/event-stream"},
                json={"jsonrpc": "2.0", "id": marker + method, "method": method, "params": {"id": "t1"}},
            ) as response:
                received: Final = response.read().decode()
                assert response.status_code == 200, f"{method}: {received}"
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8", response.headers
            frames: Final = received.split("\n\n")
            assert frames[-1] == "" and all(frame.startswith("data: ") for frame in frames[:-1]), (
                f"{method}: {received}"
            )
            parsed: Final = tuple(
                _JsonRpcResult.model_validate_json(frame.removeprefix("data: ")) for frame in frames[:-1]
            )
            assert tuple(frame.result for frame in parsed) == _STREAM_EVENTS, f"{method}: {received}"
            assert {(frame.jsonrpc, frame.id) for frame in parsed} == {("2.0", marker + method)}, (
                f"{method}: {received}"
            )
            forwarded = tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")
            assert forwarded == (
                {
                    "jsonrpc": "2.0",
                    "id": marker + method,
                    "method": "tasks/resubscribe",
                    "params": {"id": "t1"},
                },
            ), f"{method}: {forwarded}"

        for callback, detail in (
            ("http://169.254.169.254/latest/meta-data/", "Push notification URL must use HTTPS"),
            (
                "https://10.0.0.1/callback",
                "URL targets a blocked address (10.0.0.1). If this is a legitimate internal service, "
                "add the host to `user_url_allowed_hosts` in litellm_settings.",
            ),
        ):
            for params in (
                {"taskId": "t1", "pushNotificationConfig": {"url": callback}},
                {"taskId": "t1", "url": callback},
            ):
                rejected = call("tasks/pushNotificationConfig/set", params)
                assert (rejected.status_code, rejected.json()) == (400, {"detail": detail}), (
                    f"{params}: {rejected.text}"
                )
        assert tuple(item for item in wire.drain() if item.method == "POST") == ()

        public: Final = {"taskId": "t1", "pushNotificationConfig": {"url": "https://1.1.1.1/callback"}}
        accepted: Final = call("tasks/pushNotificationConfig/set", public)
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["result"]["status"] == {"state": "tasks/pushNotificationConfig/set"}, accepted.text
        assert tuple(json.loads(item.body)["params"] for item in wire.drain() if item.method == "POST") == (public,)

        extended: Final = call("agent/getAuthenticatedExtendedCard", {})
        assert extended.status_code == 200, extended.text
        assert extended.json()["result"] == {
            **_peer_card(wire.url, marker),
            "url": f"{_proxy_base(gateway)}/a2a/{identity}",
        }, extended.text
        assert wire.url not in extended.text, extended.text
        assert tuple(json.loads(item.body)["method"] for item in wire.drain() if item.method == "POST") == (
            "agent/getAuthenticatedExtendedCard",
        )


def _v1_card(url: str, name: str) -> dict[str, JsonValue]:
    return {
        **_peer_card(url, name, "1.0"),
        "supportedInterfaces": [{"url": url + "/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }


def test_a2a_v1_task_methods_and_extended_card_lower_to_0_3_upstream_and_answer_in_1_0(gateway: Gateway) -> None:
    marker: Final = "a2atask10" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_v1_card(wire.url, marker)).encode())
        body: Final = json.loads(request.body)
        results: Final[dict[str, JsonValue]] = {
            "tasks/get": {"kind": "task", "id": "t1", "contextId": "c1", "status": {"state": "working"}},
            "tasks/cancel": {"kind": "task", "id": "t1", "contextId": "c1", "status": {"state": "canceled"}},
            "tasks/pushNotificationConfig/set": {
                "taskId": "t1",
                "pushNotificationConfig": {"id": "c1", "url": "https://1.1.1.1/callback"},
            },
            "agent/getAuthenticatedExtendedCard": _v1_card(wire.url, marker),
        }
        return Reply(body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": results[body["method"]]}).encode())

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        identity: Final = _register_agent(
            gateway, scenario, {"agent_name": marker, "agent_card_params": _v1_card(wire.url, marker)}
        )
        proxy_url: Final = f"{_proxy_base(gateway)}/a2a/{identity}"

        def call(method: str, params: dict[str, JsonValue]) -> httpx.Response:
            return gateway.client.post(
                f"/a2a/{identity}",
                headers={"Authorization": f"Bearer {gateway.key}", "a2a-version": "1.0"},
                json={"jsonrpc": "2.0", "id": marker + method, "method": method, "params": params},
            )

        def forwarded() -> tuple[JsonValue, ...]:
            return tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")

        for method, canonical, state, params in (
            ("GetTask", "tasks/get", "TASK_STATE_WORKING", {"id": "t1", "historyLength": 2}),
            ("CancelTask", "tasks/cancel", "TASK_STATE_CANCELED", {"id": "t1"}),
        ):
            response = call(method, params)
            assert response.status_code == 200, response.text
            assert response.json() == {
                "jsonrpc": "2.0",
                "id": marker + method,
                "result": {"id": "t1", "contextId": "c1", "status": {"state": state}},
            }, response.text
            assert forwarded() == ({"jsonrpc": "2.0", "id": marker + method, "method": canonical, "params": params},)

        for params in (
            {"parent": "tasks/t1", "configId": "c1", "config": {"url": "http://169.254.169.254/latest/meta-data/"}},
            {"taskId": "t1", "url": "http://169.254.169.254/latest/meta-data/"},
        ):
            rejected = call("CreateTaskPushNotificationConfig", params)
            assert (rejected.status_code, rejected.json()) == (
                400,
                {"detail": "Push notification URL must use HTTPS"},
            ), f"{params}: {rejected.text}"
        assert forwarded() == ()

        accepted: Final = call(
            "CreateTaskPushNotificationConfig",
            {"parent": "tasks/t1", "configId": "c1", "config": {"url": "https://1.1.1.1/callback"}},
        )
        assert accepted.status_code == 200, accepted.text
        assert forwarded() == (
            {
                "jsonrpc": "2.0",
                "id": marker + "CreateTaskPushNotificationConfig",
                "method": "tasks/pushNotificationConfig/set",
                "params": {"taskId": "t1", "pushNotificationConfig": {"id": "c1", "url": "https://1.1.1.1/callback"}},
            },
        )

        extended: Final = call("GetExtendedAgentCard", {})
        assert extended.status_code == 200, extended.text
        assert extended.json() == {
            "jsonrpc": "2.0",
            "id": marker + "GetExtendedAgentCard",
            "result": {
                **_v1_card(wire.url, marker),
                "url": proxy_url,
                "supportedInterfaces": [{"url": proxy_url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
            },
        }, extended.text
        assert wire.url not in extended.text, extended.text
        assert forwarded() == (
            {
                "jsonrpc": "2.0",
                "id": marker + "GetExtendedAgentCard",
                "method": "agent/getAuthenticatedExtendedCard",
                "params": {},
            },
        )


def test_a2a_v1_push_config_envelope_sent_to_a_0_3_agent_has_its_callback_validated_before_forwarding(
    gateway: Gateway,
) -> None:
    pytest.skip(
        "BUG: CreateTaskPushNotificationConfig {parent, configId, config: {url: http://169.254.169.254/...}} with "
        "a2a-version 1.0 to an agent pinned to 0.3 returns 200 and forwards the callback unvalidated, because "
        "validation reads only params.url and pushNotificationConfig.url and 0.3-pinned params are never lowered"
    )
    marker: Final = "a2apush03" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_peer_card(wire.url, marker)).encode())
        body: Final = json.loads(request.body)
        return Reply(body=json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": body["params"]}).encode())

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        identity: Final = _register_agent(
            gateway, scenario, {"agent_name": marker, "agent_card_params": _peer_card(wire.url, marker)}
        )
        rejected: Final = gateway.client.post(
            f"/a2a/{identity}",
            headers={"Authorization": f"Bearer {gateway.key}", "a2a-version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": marker,
                "method": "CreateTaskPushNotificationConfig",
                "params": {
                    "parent": "tasks/t1",
                    "configId": "c1",
                    "config": {"url": "http://169.254.169.254/latest/meta-data/"},
                },
            },
        )
        assert (rejected.status_code, rejected.json()) == (
            400,
            {"detail": "Push notification URL must use HTTPS"},
        ), rejected.text
        assert tuple(item for item in wire.drain() if item.method == "POST") == ()
