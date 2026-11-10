import asyncio
import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


async def _async_post(gateway: Gateway, body: dict[str, JsonValue]) -> httpx.Response:
    async with httpx.AsyncClient(
        base_url=str(gateway.client.base_url),
        timeout=20,
        trust_env=False,
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {gateway.key}"},
        )


@pytest.mark.parametrize(
    ("asynchronous", "streaming"),
    ((True, False), (True, True), (False, False), (False, True)),
    ids=("async-non-streaming", "async-streaming", "sync-non-streaming", "sync-streaming"),
)
def test_a2a_completion_through_proxy(
    gateway: Gateway,
    asynchronous: bool,
    streaming: bool,
) -> None:
    agent_name: Final = "translation-a2a-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        if request.method == "GET":
            assert request.target in ("/.well-known/agent-card.json", "/.well-known/agent.json")
            return Reply(
                body=json.dumps(
                    {
                        "protocolVersion": "0.3",
                        "name": agent_name,
                        "description": "Scripted A2A peer",
                        "version": "1.0.0",
                        "url": wire.url + "/",
                        "capabilities": {"streaming": True},
                        "defaultInputModes": ["text"],
                        "defaultOutputModes": ["text"],
                        "skills": [],
                    }
                ).encode()
            )

        assert request.method == "POST" and request.target == "/"
        outbound: Final = json.loads(request.body)
        outbound_message: Final = outbound["params"]["message"]
        request_id: Final = outbound["id"]
        message_id: Final = outbound_message["messageId"]
        method: Final = "message/stream" if streaming else "message/send"
        expected_params: Final = {
            "message": {
                "kind": "message",
                "role": "user",
                "parts": [{"kind": "text", "text": "user: Hello"}],
                "messageId": message_id,
            },
            **({} if streaming else {"configuration": {"blocking": True}}),
        }
        assert outbound == {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": expected_params,
            **({"stream": True} if streaming else {}),
        }

        if streaming:
            event: Final = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "kind": "task",
                    "status": {"state": "completed"},
                    "artifacts": [{"parts": [{"kind": "text", "text": "A2A response"}]}],
                },
            }
            return Reply(
                body=f"event: message\ndata: {json.dumps(event)}\n\n".encode(),
                content_type="text/event-stream",
            )

        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "kind": "message",
                        "role": "agent",
                        "messageId": agent_name,
                        "parts": [{"kind": "text", "text": "A2A response"}],
                    },
                }
            ).encode()
        )

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        card: Final = {
            "protocolVersion": "0.3",
            "name": agent_name,
            "description": "Scripted A2A peer",
            "version": "1.0.0",
            "url": wire.url + "/",
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [],
        }
        created: Final = gateway.request(
            "POST",
            "/v1/agents",
            {"agent_name": agent_name, "agent_card_params": card},
        )
        assert created.status_code == 200, created.text
        agent_id: Final = created.json()["agent_id"]

        def cleanup() -> None:
            deleted: Final = gateway.request("DELETE", f"/v1/agents/{agent_id}")
            assert deleted.status_code == 200, deleted.text

        scenario.cleanups.callback(cleanup)
        body: Final = {
            "model": f"a2a/{agent_name}",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": streaming,
        }
        response: Final = (
            asyncio.run(_async_post(gateway, body))
            if asynchronous
            else gateway.request("POST", "/v1/chat/completions", body)
        )
        assert response.status_code == 200, response.text

        if streaming:
            frames: Final = tuple(
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
            chunks: Final = tuple(json.loads(frame) for frame in frames)
            content: Final = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
            finish_reasons: Final = tuple(
                chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"][0].get("finish_reason")
            )
            assert content == "A2A response"
            assert finish_reasons == ("stop",)
        else:
            completion: Final = response.json()
            assert completion["choices"][0]["message"]["content"] == "A2A response"
            assert completion["choices"][0]["finish_reason"] == "stop"

        requests: Final = wire.drain()
        assert len(tuple(item for item in requests if item.method == "POST")) == 1
