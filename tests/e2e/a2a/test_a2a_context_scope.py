"""Per-key A2A conversation scope, end to end against whatever proxy `client` talks to.

Two keys each ask a scripted echo agent for a turn with their own `message.contextId`,
and the agent answers with the id it actually received. An agent that keeps its memory
per contextId has to see two different ids, or the second caller reads the first one's
conversation.

The scripted agent answers `message/send` directly; every other route the registrations
need (register, read back, delete, the discovery card) goes through `A2AClient` exactly
as the other A2A tests drive it, so this file runs against a live proxy the same way.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from a2a_client import A2AClient, A2ASkill, AgentRegisterBody
from e2e_config import unique_marker
from e2e_http import Result, Success, is_ok, unwrap
from e2e_metadata import Domain, Route, Subject, meta
from lifecycle import ResourceManager
from pydantic import BaseModel, ConfigDict, Field

pytestmark = pytest.mark.e2e


class _Reply(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: str = "message"
    role: str = "agent"
    parts: list[dict[str, object]]
    message_id: str = Field(serialization_alias="messageId")
    context_id: str | None = Field(default=None, serialization_alias="contextId")


class _Answer(BaseModel):
    jsonrpc: str = "2.0"
    id: str | None = None
    result: _Reply


class EchoAgent:
    """A minimal A2A agent that answers with the contextId it received."""

    def __init__(self) -> None:
        self.received: list[str | None] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        agent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                message = (body.get("params") or {}).get("message") or {}
                agent.received.append(message.get("contextId"))
                answer = _Answer(
                    id=body.get("id"),
                    result=_Reply(
                        parts=[{"kind": "text", "text": "ok"}],
                        message_id=f"{message.get('messageId', 'm')}-reply",
                        context_id=message.get("contextId"),
                    ),
                )
                payload = json.dumps(answer.model_dump(by_alias=True, exclude_none=True)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        return Handler

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def send(self, key: str, agent_id: str, context_id: str) -> Result[_Answer]:
        """Drive one message/send turn, the way a caller's A2A client would."""
        request = {
            "jsonrpc": "2.0",
            "id": f"e2e-{unique_marker()}",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "USER",
                    "messageId": unique_marker(),
                    "parts": [{"text": "hello"}],
                    "contextId": context_id,
                }
            },
        }
        return self.client.proxy.transport.post(
            f"/a2a/{agent_id}",
            headers=self.client.proxy.transport.bearer(key),
            json=request,
            response_type=_Answer,
        )

    def register(self, client: A2AClient, resources: ResourceManager, scope: str | None) -> str:
        marker = unique_marker()
        card: dict[str, object] = {
            "protocolVersion": "0.3",
            "name": f"A2A echo {marker}",
            "description": "echoes the contextId it received",
            "version": "1.0.0",
            "url": self.url,
            "skills": [A2ASkill(id="echo", name="Echo", description="echo", tags=["echo"])],
        }
        if scope is not None:
            card["litellm_context_scope"] = scope
        body = AgentRegisterBody(agent_name=f"e2e-a2a-scope-{marker}", agent_card_params=card)  # type: ignore[arg-type]
        result = client.register_agent(body)
        assert isinstance(result, Success), result
        agent_id = result.data.agent_id
        # /v1/agents answers before the data plane reloads, so wait for the card
        # to show up rather than racing the first message/send into a 404.
        for _ in range(120):
            if is_ok(client.agent_card(agent_id, key=resources.key())):
                break
            time.sleep(0.5)
        resources.defer(lambda: client.delete_agent(agent_id))
        return agent_id


@contextmanager
def _echo_agent() -> Iterator[EchoAgent]:
    agent = EchoAgent()
    agent.start()
    try:
        yield agent
    finally:
        agent.stop()


def _reply_context(result: Result[_Answer]) -> str | None:
    answer = unwrap(result)
    return answer.result.context_id


class TestA2AContextScope:
    @meta(Subject(domain=Domain.AGENTS_API, route=Route.A2A))
    @pytest.mark.covers("other.a2a.context_scope.per_caller")
    def test_two_keys_with_one_context_id_reach_the_agent_as_two_ids(
        self, client: A2AClient, resources: ResourceManager
    ) -> None:
        with _echo_agent() as agent:
            agent.client = client
            agent_id = agent.register(client, resources, scope="caller")
            alice, bob = resources.key(), resources.key()
            shared = f"ctx-{unique_marker()}"

            assert _reply_context(agent.send(alice, agent_id, shared)) == shared
            assert _reply_context(agent.send(bob, agent_id, shared)) == shared

            first, second = agent.received
            assert first is not None and second is not None
            assert first != second, "both keys landed in one conversation on the agent"
            assert first.endswith(shared) and second.endswith(shared)

    @meta(Subject(domain=Domain.AGENTS_API, route=Route.A2A))
    @pytest.mark.covers("other.a2a.context_scope.off_by_default")
    def test_agent_without_the_opt_in_still_gets_the_id_verbatim(
        self, client: A2AClient, resources: ResourceManager
    ) -> None:
        with _echo_agent() as agent:
            agent.client = client
            agent_id = agent.register(client, resources, scope=None)
            shared = f"ctx-{unique_marker()}"

            assert _reply_context(agent.send(resources.key(), agent_id, shared)) == shared
            assert agent.received == [shared]

    @meta(Subject(domain=Domain.AGENTS_API, route=Route.A2A))
    def test_served_card_does_not_advertise_the_proxy_setting(
        self, client: A2AClient, resources: ResourceManager
    ) -> None:
        with _echo_agent() as agent:
            agent.client = client
            agent_id = agent.register(client, resources, scope="caller")
            card = client.agent_card(agent_id, key=resources.key())
            assert isinstance(card, Success), card
            assert "litellm_context_scope" not in card.data.model_dump(by_alias=True, exclude_none=True)
