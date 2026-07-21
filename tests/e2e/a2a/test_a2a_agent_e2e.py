"""A2A agents end to end, against a live proxy.

An admin registers an agent whose card pins an A2A protocol version and whose
litellm_params route it through the completion bridge; a caller then discovers the
proxy-owned card and drives it over A2A JSON-RPC. These tests assert the recorded
state (the agent persists, a spend row lands) and the enforced behavior (the served
card points back at the proxy, message/send returns a real completion in the pinned
protocol version, and an unsupported version is refused at registration).
"""

from __future__ import annotations

import pytest

from a2a_client import (
    A2ABridgeParams,
    A2AClient,
    A2AJsonRpcRequest,
    A2AMessageSendParams,
    A2AOutboundMessage,
    A2ASkill,
    A2ATextPart,
    AgentCardParams,
    AgentRegisterBody,
    AgentResponse,
)
from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager

BRIDGE = A2ABridgeParams(custom_llm_provider="anthropic", model="claude-haiku-4-5")

pytestmark = pytest.mark.e2e


def _register(client: A2AClient, resources: ResourceManager, protocol_version: str) -> AgentResponse:
    marker = unique_marker()
    body = AgentRegisterBody(
        agent_name=f"e2e-a2a-{marker}",
        agent_card_params=AgentCardParams(
            protocol_version=protocol_version,
            name=f"E2E A2A {marker}",
            description="e2e agent backed by the litellm completion bridge",
            version="1.0.0",
            skills=[A2ASkill(id="chat", name="Chat", description="general chat", tags=["chat"])],
        ),
        litellm_params=BRIDGE,
    )
    agent = unwrap(client.register_agent(body))
    resources.defer(lambda: client.delete_agent(agent.agent_id))
    return agent


def _ask(text: str) -> A2AJsonRpcRequest:
    return A2AJsonRpcRequest(
        id=f"e2e-{unique_marker()}",
        params=A2AMessageSendParams(
            message=A2AOutboundMessage(parts=[A2ATextPart(text=text)], message_id=unique_marker())
        ),
    )


class TestA2AAgentLifecycle:
    @pytest.mark.covers("other.a2a.register.persists")
    def test_register_persists(self, client: A2AClient, resources: ResourceManager) -> None:
        agent = _register(client, resources, "0.3")
        fetched = unwrap(client.get_agent(agent.agent_id))
        assert fetched.agent_id == agent.agent_id
        assert fetched.agent_name == agent.agent_name
        assert fetched.agent_card_params.protocol_version == "0.3"

    @pytest.mark.covers("other.a2a.discovery.proxy_fronted_card")
    def test_discovery_card_is_proxy_fronted(self, client: A2AClient, resources: ResourceManager, scoped_key: str) -> None:
        agent = _register(client, resources, "0.3")
        card = unwrap(client.agent_card(agent.agent_id, scoped_key))
        assert card.url is not None and card.url.endswith(f"/a2a/{agent.agent_id}")
        assert card.security_schemes is not None
        scheme = next(iter(card.security_schemes.values()))
        assert scheme.scheme == "bearer"
        assert card.supported_interfaces is not None
        assert card.supported_interfaces[0].url == card.url

    @pytest.mark.covers("other.a2a.message_send.bridge_invokes")
    def test_message_send_runs_completion_bridge(self, client: A2AClient, resources: ResourceManager, scoped_key: str) -> None:
        agent = _register(client, resources, "0.3")
        request = _ask("Reply with exactly the word PONG and nothing else")
        response = unwrap(client.send_message(agent.agent_id, scoped_key, request))
        assert response.error is None
        assert response.result is not None
        assert "PONG" in response.result.text.upper()

        rows = client.proxy.poll_logs_for_request_id(request.id)
        assert rows, f"no spend log row landed for a2a request {request.id}"
        assert rows[0].call_type == "asend_message"
        assert rows[0].model == f"a2a_agent/{agent.agent_card_params.name}"

    @pytest.mark.covers("other.a2a.version.serves_pinned_0_3")
    def test_pinned_v0_3_serves_flat_message_shape(self, client: A2AClient, resources: ResourceManager, scoped_key: str) -> None:
        agent = _register(client, resources, "0.3")
        request = _ask("Say hi in one word")
        result = unwrap(client.send_message(agent.agent_id, scoped_key, request)).result
        assert result is not None
        assert not result.is_nested_v1_shape
        assert result.kind == "message"
        assert result.role == "agent"
        assert result.text != ""

    @pytest.mark.covers("other.a2a.version.serves_pinned_1_0")
    def test_pinned_v1_0_serves_nested_message_shape(self, client: A2AClient, resources: ResourceManager, scoped_key: str) -> None:
        agent = _register(client, resources, "1.0")
        request = _ask("Say hi in one word")
        result = unwrap(client.send_message(agent.agent_id, scoped_key, request)).result
        assert result is not None
        assert result.is_nested_v1_shape
        assert result.message is not None
        assert result.message.role == "ROLE_AGENT"
        assert result.text != ""

    @pytest.mark.covers("other.a2a.register.unsupported_version_rejected")
    def test_unsupported_protocol_version_rejected(self, client: A2AClient, resources: ResourceManager) -> None:
        marker = unique_marker()
        body = AgentRegisterBody(
            agent_name=f"e2e-a2a-bad-{marker}",
            agent_card_params=AgentCardParams(
                protocol_version="9.9",
                name=f"E2E A2A bad {marker}",
                description="unsupported version",
                version="1.0.0",
                skills=[A2ASkill(id="chat", name="Chat", description="c", tags=["chat"])],
            ),
            litellm_params=BRIDGE,
        )
        result = client.register_agent(body)
        match result:
            case UnknownApiError(status_code=status, body=detail):
                assert status == 400
                assert "protocolVersion" in detail
            case _:
                pytest.fail(f"expected 400 for unsupported protocolVersion, got {result}")
