"""Client for the proxy's A2A (agent-to-agent) surface.

An A2A agent is registered admin-side via POST /v1/agents with an agent card and
litellm_params; the proxy fronts it at /a2a/{id}, serving a proxy-owned agent card
at /.well-known/agent-card.json and accepting A2A JSON-RPC calls at /a2a/{id}. This
suite registers agents backed by the litellm_completion_bridge (custom_llm_provider
+ model), so message/send runs a real provider completion and comes back in the
agent's pinned A2A protocol version. The A2A request/response models are co-located
here because only this suite uses them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from e2e_http import NoBody, Result, is_ok
from proxy_client import ProxyClient


class A2ACapabilities(BaseModel):
    streaming: bool | None = None
    push_notifications: bool | None = Field(default=None, serialization_alias="pushNotifications")


class A2ASkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]


class AgentCardParams(BaseModel):
    """The upstream agent card an admin registers. `protocolVersion` is the field the
    proxy validates against SUPPORTED_A2A_PROTOCOL_VERSIONS on registration."""

    protocol_version: str = Field(serialization_alias="protocolVersion")
    name: str
    description: str
    version: str
    capabilities: A2ACapabilities = A2ACapabilities()
    skills: list[A2ASkill]
    default_input_modes: list[str] = Field(default=["text"], serialization_alias="defaultInputModes")
    default_output_modes: list[str] = Field(default=["text"], serialization_alias="defaultOutputModes")


class A2ABridgeParams(BaseModel):
    """litellm_params that route the agent through the completion bridge: an A2A
    message/send is transformed into a litellm.acompletion against this provider."""

    model_config = ConfigDict(protected_namespaces=())

    custom_llm_provider: str
    model: str


class AgentRegisterBody(BaseModel):
    agent_name: str
    agent_card_params: AgentCardParams
    litellm_params: A2ABridgeParams


class A2ASecurityScheme(BaseModel):
    type: str
    scheme: str


class A2AInterface(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: str
    protocol_version: str | None = Field(default=None, alias="protocolVersion")


class ServedAgentCard(BaseModel):
    """The proxy-owned card, either nested under a registration response's
    `agent_card_params` or served raw at /.well-known/agent-card.json. The proxy
    rewrites `url`/`supportedInterfaces` to itself and replaces the security scheme
    with its own virtual-key bearer scheme."""

    model_config = ConfigDict(populate_by_name=True)

    protocol_version: str = Field(alias="protocolVersion")
    name: str
    url: str | None = None
    security_schemes: dict[str, A2ASecurityScheme] | None = Field(default=None, alias="securitySchemes")
    security: list[dict[str, list[str]]] | None = None
    supported_interfaces: list[A2AInterface] | None = Field(default=None, alias="supportedInterfaces")


class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    agent_card_params: ServedAgentCard


class A2ATextPart(BaseModel):
    kind: str = "text"
    text: str


class A2AOutboundMessage(BaseModel):
    role: str = "user"
    parts: list[A2ATextPart]
    message_id: str = Field(serialization_alias="messageId")


class A2AMessageSendParams(BaseModel):
    message: A2AOutboundMessage


class A2AJsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    method: str = "message/send"
    params: A2AMessageSendParams


class A2AResponsePart(BaseModel):
    kind: str | None = None
    text: str | None = None


class A2AResponseMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: str | None = Field(default=None, alias="messageId")
    role: str | None = None
    parts: list[A2AResponsePart] = []


class A2AResult(BaseModel):
    """A message/send result. In 0.3 the message fields sit directly on the result
    (`kind`/`role`/`parts`); in 1.0 they are nested under `message`. `text` reads the
    agent's reply from whichever shape the served version produced."""

    model_config = ConfigDict(populate_by_name=True)

    kind: str | None = None
    role: str | None = None
    message_id: str | None = Field(default=None, alias="messageId")
    parts: list[A2AResponsePart] = []
    message: A2AResponseMessage | None = None

    @property
    def text(self) -> str:
        parts = self.message.parts if self.message is not None else self.parts
        return "".join(part.text or "" for part in parts)

    @property
    def is_nested_v1_shape(self) -> bool:
        return self.message is not None


class A2AError(BaseModel):
    code: int
    message: str


class A2AResponse(BaseModel):
    jsonrpc: str
    id: str | None = None
    result: A2AResult | None = None
    error: A2AError | None = None


@dataclass(frozen=True, slots=True)
class A2AClient:
    proxy: ProxyClient

    def register_agent(self, body: AgentRegisterBody) -> Result[AgentResponse]:
        return self.proxy.transport.post(
            "/v1/agents",
            headers=self.proxy.transport.master,
            json=body,
            response_type=AgentResponse,
        )

    def get_agent(self, agent_id: str) -> Result[AgentResponse]:
        return self.proxy.transport.get(
            f"/v1/agents/{agent_id}",
            headers=self.proxy.transport.master,
            params=NoBody(),
            response_type=AgentResponse,
        )

    def delete_agent(self, agent_id: str) -> None:
        result = self.proxy.transport.delete(
            f"/v1/agents/{agent_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
        if not is_ok(result):
            warnings.warn(f"delete_agent({agent_id!r}) failed: {result}", stacklevel=2)

    def agent_card(self, agent_id: str, key: str) -> Result[ServedAgentCard]:
        return self.proxy.transport.get(
            f"/a2a/{agent_id}/.well-known/agent-card.json",
            headers=self.proxy.transport.bearer(key),
            params=NoBody(),
            response_type=ServedAgentCard,
        )

    def send_message(self, agent_id: str, key: str, body: A2AJsonRpcRequest) -> Result[A2AResponse]:
        return self.proxy.transport.post(
            f"/a2a/{agent_id}",
            headers=self.proxy.transport.bearer(key),
            json=body,
            response_type=A2AResponse,
        )


def build_a2a_client(proxy: ProxyClient) -> A2AClient:
    return A2AClient(proxy=proxy)
