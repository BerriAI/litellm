from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.types.agents import AgentResponse


class EntraAgentIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["microsoft_entra"]
    tenant_id: UUID
    client_id: UUID

    @property
    def issuer(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


def agent_identity(params: Mapping[str, object] | None) -> EntraAgentIdentity | None:
    value: Final = params.get("identity") if params else None
    if value is None:
        return None
    return EntraAgentIdentity.model_validate(value)


def preserve_identity(incoming: Mapping[str, object], existing: Mapping[str, object]) -> Mapping[str, object]:
    if "identity" in incoming or "identity" not in existing:
        return incoming
    return MappingProxyType({**incoming, "identity": existing["identity"]})


def validate_identity_binding(
    params: Mapping[str, object] | None,
    agents: Sequence[AgentResponse],
    trusted_issuers: Sequence[str],
    agent_id: str | None = None,
) -> None:
    try:
        identity: Final = agent_identity(params)
    except ValidationError as exc:
        raise HTTPException(
            400, "Invalid agent identity: select Microsoft Entra ID and provide tenant and client UUIDs"
        ) from exc
    if identity is None:
        return
    if identity.issuer not in trusted_issuers:
        raise HTTPException(400, "This Entra tenant is not configured for trusted JWT authentication on the gateway")
    if any(agent.agent_id != agent_id and agent_identity(agent.litellm_params) == identity for agent in agents):
        raise HTTPException(409, "This Entra application is already bound to another agent")
