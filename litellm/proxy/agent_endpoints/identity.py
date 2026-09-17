import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
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


class AgentIdentityStatus(BaseModel):
    identity: EntraAgentIdentity | None = None
    last_authenticated_at: datetime | None = None


def agent_identity(params: Mapping[str, object] | None) -> EntraAgentIdentity | None:
    value: Final = params.get("identity") if params else None
    if value is None:
        return None
    return EntraAgentIdentity.model_validate(value)


def preserve_identity(incoming: Mapping[str, object], existing: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            **MappingProxyType(
                {key: existing[key] for key in ("identity", "team_id") if key in existing and key not in incoming}
            ),
            **incoming,
        }
    )


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


def match_agent_identity(agents: Sequence[AgentResponse], claims: Mapping[str, object]) -> AgentResponse | None:
    matches: Final = tuple(
        agent
        for agent in agents
        if (identity := agent_identity(agent.litellm_params)) is not None
        and claims.get("iss") == identity.issuer
        and claims.get("tid") == str(identity.tenant_id)
        and claims.get("azp") == str(identity.client_id)
    )
    if len(matches) > 1:
        raise HTTPException(403, "The authenticated Entra application matches multiple agents")
    return matches[0] if matches else None


def identity_evidence_key(agent: AgentResponse) -> str:
    identity: Final = agent_identity(agent.litellm_params)
    digest: Final = hashlib.sha256(f"{agent.agent_id}:{identity}".encode()).hexdigest()
    return f"agent-identity-authentication:{digest}"
