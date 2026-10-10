from datetime import datetime
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator

from litellm.types.llms.base import LiteLLMBaseModel

AgentExecutionMode: TypeAlias = Literal["autonomous", "delegated", "both"]


class EntraIdentityConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["microsoft_entra"]
    tenant_id: str
    client_id: str
    service_principal_id: str | None = None
    blueprint_id: str | None = Field(
        default=None,
        description="Entra Agent ID blueprint application ID. When set, only tokens issued to an agent identity created from this blueprint are accepted",
    )
    required_roles: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = Field(
        default=("user_impersonation",),
        description="Required delegated scopes. An empty list accepts any nonempty scope granted for this gateway.",
    )

    @field_validator("tenant_id", "client_id", "service_principal_id", "blueprint_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        return str(UUID(value)) if value is not None else None

    @property
    def issuer(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


class AgentIdentityBinding(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    active: bool = True
    provider: Literal["microsoft_entra"]
    tenant_id: str
    client_id: str
    service_principal_id: str | None = None
    blueprint_id: str | None = None
    issuer: str
    required_roles: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ("user_impersonation",)
    revision: str
    last_authenticated_at: datetime | None = None


class AgentSubject(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["application", "delegated_subject"]
    oid: str
    mode: Literal["autonomous", "delegated"]


class AgentIdentityFailure(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    code: Literal["identity_denied", "policy_unavailable"] = "identity_denied"
    message: str


class ManagedAgentContext(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    binding_revision: str | None = None
    mode: Literal["autonomous", "delegated"]
    user_id: str | None = None
    subject_oid: str | None = None


class VerifiedHumanSubject(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    issuer: str
    tenant_id: str
    oid: str
    user_id: str


class MicrosoftInteractiveSubject(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    issuer: str
    tenant_id: str
    oid: str


class ManagedAgentIdentityStatus(LiteLLMBaseModel):
    identity: AgentIdentityBinding | None = None
    identity_managed: bool = False
    enabled: bool = True
    execution_mode: AgentExecutionMode = "autonomous"
    last_authenticated_at: datetime | None = None
