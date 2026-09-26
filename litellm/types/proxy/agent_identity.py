from datetime import datetime
from typing import Literal, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

AgentExecutionMode: TypeAlias = Literal["autonomous", "delegated", "both"]


class EntraIdentityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["microsoft_entra"]
    tenant_id: str
    client_id: str
    provisioning_source_id: str | None = None
    service_principal_id: str | None = None
    required_roles: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ("user_impersonation",)

    @field_validator("tenant_id", "client_id", "service_principal_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        return str(UUID(value)) if value is not None else None

    @property
    def issuer(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


class AgentIdentityBinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    active: bool = True
    provider: Literal["microsoft_entra"]
    tenant_id: str
    client_id: str
    provisioning_source_id: str | None = None
    service_principal_id: str | None = None
    issuer: str
    required_roles: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ("user_impersonation",)
    revision: str
    last_authenticated_at: datetime | None = None


class AgentBudgetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float = Field(ge=0, allow_inf_nan=False)
    budget_duration: str | None = None


class AgentBudgetState(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_id: str
    max_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


class AgentSubject(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["application", "delegated_subject", "agent_user"]
    oid: str
    mode: Literal["autonomous", "delegated"]


class AgentIdentityFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: Literal["identity_denied", "policy_unavailable"] = "identity_denied"
    message: str


class ManagedAgentContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    binding_revision: str | None = None
    mode: Literal["autonomous", "delegated"]
    user_id: str | None = None
    subject_oid: str | None = None


class VerifiedHumanSubject(BaseModel):
    model_config = ConfigDict(frozen=True)

    issuer: str
    tenant_id: str
    oid: str
    user_id: str


class MicrosoftInteractiveSubject(BaseModel):
    model_config = ConfigDict(frozen=True)

    issuer: str
    tenant_id: str
    oid: str


class ManagedAgentIdentityStatus(BaseModel):
    directory_active: bool = True
    directory_access_group_ids: tuple[str, ...] | None = None
    identity: AgentIdentityBinding | None = None
    identity_managed: bool = False
    enabled: bool = True
    execution_mode: AgentExecutionMode = "autonomous"
    last_authenticated_at: datetime | None = None


class VerifiedAgentSubject(BaseModel):
    model_config = ConfigDict(frozen=True)

    issuer: str
    tenant_id: str
    oid: str
    agent_id: str
    parent_client_id: str
    scim_resource_id: str
