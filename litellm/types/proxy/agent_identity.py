from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Final, Literal, TypeAlias
from uuid import UUID

from pydantic import (
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    TypeAdapter,
    field_validator,
)
from pydantic_core import core_schema

from litellm.types.llms.base import LiteLLMBaseModel

AgentExecutionMode: TypeAlias = Literal["autonomous", "delegated", "both"]


class EntraIdentityConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["microsoft_entra"]
    tenant_id: str
    client_id: str
    service_principal_id: str | None = None
    required_roles: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = Field(
        default=("user_impersonation",),
        description="Required delegated scopes. An empty list accepts any nonempty scope granted for this gateway.",
    )

    @field_validator("tenant_id", "client_id", "service_principal_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        return str(UUID(value)) if value is not None else None

    @property
    def issuer(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"


OKTA_ISSUER_PATTERN: Final = r"^https://[^/?#]+/oauth2/[^/?#]+$"


class OktaIdentityConfig(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["okta"]
    issuer: Annotated[str, Field(pattern=OKTA_ISSUER_PATTERN)]
    client_id: Annotated[str, Field(pattern=r"^\S+$")]
    required_scopes: tuple[Annotated[str, Field(pattern=r"^[\x21\x23-\x5B\x5D-\x7E]+$")], ...] = ()


def identity_config_model(value: object) -> type[EntraIdentityConfig] | type[OktaIdentityConfig]:
    match value:
        case {"provider": "okta"} | OktaIdentityConfig():
            return OktaIdentityConfig
        case _:
            return EntraIdentityConfig


class _RouteByProvider:
    def __get_pydantic_core_schema__(self, source: object, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        okta: Final = TypeAdapter(OktaIdentityConfig)

        def route(
            value: object, entra: Callable[[object], EntraIdentityConfig]
        ) -> EntraIdentityConfig | OktaIdentityConfig:
            if identity_config_model(value) is OktaIdentityConfig:
                return okta.validate_python(value)
            return entra(value)

        return core_schema.no_info_wrap_validator_function(
            route, handler.generate_schema(EntraIdentityConfig), json_schema_input_schema=handler(source)
        )


AgentIdentityConfig: TypeAlias = Annotated[EntraIdentityConfig | OktaIdentityConfig, _RouteByProvider()]
AGENT_IDENTITY_CONFIG: Final[TypeAdapter[EntraIdentityConfig | OktaIdentityConfig]] = TypeAdapter(AgentIdentityConfig)


class AgentIdentityBinding(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    active: bool = True
    provider: Literal["microsoft_entra", "okta"]
    tenant_id: str
    client_id: str
    service_principal_id: str | None = None
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
