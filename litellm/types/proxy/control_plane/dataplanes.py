from typing import Literal, TypeAlias

from pydantic import ConfigDict

from litellm.types.utils import LiteLLMBaseModel

CredentialKindName: TypeAlias = Literal["oidc_exchange", "service_token"]
DataplaneHealthStatus: TypeAlias = Literal["ok", "unreachable", "degraded"]


class DataplaneResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    url: str
    credentials: tuple[CredentialKindName, ...]


class DataplaneListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[DataplaneResponse, ...]


class DataplaneHealthResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    status: DataplaneHealthStatus
    http_status: int | None
    detail: str | None


class DataplaneIdentityResponse(LiteLLMBaseModel):
    """Who the dataplane thinks the caller is, the end-to-end proof that identity survived the hop."""

    model_config = ConfigDict(frozen=True)

    id: str
    credential: CredentialKindName | None
    user_id: str | None
    user_email: str | None
    user_role: str | None
