"""
Credential table models.

These are the canonical credential types for the proxy. They live in the model
layer; ``litellm.types.utils`` re-exports them for backwards compatibility.
"""

from collections.abc import Mapping
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from litellm.types.llms.base import LiteLLMBaseModel

CredentialSource: TypeAlias = Literal["db", "config"]


class CredentialBase(LiteLLMBaseModel):
    credential_name: str
    display_name: str | None = None
    credential_info: dict


class CredentialItem(CredentialBase):
    credential_values: dict[  # mutable-ok: values are decrypted and merged in place
        str, object
    ]
    # PATCH-only instruction naming keys to drop from the stored credential_values. It describes an
    # edit rather than the credential, so it stays out of dumps: those feed config loading, the DB
    # write, and the in-memory list, none of which have a place for it.
    credential_values_to_delete: tuple[str, ...] | None = Field(default=None, exclude=True)
    source: CredentialSource = Field(default="db", exclude=True)


class CredentialView(CredentialBase):
    model_config = ConfigDict(frozen=True)

    credential_values: Mapping[str, object]
    source: CredentialSource


class CreateCredentialItem(CredentialBase):
    credential_values: dict | None = None
    model_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_credential_params(cls, values):
        if not values.get("credential_values") and not values.get("model_id"):
            raise ValueError("Either credential_values or model_id must be set")
        return values


class UpdateCredentialItem(LiteLLMBaseModel):
    credential_name: str | None = None
    display_name: str | None = None
    credential_info: Mapping[str, object]
    credential_values: Mapping[str, object] | None = None
    model_id: str | None = None
    credential_values_to_delete: tuple[str, ...] | None = None


class UserProviderConnection(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    credential_name: str
    provider: str
    connected: bool
    github_login: str | None = None
    connected_at: str | None = None


class UserProviderConnectionsResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    connections: list[UserProviderConnection]  # mutable-ok: pydantic response model field


class UserConnectionStartResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    user_code: str
    verification_uri: str
    expires_in: int
    interval: int
    flow_handle: str


class UserConnectionPollRequest(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    flow_handle: str


class UserConnectionFlowHandle(LiteLLMBaseModel):
    """The stateless device-flow ticket: the pending ``device_code`` travels
    encrypted inside ``flow_handle`` instead of a worker-local cache entry, so a
    poll landing on any worker can complete the connection.``"""

    model_config = ConfigDict(frozen=True)

    user_id: str
    credential_name: str
    device_code: str = Field(repr=False)
    interval: int
    expires_at: float


UserConnectionPollStatus: TypeAlias = Literal[
    "pending", "slow_down", "expired", "denied", "connected", "no_copilot_seat"
]


class UserConnectionPollResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status: UserConnectionPollStatus
    interval: int | None = None
    github_login: str | None = None


class UserConnectionDeleteResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
