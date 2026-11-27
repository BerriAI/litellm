"""Models for the experimental secrets vault."""

from datetime import datetime
from typing import Annotated, Final, TypeAlias

from pydantic import ConfigDict, Field, StringConstraints

from litellm.types.llms.base import LiteLLMBaseModel

VAULT_SECRET_NAME_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"
VAULT_SECRET_VALUE_MAX_LENGTH: Final = 16384

VaultSecretValue: TypeAlias = Annotated[str, StringConstraints(min_length=1, max_length=VAULT_SECRET_VALUE_MAX_LENGTH)]


class VaultSecretWrite(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: VaultSecretValue = Field(repr=False)
    description: str | None = Field(default=None, max_length=500)


class VaultSecretMetadata(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    secret_name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None


class VaultSecretRevealed(VaultSecretMetadata):
    value: str = Field(repr=False)


class VaultSecretListResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    secrets: tuple[VaultSecretMetadata, ...]


class VaultSecretDeleteResponse(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    secret_name: str
    deleted: bool
