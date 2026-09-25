from typing import Final
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr

SCIM_AGENT_USER_SCHEMA: Final = "urn:ietf:params:scim:schemas:extension:litellmAgent:2.0:User"


def canonical_directory_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError:
        return value


class SCIMAgentUser(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identityParentId: UUID


class SCIMGroupMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    external_group_id: UUID
    access_group_ids: tuple[str, ...] = Field(min_length=1)


class SCIMSourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: str = Field(min_length=1)
    tenant_id: UUID
    enabled: bool = True
    group_mappings: tuple[SCIMGroupMapping, ...] = ()


class SCIMSourceCreate(SCIMSourceConfig):
    provisioning_token: SecretStr


class SCIMSourceResponse(SCIMSourceConfig):
    source_id: str
