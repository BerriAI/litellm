import enum
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal  # noqa: TID251  # allowed_routes and metadata stay Any for existing callers

from pydantic import Field, PositiveInt, field_validator, model_validator
from typing_extensions import TypeIs  # noqa: TID251  # narrows the raw payload a before-validator receives

from litellm.types.proxy.management_endpoints.request_base import GenerateRequestBase
from litellm.types.router import UpdateRouterConfig
from litellm.types.utils import LiteLLMPydanticObjectBase


class AllowedVectorStoreIndexItem(LiteLLMPydanticObjectBase):
    index_name: str
    index_permissions: list[Literal["read", "write"]]


class KeyRequestBase(GenerateRequestBase):
    key: str | None = None
    tpd_limit: int | None = None
    default_estimated_output_tokens: PositiveInt | None = None
    default_estimated_output_tokens_per_model: Mapping[str, PositiveInt] | None = None
    budget_id: str | None = None
    end_user_budget_id: str | None = None
    tags: list[str] | None = None
    disable_global_guardrails: bool | None = None
    enable_prompt_caching: bool | None = None
    throttle_on_budget_exceeded: bool | None = None
    enforced_params: list[str] | None = None
    allowed_routes: list[Any] | None = []
    allowed_passthrough_routes: list[object] | None = None
    denied_passthrough_routes: list[str] | None = None
    allowed_vector_store_indexes: list[AllowedVectorStoreIndexItem] | None = None
    rpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating rpm
    )
    tpm_limit_type: Literal["guaranteed_throughput", "best_effort_throughput", "dynamic"] | None = (
        None  # raise an error if 'guaranteed_throughput' is set and we're overallocating tpm
    )
    router_settings: UpdateRouterConfig | None = None
    access_group_ids: list[str] | None = None


class LiteLLMKeyType(str, enum.Enum):
    """
    Enum for key types that determine what routes a key can access
    """

    LLM_API = "llm_api"  # Can call LLM API routes (chat/completions, embeddings, etc.)
    MANAGEMENT = "management"  # Can call management routes (user/team/key management)
    READ_ONLY = "read_only"  # Can only call info/read routes
    DEFAULT = "default"  # Uses default allowed routes


class GenerateKeyRequest(KeyRequestBase):
    soft_budget: float | None = None
    send_invite_email: bool | None = None
    key_type: LiteLLMKeyType | None = Field(
        default=LiteLLMKeyType.DEFAULT,
        description="Type of key that determines default allowed routes.",
    )
    auto_rotate: bool | None = Field(default=False, description="Whether this key should be automatically rotated")
    rotation_interval: str | None = Field(
        default=None,
        description="How often to rotate this key (e.g., '30d', '90d'). Required if auto_rotate=True",
    )
    organization_id: str | None = None
    project_id: str | None = None

    @field_validator("team_id", "organization_id", "project_id", mode="before")
    @classmethod
    def treat_cleared_id_as_unset(cls, v: object) -> object:
        if v == "":
            return None
        return v


def _is_mapping(value: object) -> TypeIs[Mapping[object, object]]:  # guard-ok: raw before-validator payload
    return isinstance(value, Mapping)


class UpdateKeyRequest(KeyRequestBase):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    duration: str | None = None
    spend: float | None = None
    soft_budget: float | None = None
    metadata: dict[Any, Any] | None = None
    temp_budget_increase: float | None = None
    temp_budget_expiry: datetime | None = None
    auto_rotate: bool | None = None
    rotation_interval: str | None = None
    organization_id: str | None = None

    project_id: str | None = Field(
        default=None,
        description="Omit to retain the project, or send null to detach. Assigning a different project is not supported.",
    )

    @model_validator(mode="before")
    @classmethod
    def drop_blank_team_id(cls, values: object) -> object:
        if _is_mapping(values) and values.get("team_id") == "":
            return MappingProxyType({k: v for k, v in values.items() if k != "team_id"})
        return values

    @field_validator("organization_id", mode="before")
    @classmethod
    def treat_cleared_organization_id_as_unset(cls, v: object) -> object:
        if v == "":
            return None
        return v

    @model_validator(mode="after")
    def validate_temp_budget(self) -> "UpdateKeyRequest":
        if self.temp_budget_increase is not None or self.temp_budget_expiry is not None:
            if self.temp_budget_increase is None or self.temp_budget_expiry is None:
                raise ValueError("temp_budget_increase and temp_budget_expiry must be set together")
        return self

    @model_validator(mode="after")
    def validate_key_identifier(self) -> "UpdateKeyRequest":
        if self.key is None and self.key_alias is None:
            raise ValueError("either key or key_alias must be provided")
        return self


class RegenerateKeyRequest(GenerateKeyRequest):
    # This needs to be different from UpdateKeyRequest, because "key" is optional for this
    key: str | None = None
    new_key: str | None = None
    duration: str | None = None
    spend: float | None = None
    metadata: dict[Any, Any] | None = None
    new_master_key: str | None = None
    grace_period: str | None = None  # Duration to keep old key valid (e.g. "24h", "2d"); None = immediate revoke
