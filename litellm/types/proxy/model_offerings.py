from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal, cast  # noqa: TID251  # Pydantic runtime schema annotations

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from litellm.types.proxy.model_metadata import GatewayModelMetadata


class OfferingModelInfo(GatewayModelMetadata):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["chat", "completion", "embedding"] | None = None

    @field_validator("*", mode="before")
    @classmethod
    def strict_configured_metadata(cls, value: object, info: ValidationInfo) -> object:
        if info.field_name is not None:
            TypeAdapter[object](cast(object, cls.model_fields[info.field_name].rebuild_annotation())).validate_python(
                value, strict=True
            )
        return value


class SupplierConnection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai", "openrouter", "vercel_ai_gateway", "chatgpt"]
    api_base: AnyHttpUrl | None = None
    api_key: SecretStr | None = None
    headers: Mapping[str, SecretStr] = Field(default_factory=lambda: MappingProxyType({}))

    @field_validator("headers")
    @classmethod
    def immutable_headers(cls, value: Mapping[str, SecretStr]) -> Mapping[str, SecretStr]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def generic_base_required(self) -> "SupplierConnection":
        if self.provider == "openai" and self.api_base is None:
            raise ValueError("Generic OpenAI connections require api_base")
        if self.provider == "chatgpt" and (self.api_key is not None or self.headers):
            raise ValueError(
                "ChatGPT connections use the active native OAuth account rather than configured keys or headers"
            )
        return self


class ModelOffering(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Annotated[str, Field(min_length=1, strict=True)]
    source: Literal["auto", "manual"]
    provider: Annotated[str, Field(min_length=1, strict=True)]
    upstream_model: Annotated[str, Field(min_length=1, strict=True)]
    enabled: bool = Field(default=True, strict=True)
    model_info: OfferingModelInfo = Field(default_factory=OfferingModelInfo)

    @field_validator("model_name", "provider", "upstream_model")
    @classmethod
    def exact_identifier(cls, value: str) -> str:
        if not value.strip() or value != value.strip() or "*" in value:
            raise ValueError("Offering identifiers must be nonempty exact names without wildcards")
        return value


class ModelOfferingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1]
    config_poll_seconds: int = Field(default=5, ge=1, le=300, strict=True)
    inventory_poll_seconds: int = Field(default=300, ge=1, le=86400, strict=True)
    providers: Mapping[str, SupplierConnection]
    offerings: tuple[ModelOffering, ...]

    @field_validator("version", mode="before")
    @classmethod
    def supported_version(cls, value: object) -> Literal[1]:
        if isinstance(value, int) and not isinstance(value, bool) and value == 1:
            return 1
        raise ValueError("Offering config version must be integer 1")

    @field_validator("providers")
    @classmethod
    def immutable_providers(cls, value: Mapping[str, SupplierConnection]) -> Mapping[str, SupplierConnection]:
        if any(not name.strip() or name != name.strip() or "*" in name for name in value):
            raise ValueError("Connection names must be nonempty exact names without wildcards")
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def known_connections_and_unique_names(self) -> "ModelOfferingsConfig":
        names: Final = tuple(offering.model_name for offering in self.offerings)
        if len(frozenset(names)) != len(names):
            raise ValueError("Each offering must have a unique model_name")
        if any(offering.provider not in self.providers for offering in self.offerings):
            raise ValueError("Each offering must reference a configured connection")
        return self
