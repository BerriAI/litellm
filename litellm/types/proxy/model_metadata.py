from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, ValidationInfo, field_validator

_STRING_VALUES_ADAPTER: Final = TypeAdapter(tuple[str, ...])
_VALIDATED_FIELDS_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class GatewayModelMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    context_window: int | None = Field(
        default=None, gt=0, description="Maximum combined input and generated output tokens"
    )
    max_input_tokens: int | None = Field(default=None, gt=0, description="Independent maximum input tokens")
    max_output_tokens: int | None = Field(default=None, gt=0, description="Independent maximum generated output tokens")
    supports_function_calling: bool | None = None
    supports_parallel_function_calling: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_effort_levels: Sequence[str] | None = Field(
        default=None, description="Exact effort values this route accepts"
    )
    default_reasoning_effort: str | None = Field(default=None, description="Effort used when the request omits it")
    supported_endpoints: Sequence[str] | None = Field(
        default=None, description="Inference endpoints exposed by this route"
    )
    supported_modalities: Sequence[str] | None = None
    supported_output_modalities: Sequence[str] | None = None

    @field_validator("context_window", "max_input_tokens", "max_output_tokens", mode="before")
    @classmethod
    def positive_limit(cls, value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    @field_validator(
        "supports_function_calling", "supports_parallel_function_calling", "supports_reasoning", mode="before"
    )
    @classmethod
    def known_boolean(cls, value: object) -> bool | None:
        return value if isinstance(value, bool) else None

    @field_validator(
        "reasoning_effort_levels",
        "supported_endpoints",
        "supported_modalities",
        "supported_output_modalities",
        mode="before",
    )
    @classmethod
    def known_string_list(cls, value: object) -> Sequence[str] | None:
        try:
            return tuple(dict.fromkeys(_STRING_VALUES_ADAPTER.validate_python(value)))
        except ValidationError:
            return None

    @field_validator("default_reasoning_effort", mode="before")
    @classmethod
    def known_string(cls, value: object) -> str | None:
        return value if isinstance(value, str) else None

    @field_validator("reasoning_effort_levels")
    @classmethod
    def enabled_reasoning_levels(cls, value: Sequence[str] | None, info: ValidationInfo) -> Sequence[str] | None:
        data: Final = _VALIDATED_FIELDS_ADAPTER.validate_python(info.data)
        return None if data.get("supports_reasoning") is False else value

    @field_validator("default_reasoning_effort")
    @classmethod
    def supported_default(cls, value: str | None, info: ValidationInfo) -> str | None:
        data: Final = _VALIDATED_FIELDS_ADAPTER.validate_python(info.data)
        levels: Final = data.get("reasoning_effort_levels")
        if data.get("supports_reasoning") is False:
            return None
        return value if levels is None or isinstance(levels, (list, tuple)) and value in levels else None


def resolve_gateway_model_metadata(
    catalog: Mapping[str, object], configured: Mapping[str, object]
) -> GatewayModelMetadata:
    values: Final = {
        field: configured[field] if configured.get(field) is not None else catalog.get(field)
        for field in GatewayModelMetadata.model_fields
    }
    return GatewayModelMetadata.model_validate(values)
