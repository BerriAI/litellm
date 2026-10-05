from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, cast  # noqa: TID251  # Pydantic runtime schema boundary

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
)

_STRING_VALUES_ADAPTER: Final = TypeAdapter(tuple[str, ...])
_VALIDATED_FIELDS_ADAPTER: Final = TypeAdapter(Mapping[str, object])


class ModelRequestDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_token_budget: Annotated[int, Field(gt=0, strict=True)] | None = None
    output_token_budget_by_reasoning_effort: Mapping[str, Annotated[int, Field(gt=0, strict=True)]] | None = None

    @field_validator("output_token_budget_by_reasoning_effort")
    @classmethod
    def immutable_budgets(cls, value: Mapping[str, int] | None) -> Mapping[str, int] | None:
        return MappingProxyType(dict(value)) if value is not None else None

    @field_serializer("output_token_budget_by_reasoning_effort")
    def serialize_budgets(self, value: Mapping[str, int] | None) -> Mapping[str, int] | None:
        return dict(value) if value is not None else None

    def output_budget(self, effort: str | None) -> int | None:
        return (
            (self.output_token_budget_by_reasoning_effort or {}).get(effort, self.output_token_budget)
            if effort is not None
            else self.output_token_budget
        )


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
    request_defaults: ModelRequestDefaults | None = Field(
        default=None, description="Configured request defaults, independent of supplier capability limits"
    )

    @classmethod
    def validate_field_input(cls, field_name: str, value: object) -> None:
        TypeAdapter[object](
            cast(object, cls.model_fields[field_name].rebuild_annotation())  # cast-ok: runtime Pydantic field schema
        ).validate_python(value, strict=True)

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
