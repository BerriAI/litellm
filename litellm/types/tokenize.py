from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, PrivateAttr, TypeAdapter, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticCustomError

from litellm.types.llms.base import LiteLLMPydanticObjectBase

TokenizeRepresentation: TypeAlias = Literal["token_ids", "tokens"]
_INCLUDE_ARRAY: Final = TypeAdapter(tuple[object, ...])


def _include_array(value: object) -> object:
    if not isinstance(value, (list, tuple)):
        raise PydanticCustomError("include_type", "include must be an array of token_ids or tokens")
    return _INCLUDE_ARRAY.validate_python(value)


def _unique_include(value: tuple[TokenizeRepresentation, ...]) -> tuple[TokenizeRepresentation, ...]:
    if len(value) != len(frozenset(value)):
        raise ValueError("include entries must be unique")
    return value


TokenizeInclude: TypeAlias = Annotated[
    tuple[TokenizeRepresentation, ...],
    BeforeValidator(_include_array),
    AfterValidator(_unique_include),
    Field(json_schema_extra={"uniqueItems": True}),
]


class TokenizeRequest(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str = Field(min_length=1)
    prompt: str
    include: Annotated[TokenizeInclude | SkipJsonSchema[None], BeforeValidator(_include_array)] = Field(
        default_factory=lambda: None, exclude_if=lambda value: value is None
    )
    add_special_tokens: bool | SkipJsonSchema[None] = Field(
        default_factory=lambda: None, exclude_if=lambda value: value is None
    )

    @field_validator("add_special_tokens", mode="before")
    @classmethod
    def explicit_special_tokens(cls, value: object) -> object:
        if not isinstance(value, bool):
            raise PydanticCustomError("special_tokens_type", "add_special_tokens must be a boolean when supplied")
        return value


class TokenizeResponse(LiteLLMPydanticObjectBase):
    model_config = ConfigDict(frozen=True)
    model: str
    count: int = Field(ge=0)
    token_ids: Sequence[int] | None = None
    tokens: Sequence[str] | None = None

    _hidden_params: Mapping[str, object] = PrivateAttr(default_factory=dict)
