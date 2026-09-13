from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Final, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .recorded_http import RecordedResponse

JsonObject = dict[str, JsonValue]


class FixtureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True)


class SdkInputBase(FixtureModel):
    fixture_only_fields: ClassVar[tuple[str, ...]] = ()

    def as_sdk_kwargs(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            self.model_dump(
                mode="python",
                exclude_unset=True,
                exclude=set(self.fixture_only_fields),
            ),
        )

    def canonical_input(self) -> dict[str, object]:
        dumped: Final = cast(dict[str, object], self.model_dump(mode="json", exclude_unset=True))
        fixture_fields: Final = {field: getattr(self, field) for field in self.fixture_only_fields}
        return {**fixture_fields, **dumped}


class JsonSchemaDefinition(FixtureModel):
    name: str
    description: str | None = None
    schema_definition: JsonObject = Field(alias="schema")
    strict: bool = False


class JsonSchemaResponseFormat(FixtureModel):
    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


InputT = TypeVar("InputT", bound=SdkInputBase)


class ParityCase(FixtureModel, Generic[InputT]):
    litellm_input: InputT
    provider_responses: tuple[RecordedResponse, ...]

    @model_validator(mode="before")
    @classmethod
    def load_legacy_single_response(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        migrated: Final = dict(cast(Mapping[str, object], value))
        provider_response: Final = migrated.pop("provider_response", None)
        if "provider_responses" not in migrated and provider_response is not None:
            migrated["provider_responses"] = (provider_response,)
        return migrated
