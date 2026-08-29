from __future__ import annotations

from typing import Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from tests.test_litellm._recorded_http import RecordedResponse

JsonObject = dict[str, JsonValue]


class FixtureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True)


class SdkInputBase(FixtureModel):
    def as_sdk_kwargs(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="python", exclude_unset=True))

    def canonical_input(self) -> dict[str, object]:
        return cast(dict[str, object], self.model_dump(mode="json", exclude_unset=True))


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
    provider_response: RecordedResponse
