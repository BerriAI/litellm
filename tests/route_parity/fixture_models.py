from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tests.route_parity.recorded_http import RecordedResponse

JsonObject = dict[str, JsonValue]


class FixtureModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True)


class SdkInputBase(FixtureModel):
    boundary: str = "default"

    def as_sdk_kwargs(self) -> dict[str, object]:
        dumped: Final = cast(dict[str, object], self.model_dump(mode="python", exclude_unset=True))
        return {key: value for key, value in dumped.items() if key != "boundary"}

    def canonical_input(self) -> dict[str, object]:
        dumped: Final = cast(dict[str, object], self.model_dump(mode="json", exclude_unset=True))
        return {"boundary": self.boundary, **{key: value for key, value in dumped.items() if key != "boundary"}}


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
        migrated: Final = dict(value)
        provider_response: Final = migrated.pop("provider_response", None)
        if "provider_responses" not in migrated and provider_response is not None:
            migrated["provider_responses"] = (provider_response,)
        litellm_input: Final = migrated.get("litellm_input")
        if isinstance(litellm_input, Mapping) and "boundary" not in litellm_input:
            model: Final = litellm_input.get("model")
            if isinstance(model, str):
                migrated["litellm_input"] = {"boundary": _legacy_boundary(model), **litellm_input}
        return migrated


def _legacy_boundary(model: str) -> str:
    if model.startswith("azure_ai/doc-intelligence/"):
        return "azure_document_intelligence"
    if model.startswith("azure_ai/"):
        return "azure_mistral"
    if model.startswith("vertex_ai/deepseek"):
        return "vertex_deepseek"
    if model.startswith("vertex_ai/"):
        return "vertex_mistral"
    if model.endswith("parse-v3"):
        return "reducto_v3"
    if model.endswith("parse-legacy"):
        return "reducto_legacy"
    return "mistral"
