from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    JsonValue: TypeAlias = list["JsonValue"] | dict[str, "JsonValue"] | str | bool | int | float | None
else:
    from pydantic import JsonValue

JsonObject = dict[str, JsonValue]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DocumentUrl(_ContractModel):
    type: Literal["document_url"]
    document_url: str


class ImageUrl(_ContractModel):
    type: Literal["image_url"]
    image_url: str


OCRDocument = Annotated[DocumentUrl | ImageUrl, Field(discriminator="type")]


class GetSupportedOCRParamsInput(_ContractModel):
    model: str


class MapOCRParamsInput(_ContractModel):
    model: str
    non_default_params: JsonObject
    optional_params: JsonObject


class TransformOCRRequestInput(_ContractModel):
    model: str
    document: OCRDocument
    optional_params: JsonObject
    headers: dict[str, str]


class HTTPResponseInput(_ContractModel):
    status_code: int = Field(ge=100, le=599)
    body: JsonObject
    headers: dict[str, str] = Field(default_factory=dict)


class TransformOCRResponseInput(_ContractModel):
    model: str
    response: HTTPResponseInput


class OCRRequestOutput(_ContractModel):
    data: JsonObject
    files: JsonObject | None


class _Case(_ContractModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")


class GetSupportedOCRParamsCase(_Case):
    operation: Literal["mistral.ocr.get_supported_ocr_params"]
    input: GetSupportedOCRParamsInput
    expected: tuple[str, ...]


class MapOCRParamsCase(_Case):
    operation: Literal["mistral.ocr.map_ocr_params"]
    input: MapOCRParamsInput
    expected: JsonObject


class TransformOCRRequestCase(_Case):
    operation: Literal["mistral.ocr.transform_ocr_request"]
    input: TransformOCRRequestInput
    expected: OCRRequestOutput


class TransformOCRResponseCase(_Case):
    operation: Literal["mistral.ocr.transform_ocr_response"]
    input: TransformOCRResponseInput
    expected: JsonObject


TransformationCase = Annotated[
    GetSupportedOCRParamsCase | MapOCRParamsCase | TransformOCRRequestCase | TransformOCRResponseCase,
    Field(discriminator="operation"),
]

_JSON_OBJECT_ADAPTER: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def expected_output(case: TransformationCase) -> JsonValue:
    if isinstance(case, GetSupportedOCRParamsCase):
        return list(case.expected)
    if isinstance(case, TransformOCRRequestCase):
        return _JSON_OBJECT_ADAPTER.validate_python(cast(object, case.expected.model_dump(mode="json")))
    return case.expected


class ContractSuiteV1(_ContractModel):
    schema_version: Literal[1]
    cases: tuple[TransformationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_id_namespaces(self) -> ContractSuiteV1:
        invalid: tuple[TransformationCase, ...] = tuple(
            case for case in self.cases if not case.id.startswith(f"{case.operation}.")
        )
        if not invalid:
            return self
        case = invalid[0]
        raise ValueError(f"case id must start with '{case.operation}.'")


CONTRACT_SUITE_ADAPTER: TypeAdapter[ContractSuiteV1] = TypeAdapter(ContractSuiteV1)
