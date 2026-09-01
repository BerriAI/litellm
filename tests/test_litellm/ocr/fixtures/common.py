from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Final, Literal, Protocol, TypeVar

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from tests.route_parity.fixtures.media import dummy_image_url, structured_pdf_data_uri
from tests.route_parity.fixtures.pipeline import RecordingTarget
from tests.test_litellm.ocr.fixtures.base import (
    DocumentUrlDocument,
    ImageUrlDocument,
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    OcrSdkInputBase,
)

OcrRecordingTarget = RecordingTarget[OcrSdkInputBase]
ValueT = TypeVar("ValueT")


class OcrFixtureClient(Protocol):
    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None: ...


class OcrSdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class ApiKeyOcrInvocation:
    client: OcrFixtureClient
    api_key: str = field(repr=False)

    def execute(self, provider_url: str, case_input: OcrSdkInputBase) -> None:
        self.client.execute(provider_url, self.api_key, case_input)


def image_document(text: str, font_size: int) -> ImageUrlDocument:
    return ImageUrlDocument(type="image_url", image_url=dummy_image_url(text, font_size))


def image_data_document(data_uri: str) -> ImageUrlDocument:
    return ImageUrlDocument(type="image_url", image_url=data_uri)


@cache
def remote_pdf_document() -> DocumentUrlDocument:
    return DocumentUrlDocument(
        type="document_url",
        document_url="https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    )


@cache
def pdf_document() -> DocumentUrlDocument:
    return DocumentUrlDocument(type="document_url", document_url=structured_pdf_data_uri())


def document_transport_strategy(inline_image_data_uri: str) -> SearchStrategy[ImageUrlDocument | DocumentUrlDocument]:
    transports: Final[tuple[Literal["remote_image", "inline_image", "remote_pdf", "inline_pdf"], ...]] = (
        "remote_image",
        "inline_image",
        "remote_pdf",
        "inline_pdf",
    )

    def as_document(
        transport: Literal["remote_image", "inline_image", "remote_pdf", "inline_pdf"],
    ) -> ImageUrlDocument | DocumentUrlDocument:
        if transport == "remote_image":
            return image_document("invoice 123", 24)
        if transport == "inline_image":
            return image_data_document(inline_image_data_uri)
        if transport == "remote_pdf":
            return remote_pdf_document()
        return pdf_document()

    return st.sampled_from(transports).map(as_document)


def sampled_scalar_strategy(values: tuple[ValueT, ...]) -> SearchStrategy[ValueT]:
    return st.sampled_from(values)


def sampled_list_strategy(values: tuple[tuple[ValueT, ...], ...]) -> SearchStrategy[list[ValueT]]:
    return st.sampled_from(values).map(list)


def sampled_parameter_group_strategy(
    values: tuple[tuple[tuple[str, object], ...], ...],
) -> SearchStrategy[dict[str, object]]:
    return st.sampled_from(values).map(dict)


def parameter_strategy(name: str, values: SearchStrategy[ValueT]) -> SearchStrategy[dict[str, object]]:
    def as_parameter(value: ValueT) -> dict[str, object]:
        return {name: value}

    return values.map(as_parameter)


def annotation_format(name: str) -> JsonSchemaResponseFormat:
    return JsonSchemaResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(
            name=name,
            description="Extract the visible document fields",
            schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )


def invoke_with_api_key(client: OcrFixtureClient, api_key: str) -> ApiKeyOcrInvocation:
    return ApiKeyOcrInvocation(client=client, api_key=api_key)
