from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import quote

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from tests.route_parity.fixtures.pipeline import RecordingTarget
from tests.test_litellm.ocr.fixtures.models import (
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    MistralDocumentUrlDocument,
    MistralImageUrlDocument,
    OcrSdkInputBase,
)

OcrRecordingTarget = RecordingTarget[OcrSdkInputBase]


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


def image_document(text: str, font_size: int) -> MistralImageUrlDocument:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    return MistralImageUrlDocument(type="image_url", image_url=url)


def fixture_pdf_data_uri() -> str:
    fixture: Final = Path(__file__).resolve().parents[3] / "llm_translation" / "fixtures" / "dummy.pdf"
    encoded: Final = base64.b64encode(fixture.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def pdf_document() -> MistralDocumentUrlDocument:
    return MistralDocumentUrlDocument(type="document_url", document_url=fixture_pdf_data_uri())


def public_document_strategy() -> SearchStrategy[MistralImageUrlDocument | MistralDocumentUrlDocument]:
    return st.sampled_from((image_document("invoice 123", 24), pdf_document()))


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
