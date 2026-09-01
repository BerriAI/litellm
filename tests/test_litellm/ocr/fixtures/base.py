from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tests.route_parity.fixture_models import (
    FixtureModel,
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    SdkInputBase,
)

__all__ = (
    "DocumentUrlDocument",
    "ImageUrlDocument",
    "ImageUrlValue",
    "JsonSchemaDefinition",
    "JsonSchemaResponseFormat",
    "OcrDocument",
    "OcrSdkInputBase",
)

OcrSdkInputBase = SdkInputBase


class ImageUrlValue(FixtureModel):
    url: str
    detail: Literal["low", "auto", "high"] | None = None


class ImageUrlDocument(FixtureModel):
    type: Literal["image_url"]
    image_url: str | ImageUrlValue


class DocumentUrlDocument(FixtureModel):
    type: Literal["document_url"]
    document_url: str
    document_name: str | None = None


OcrDocument = Annotated[
    ImageUrlDocument | DocumentUrlDocument,
    Field(discriminator="type"),
]
