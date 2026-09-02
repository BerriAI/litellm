from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, cast

from pydantic import Discriminator, Tag

from tests.route_parity.fixture_models import ParityCase
from tests.test_litellm.ocr.fixtures.azure import (
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
)
from tests.test_litellm.ocr.fixtures.base import OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.mistral import MistralOcrSdkInput, MistralProviderRejectedOcrSdkInput
from tests.test_litellm.ocr.fixtures.reducto import ReductoParseLegacySdkInput, ReductoParseV3SdkInput
from tests.test_litellm.ocr.fixtures.vertex import VertexDeepSeekOcrSdkInput, VertexMistralOcrSdkInput

__all__ = ("OcrParityCase", "OcrSdkInput")


def _ocr_boundary(value: object) -> str | None:
    if isinstance(value, Mapping):
        mapping: Final = cast(Mapping[object, object], value)
        boundary: Final = mapping.get("boundary")
        model: Final = mapping.get("model")
        if boundary == "mistral" and model == "mistral/invalid-ocr-model-for-parity":
            return "mistral_provider_rejected"
        return boundary if isinstance(boundary, str) else None
    if isinstance(value, OcrSdkInputBase):
        if isinstance(value, MistralProviderRejectedOcrSdkInput):
            return "mistral_provider_rejected"
        return value.boundary
    return None


OcrSdkInput = Annotated[
    Annotated[MistralOcrSdkInput, Tag("mistral")]
    | Annotated[MistralProviderRejectedOcrSdkInput, Tag("mistral_provider_rejected")]
    | Annotated[AzureMistralOcrSdkInput, Tag("azure_mistral")]
    | Annotated[VertexMistralOcrSdkInput, Tag("vertex_mistral")]
    | Annotated[AzureDocumentIntelligenceOcrSdkInput, Tag("azure_document_intelligence")]
    | Annotated[VertexDeepSeekOcrSdkInput, Tag("vertex_deepseek")]
    | Annotated[ReductoParseV3SdkInput, Tag("reducto_v3")]
    | Annotated[ReductoParseLegacySdkInput, Tag("reducto_legacy")],
    Discriminator(_ocr_boundary),
]


class OcrParityCase(ParityCase[OcrSdkInput]):
    pass
