from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, cast

from pydantic import Field, model_validator

from tests.route_parity.fixture_models import ParityCase
from tests.test_litellm.ocr.fixtures.azure import (
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
)
from tests.test_litellm.ocr.fixtures.mistral import MistralOcrSdkInput
from tests.test_litellm.ocr.fixtures.reducto import ReductoParseLegacySdkInput, ReductoParseV3SdkInput
from tests.test_litellm.ocr.fixtures.vertex import VertexDeepSeekOcrSdkInput, VertexMistralOcrSdkInput

__all__ = ("OcrParityCase", "OcrSdkInput")


OcrSdkInput = Annotated[
    MistralOcrSdkInput
    | AzureMistralOcrSdkInput
    | VertexMistralOcrSdkInput
    | AzureDocumentIntelligenceOcrSdkInput
    | VertexDeepSeekOcrSdkInput
    | ReductoParseV3SdkInput
    | ReductoParseLegacySdkInput,
    Field(discriminator="contract"),
]


class OcrParityCase(ParityCase[OcrSdkInput]):
    @model_validator(mode="before")
    @classmethod
    def load_legacy_contract(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        fixture: Final = cast(Mapping[str, object], value)
        litellm_input: Final = fixture.get("litellm_input")
        if not isinstance(litellm_input, Mapping) or "contract" in litellm_input:
            return fixture
        legacy_input: Final = cast(Mapping[str, object], litellm_input)
        legacy_contract: Final = legacy_input.get("boundary")
        if isinstance(legacy_contract, str):
            return {
                **fixture,
                "litellm_input": {
                    "contract": legacy_contract,
                    **{key: item for key, item in legacy_input.items() if key != "boundary"},
                },
            }
        model: Final = legacy_input.get("model")
        if not isinstance(model, str):
            return fixture
        return {**fixture, "litellm_input": {"contract": _legacy_contract(model), **legacy_input}}


def _legacy_contract(model: str) -> str:
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
