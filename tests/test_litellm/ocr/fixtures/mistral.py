from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy
from pydantic import Field, model_validator
from typing_extensions import Self

from tests.route_parity.fixtures.recording import ProviderSpec
from tests.test_litellm.ocr.fixtures.base import (
    JsonSchemaResponseFormat,
    OcrDocument,
    OcrSdkInputBase,
)
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    annotation_format,
    image_document,
    invoke_with_api_key,
    public_document_strategy,
)

MistralModel = Literal[
    "mistral/mistral-ocr-2512",
    "mistral/mistral-ocr-4-0",
    "mistral/mistral-ocr-4-1",
    "mistral/mistral-ocr-4",
    "mistral/mistral-ocr-latest",
    "mistral-ocr-2512",
    "mistral-ocr-4-0",
    "mistral-ocr-4-1",
    "mistral-ocr-4",
    "mistral-ocr-latest",
]


class MistralCompatibleOcrSdkInput(OcrSdkInputBase):
    document: OcrDocument
    pages: str | list[int] | None = None
    include_image_base64: bool | None = None
    image_limit: int | None = None
    image_min_size: int | None = None
    bbox_annotation_format: JsonSchemaResponseFormat | None = None
    document_annotation_format: JsonSchemaResponseFormat | None = None
    document_annotation_prompt: str | None = None
    extract_header: bool = False
    extract_footer: bool = False
    table_format: Literal["markdown", "html"] | None = None
    confidence_scores_granularity: Literal["page", "word", "block"] | None = None
    include_blocks: bool = True
    id: str | None = None

    @model_validator(mode="after")
    def validate_annotation_prompt(self) -> Self:
        if self.document_annotation_prompt is not None and self.document_annotation_format is None:
            raise ValueError("document_annotation_prompt requires document_annotation_format")
        return self


class MistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    boundary: str = Field(default="mistral", pattern=r"^mistral$")
    model: MistralModel
    custom_llm_provider: Literal["mistral"] | None = None

    @model_validator(mode="after")
    def validate_provider_routing(self) -> Self:
        if not self.model.startswith("mistral/") and self.custom_llm_provider != "mistral":
            raise ValueError("unqualified Mistral models require custom_llm_provider='mistral'")
        return self


MISTRAL_MODEL: Final = "mistral/mistral-ocr-latest"
_VALUE_TEXT: Final = st.just("case-1")


@st.composite
def mistral_input_strategy(draw: DrawFn, model: str) -> MistralOcrSdkInput:
    document: Final = draw(public_document_strategy())
    annotation: Final = draw(
        st.sampled_from(
            (
                {},
                {"document_annotation_format": annotation_format("document_title")},
                {
                    "document_annotation_format": annotation_format("prompted_document_title"),
                    "document_annotation_prompt": "Extract the visible title",
                },
            )
        )
    )
    optional_params: Final = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "pages": st.just([0]),
                "include_image_base64": st.booleans(),
                "image_limit": st.just(1),
                "image_min_size": st.just(300),
                "bbox_annotation_format": st.just(annotation_format("bounding_boxes")),
                "extract_header": st.booleans(),
                "extract_footer": st.booleans(),
                "table_format": st.just("markdown"),
                "confidence_scores_granularity": st.just("page"),
                "include_blocks": st.booleans(),
                "id": _VALUE_TEXT,
            },
        )
    )
    return MistralOcrSdkInput.model_validate({"model": model, "document": document, **annotation, **optional_params})


def required_mistral_inputs(model: str) -> tuple[MistralOcrSdkInput, ...]:
    document: Final = image_document("invoice 123", 24)
    annotation: Final = annotation_format("document_title")
    bbox_annotation: Final = annotation_format("bounding_boxes")
    cases: Final[tuple[dict[str, object], ...]] = (
        {},
        {"pages": [0]},
        {"include_image_base64": True},
        {"image_limit": 1},
        {"image_min_size": 300},
        {"bbox_annotation_format": bbox_annotation},
        {"document_annotation_format": annotation},
        {"document_annotation_format": annotation, "document_annotation_prompt": "Extract the visible title"},
        {"extract_header": True},
        {"extract_footer": True},
        {"table_format": "markdown"},
        {"confidence_scores_granularity": "page"},
        {"include_blocks": False},
        {"id": "case-1"},
    )
    return tuple(MistralOcrSdkInput.model_validate({"model": model, "document": document, **case}) for case in cases)


def mistral_recording_targets(environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("MISTRAL_API_KEY")
    if not api_key:
        return ()
    configured: Final = environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    upstream_base: Final = configured.removesuffix("/v1")
    return (
        OcrRecordingTarget(
            name="mistral-ocr",
            provider_spec=ProviderSpec(upstream_base=upstream_base),
            strategy=cast(SearchStrategy[OcrSdkInputBase], mistral_input_strategy(MISTRAL_MODEL)),
            invocation=invoke_with_api_key(client, api_key),
            required_inputs=cast(tuple[OcrSdkInputBase, ...], required_mistral_inputs(MISTRAL_MODEL)),
        ),
    )
