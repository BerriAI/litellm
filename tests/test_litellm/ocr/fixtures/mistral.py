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
    parameter_strategy,
    public_document_strategy,
    sampled_list_strategy,
    sampled_parameter_group_strategy,
    sampled_scalar_strategy,
)

MistralModel = Literal[
    "mistral/mistral-ocr-3",
    "mistral/mistral-ocr-3-0",
    "mistral/mistral-ocr-2512",
    "mistral/mistral-ocr-4-0",
    "mistral/mistral-ocr-4-1",
    "mistral/mistral-ocr-4",
    "mistral/mistral-ocr-latest",
    "mistral-ocr-3",
    "mistral-ocr-3-0",
    "mistral-ocr-2512",
    "mistral-ocr-4-0",
    "mistral-ocr-4-1",
    "mistral-ocr-4",
    "mistral-ocr-latest",
]

MISTRAL_MODELS: Final[tuple[MistralModel, ...]] = (
    "mistral/mistral-ocr-3",
    "mistral/mistral-ocr-3-0",
    "mistral/mistral-ocr-2512",
    "mistral/mistral-ocr-4",
    "mistral/mistral-ocr-4-0",
    "mistral/mistral-ocr-4-1",
    "mistral/mistral-ocr-latest",
)


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


MISTRAL_MODEL: Final[MistralModel] = "mistral/mistral-ocr-latest"
MistralFeatureLevel = Literal["2505", "2512", "4"]
_MISTRAL_4_MODELS: Final = frozenset(
    {
        "mistral/mistral-ocr-4",
        "mistral/mistral-ocr-4-0",
        "mistral/mistral-ocr-4-1",
        "mistral/mistral-ocr-latest",
    }
)
_MISTRAL_2512_MODELS: Final = frozenset(
    {*_MISTRAL_4_MODELS, "mistral/mistral-ocr-2512", "mistral/mistral-ocr-3", "mistral/mistral-ocr-3-0"}
)


def _feature_level(model: str) -> MistralFeatureLevel:
    if model in _MISTRAL_4_MODELS:
        return "4"
    if model in _MISTRAL_2512_MODELS:
        return "2512"
    return "2505"


def mistral_optional_params_strategy(feature_level: MistralFeatureLevel) -> SearchStrategy[dict[str, object]]:
    annotation: Final = annotation_format("document_title")
    common: Final[tuple[SearchStrategy[dict[str, object]], ...]] = (
        parameter_strategy("pages", sampled_list_strategy(((0,), (0, 1)))),
        parameter_strategy("include_image_base64", sampled_scalar_strategy((False, True))),
        parameter_strategy("image_limit", sampled_scalar_strategy((1,))),
        parameter_strategy("image_min_size", sampled_scalar_strategy((300,))),
        parameter_strategy(
            "bbox_annotation_format",
            sampled_scalar_strategy((annotation_format("bounding_boxes"),)),
        ),
        parameter_strategy("document_annotation_format", sampled_scalar_strategy((annotation,))),
        sampled_parameter_group_strategy(
            (
                (
                    ("document_annotation_format", annotation),
                    ("document_annotation_prompt", "Extract the visible title"),
                ),
            )
        ),
        parameter_strategy("confidence_scores_granularity", sampled_scalar_strategy(("page", "word"))),
        parameter_strategy("id", sampled_scalar_strategy(("case-1",))),
    )
    feature_2512: Final[tuple[SearchStrategy[dict[str, object]], ...]] = (
        parameter_strategy("extract_header", sampled_scalar_strategy((False, True))),
        parameter_strategy("extract_footer", sampled_scalar_strategy((False, True))),
        parameter_strategy("table_format", sampled_scalar_strategy(("markdown", "html"))),
    )
    feature_4: Final[tuple[SearchStrategy[dict[str, object]], ...]] = (
        parameter_strategy("include_blocks", sampled_scalar_strategy((False, True))),
        sampled_parameter_group_strategy(((("include_blocks", True), ("confidence_scores_granularity", "block")),)),
    )
    return st.one_of(
        *common,
        *(feature_2512 if feature_level in {"2512", "4"} else ()),
        *(feature_4 if feature_level == "4" else ()),
    )


@st.composite
def mistral_input_strategy(
    draw: DrawFn,
    model: str,
    feature_level: MistralFeatureLevel | None = None,
) -> MistralOcrSdkInput:
    canonical_document: Final = image_document("invoice 123", 24)
    values: Final = draw(
        st.one_of(
            public_document_strategy().map(lambda document: {"document": document}),
            mistral_optional_params_strategy(feature_level or _feature_level(model)).map(
                lambda optional_params: {"document": canonical_document, **optional_params}
            ),
        )
    )
    return MistralOcrSdkInput.model_validate({"model": model, **values})


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
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                sampled_scalar_strategy(MISTRAL_MODELS).flatmap(mistral_input_strategy),
            ),
            invocation=invoke_with_api_key(client, api_key),
        ),
    )
