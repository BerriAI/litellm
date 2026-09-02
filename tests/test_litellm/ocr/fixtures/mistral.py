from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
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
    document_transport_strategy,
    invoke_with_api_key,
    parameter_strategy,
    pdf_document,
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


class MistralProviderRejectedOcrSdkInput(MistralCompatibleOcrSdkInput):
    boundary: str = Field(default="mistral", pattern=r"^mistral$")
    model: Literal["mistral/invalid-ocr-model-for-parity"]
    custom_llm_provider: Literal["mistral"] | None = None


MISTRAL_MODEL: Final[MistralModel] = "mistral/mistral-ocr-latest"
MISTRAL_PROVIDER_REJECTED_INPUTS: Final[tuple[MistralProviderRejectedOcrSdkInput, ...]] = (
    MistralProviderRejectedOcrSdkInput(
        model="mistral/invalid-ocr-model-for-parity",
        document=pdf_document(),
    ),
)
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


def _optional_param_strategies(
    *,
    include_document_annotation_prompt: bool = True,
) -> tuple[
    tuple[SearchStrategy[dict[str, object]], ...],
    tuple[SearchStrategy[dict[str, object]], ...],
    tuple[SearchStrategy[dict[str, object]], ...],
]:
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
        *(
            (
                sampled_parameter_group_strategy(
                    (
                        (
                            ("document_annotation_format", annotation),
                            ("document_annotation_prompt", "Extract the visible title"),
                        ),
                    )
                ),
            )
            if include_document_annotation_prompt
            else ()
        ),
        parameter_strategy("confidence_scores_granularity", sampled_scalar_strategy(("page", "word"))),
    )
    feature_2512: Final[tuple[SearchStrategy[dict[str, object]], ...]] = (
        parameter_strategy("extract_header", sampled_scalar_strategy((False, True))),
        parameter_strategy("extract_footer", sampled_scalar_strategy((False, True))),
        parameter_strategy("table_format", sampled_scalar_strategy(("markdown", "html"))),
    )
    feature_4: Final[tuple[SearchStrategy[dict[str, object]], ...]] = (
        parameter_strategy("pages", sampled_scalar_strategy(("0-2",))),
        parameter_strategy("include_blocks", sampled_scalar_strategy((False, True))),
        sampled_parameter_group_strategy(((("include_blocks", True), ("confidence_scores_granularity", "block")),)),
    )
    return common, feature_2512, feature_4


def mistral_optional_params_strategy(
    feature_level: MistralFeatureLevel,
    *,
    include_document_annotation_prompt: bool = True,
) -> SearchStrategy[dict[str, object]]:
    common, feature_2512, feature_4 = _optional_param_strategies(
        include_document_annotation_prompt=include_document_annotation_prompt
    )
    return st.one_of(
        *common,
        *(feature_2512 if feature_level in {"2512", "4"} else ()),
        *(feature_4 if feature_level == "4" else ()),
    )


def _mistral_input_values(
    document: OcrDocument,
    optional_params: dict[str, object] | None = None,
) -> dict[str, object]:
    return {"document": document, **(optional_params or {})}


def _mistral_input(
    model: str,
    document: OcrDocument,
    optional_params: dict[str, object] | None = None,
) -> MistralOcrSdkInput:
    return MistralOcrSdkInput.model_validate({"model": model, **_mistral_input_values(document, optional_params)})


def mistral_input_values_strategy(
    feature_level: MistralFeatureLevel,
    inline_image_data_uri: str,
    *,
    include_document_annotation_prompt: bool = True,
) -> SearchStrategy[dict[str, object]]:
    option_document: Final = pdf_document()
    return st.one_of(
        document_transport_strategy(inline_image_data_uri).map(_mistral_input_values),
        mistral_optional_params_strategy(
            feature_level,
            include_document_annotation_prompt=include_document_annotation_prompt,
        ).map(lambda optional_params: _mistral_input_values(option_document, optional_params)),
    )


def mistral_input_strategy(
    model: str,
    inline_image_data_uri: str,
    feature_level: MistralFeatureLevel | None = None,
) -> SearchStrategy[MistralOcrSdkInput]:
    return mistral_input_values_strategy(feature_level or _feature_level(model), inline_image_data_uri).map(
        lambda values: MistralOcrSdkInput.model_validate({"model": model, **values})
    )


def _mistral_recording_strategy(inline_image_data_uri: str) -> SearchStrategy[MistralOcrSdkInput]:
    document: Final = pdf_document()
    baseline_models: Final = tuple(model for model in MISTRAL_MODELS if model != MISTRAL_MODEL)
    common, feature_2512, feature_4 = _optional_param_strategies()
    common_options: Final[SearchStrategy[dict[str, object]]] = st.one_of(*common)
    feature_2512_options: Final[SearchStrategy[dict[str, object]]] = st.one_of(*feature_2512)
    feature_4_options: Final[SearchStrategy[dict[str, object]]] = st.one_of(*feature_4)
    return st.one_of(
        sampled_scalar_strategy(baseline_models).map(lambda model: _mistral_input(model, document)),
        document_transport_strategy(inline_image_data_uri).map(
            lambda selected_document: _mistral_input(MISTRAL_MODEL, selected_document)
        ),
        common_options.map(lambda optional_params: _mistral_input(MISTRAL_MODEL, document, optional_params)),
        feature_2512_options.map(
            lambda optional_params: _mistral_input("mistral/mistral-ocr-2512", document, optional_params)
        ),
        feature_4_options.map(
            lambda optional_params: _mistral_input("mistral/mistral-ocr-4-1", document, optional_params)
        ),
    )


def mistral_recording_targets(
    environ: Mapping[str, str], client: OcrFixtureClient, inline_image_data_uri: str
) -> tuple[OcrRecordingTarget, ...]:
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
                _mistral_recording_strategy(inline_image_data_uri),
            ),
            invocation=invoke_with_api_key(client, api_key),
            required_inputs=MISTRAL_PROVIDER_REJECTED_INPUTS,
        ),
    )
