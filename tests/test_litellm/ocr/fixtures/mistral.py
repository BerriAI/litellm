from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.route_parity.fixture_generator import FixtureSdkCall
from tests.route_parity.fixture_recorder import ProviderSpec
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureTarget,
    annotation_format,
    image_document,
    invoke_with_api_key,
    public_document_strategy,
)
from tests.test_litellm.ocr.fixtures.models import MistralOcrSdkInput, OcrSdkInputBase

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


class MistralFixtureProvider:
    def targets(self, environ: Mapping[str, str], sdk_call: FixtureSdkCall) -> tuple[OcrFixtureTarget, ...]:
        api_key: Final = environ.get("MISTRAL_API_KEY")
        if not api_key:
            return ()
        configured: Final = environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
        upstream_base: Final = configured.removesuffix("/v1")
        return (
            OcrFixtureTarget(
                name="mistral-ocr",
                provider_spec=ProviderSpec(upstream_base=upstream_base),
                strategy=cast(SearchStrategy[OcrSdkInputBase], mistral_input_strategy(MISTRAL_MODEL)),
                invoke=invoke_with_api_key(sdk_call, api_key),
                required_inputs=cast(tuple[OcrSdkInputBase, ...], required_mistral_inputs(MISTRAL_MODEL)),
            ),
        )
