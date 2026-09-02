from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.route_parity.fixtures.recording import UpstreamEndpoint
from tests.test_litellm.ocr.fixtures.base import OcrDocument, OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    image_data_document,
    invoke_with_api_key,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MistralCompatibleOcrSdkInput,
    mistral_input_values_strategy,
)

VertexMistralModel = Literal["vertex_ai/mistral-ocr-2505"]
VertexDeepSeekModel = Literal["vertex_ai/deepseek-ai/deepseek-ocr-maas"]
VertexMistralFixtureModel = VertexMistralModel | Literal["vertex_ai/invalid-ocr-model-for-parity"]
VertexDeepSeekFixtureModel = VertexDeepSeekModel | Literal["vertex_ai/deepseek-ai/invalid-ocr-model-for-parity"]

VERTEX_MISTRAL_MODELS: Final[tuple[VertexMistralModel, ...]] = ("vertex_ai/mistral-ocr-2505",)
VERTEX_DEEPSEEK_MODELS: Final[tuple[VertexDeepSeekModel, ...]] = ("vertex_ai/deepseek-ai/deepseek-ocr-maas",)


class VertexMistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    contract: Literal["vertex_mistral"] = "vertex_mistral"
    model: VertexMistralFixtureModel = "vertex_ai/mistral-ocr-2505"
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


class VertexDeepSeekOcrSdkInput(OcrSdkInputBase):
    contract: Literal["vertex_deepseek"] = "vertex_deepseek"
    model: VertexDeepSeekFixtureModel = "vertex_ai/deepseek-ai/deepseek-ocr-maas"
    document: OcrDocument
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


def vertex_mistral_provider_rejected_inputs(
    project: str,
    location: str,
    inline_image_data_uri: str,
) -> tuple[VertexMistralOcrSdkInput, ...]:
    return (
        VertexMistralOcrSdkInput(
            model="vertex_ai/invalid-ocr-model-for-parity",
            document=image_data_document(inline_image_data_uri),
            vertex_project=project,
            vertex_location=location,
        ),
    )


def vertex_deepseek_provider_rejected_inputs(
    project: str,
    location: str,
    inline_image_data_uri: str,
) -> tuple[VertexDeepSeekOcrSdkInput, ...]:
    return (
        VertexDeepSeekOcrSdkInput(
            model="vertex_ai/deepseek-ai/invalid-ocr-model-for-parity",
            document=image_data_document(inline_image_data_uri),
            vertex_project=project,
            vertex_location=location,
        ),
    )


def _as_vertex_mistral(
    values: dict[str, object],
    project: str,
    location: str,
    model: VertexMistralModel,
) -> VertexMistralOcrSdkInput:
    return VertexMistralOcrSdkInput.model_validate(
        {**values, "model": model, "vertex_project": project, "vertex_location": location}
    )


def vertex_mistral_input_strategy(
    project: str,
    location: str,
    inline_image_data_uri: str,
) -> SearchStrategy[VertexMistralOcrSdkInput]:
    return st.builds(
        _as_vertex_mistral,
        project=st.just(project),
        location=st.just(location),
        model=st.sampled_from(VERTEX_MISTRAL_MODELS),
        values=mistral_input_values_strategy("2505", inline_image_data_uri),
    )


@st.composite
def vertex_deepseek_input_strategy(
    draw: DrawFn, project: str, location: str, inline_image_data_uri: str
) -> VertexDeepSeekOcrSdkInput:
    return VertexDeepSeekOcrSdkInput.model_validate(
        {
            "model": draw(st.sampled_from(VERTEX_DEEPSEEK_MODELS)),
            "document": image_data_document(inline_image_data_uri),
            "vertex_project": project,
            "vertex_location": location,
        }
    )


def vertex_recording_targets(
    environ: Mapping[str, str], client: OcrFixtureClient, inline_image_data_uri: str
) -> tuple[OcrRecordingTarget, ...]:
    api_key: Final = environ.get("VERTEX_AI_API_KEY")
    project: Final = environ.get("VERTEXAI_PROJECT") or environ.get("VERTEX_PROJECT")
    location: Final = environ.get("VERTEXAI_LOCATION") or environ.get("VERTEX_LOCATION") or "us-central1"
    if not api_key or not project:
        return ()
    base_url: Final = environ.get("VERTEX_AI_API_BASE") or f"https://{location}-aiplatform.googleapis.com"
    invocation: Final = invoke_with_api_key(client, api_key)
    return (
        OcrRecordingTarget(
            name="vertex-mistral",
            upstream=UpstreamEndpoint(base_url=base_url.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                vertex_mistral_input_strategy(project, location, inline_image_data_uri),
            ),
            invocation=invocation,
            required_inputs=vertex_mistral_provider_rejected_inputs(project, location, inline_image_data_uri),
        ),
        OcrRecordingTarget(
            name="vertex-deepseek",
            upstream=UpstreamEndpoint(base_url=base_url.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                vertex_deepseek_input_strategy(project, location, inline_image_data_uri),
            ),
            invocation=invocation,
            required_inputs=vertex_deepseek_provider_rejected_inputs(project, location, inline_image_data_uri),
        ),
    )
