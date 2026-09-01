from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy
from pydantic import Field

from tests.route_parity.fixtures.recording import ProviderSpec
from tests.test_litellm.ocr.fixtures.base import OcrDocument, OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrRecordingTarget,
    image_data_document,
    invoke_with_api_key,
    sampled_scalar_strategy,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MistralCompatibleOcrSdkInput,
    mistral_input_values_strategy,
)

VertexMistralModel = Literal["vertex_ai/mistral-ocr-2505"]
VertexDeepSeekModel = Literal["vertex_ai/deepseek-ai/deepseek-ocr-maas"]

VERTEX_MISTRAL_MODELS: Final[tuple[VertexMistralModel, ...]] = ("vertex_ai/mistral-ocr-2505",)
VERTEX_DEEPSEEK_MODELS: Final[tuple[VertexDeepSeekModel, ...]] = ("vertex_ai/deepseek-ai/deepseek-ocr-maas",)


class VertexMistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    boundary: str = Field(default="vertex_mistral", pattern=r"^vertex_mistral$")
    model: VertexMistralModel = "vertex_ai/mistral-ocr-2505"
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


class VertexDeepSeekOcrSdkInput(OcrSdkInputBase):
    boundary: str = Field(default="vertex_deepseek", pattern=r"^vertex_deepseek$")
    model: VertexDeepSeekModel = "vertex_ai/deepseek-ai/deepseek-ocr-maas"
    document: OcrDocument
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


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
        model=sampled_scalar_strategy(VERTEX_MISTRAL_MODELS),
        values=mistral_input_values_strategy("2505", inline_image_data_uri),
    )


@st.composite
def vertex_deepseek_input_strategy(
    draw: DrawFn, project: str, location: str, inline_image_data_uri: str
) -> VertexDeepSeekOcrSdkInput:
    return VertexDeepSeekOcrSdkInput.model_validate(
        {
            "model": draw(sampled_scalar_strategy(VERTEX_DEEPSEEK_MODELS)),
            # The current Vertex model card documents image input only. Keep
            # the broader fixture model for existing recordings, but do not
            # spend a paid request on the transform's unsupported PDF branch.
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
    upstream_base: Final = environ.get("VERTEX_AI_API_BASE") or f"https://{location}-aiplatform.googleapis.com"
    invocation: Final = invoke_with_api_key(client, api_key)
    return (
        OcrRecordingTarget(
            name="vertex-mistral",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                vertex_mistral_input_strategy(project, location, inline_image_data_uri),
            ),
            invocation=invocation,
        ),
        OcrRecordingTarget(
            name="vertex-deepseek",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                vertex_deepseek_input_strategy(project, location, inline_image_data_uri),
            ),
            invocation=invocation,
        ),
    )
