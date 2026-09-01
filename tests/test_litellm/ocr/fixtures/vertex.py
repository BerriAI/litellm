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
    image_document,
    invoke_with_api_key,
    public_document_strategy,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MISTRAL_MODEL,
    MistralCompatibleOcrSdkInput,
    MistralOcrSdkInput,
    mistral_input_strategy,
    required_mistral_inputs,
)


class VertexMistralOcrSdkInput(MistralCompatibleOcrSdkInput):
    boundary: str = Field(default="vertex_mistral", pattern=r"^vertex_mistral$")
    model: Literal["vertex_ai/mistral-ocr-2505"] = "vertex_ai/mistral-ocr-2505"
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


class VertexDeepSeekOcrSdkInput(OcrSdkInputBase):
    boundary: str = Field(default="vertex_deepseek", pattern=r"^vertex_deepseek$")
    model: Literal["vertex_ai/deepseek-ocr-maas"] = "vertex_ai/deepseek-ocr-maas"
    document: OcrDocument
    custom_llm_provider: Literal["vertex_ai"] | None = None
    vertex_project: str
    vertex_location: str = "us-central1"


def _as_vertex_mistral(case_input: MistralOcrSdkInput, project: str, location: str) -> VertexMistralOcrSdkInput:
    values: Final = case_input.model_dump(mode="python", exclude={"boundary", "model", "custom_llm_provider"})
    return VertexMistralOcrSdkInput.model_validate({**values, "vertex_project": project, "vertex_location": location})


def _required_deepseek_inputs(project: str, location: str) -> tuple[VertexDeepSeekOcrSdkInput, ...]:
    return (
        VertexDeepSeekOcrSdkInput(
            document=image_document("invoice 123", 24),
            vertex_project=project,
            vertex_location=location,
        ),
    )


@st.composite
def vertex_deepseek_input_strategy(draw: DrawFn, project: str, location: str) -> VertexDeepSeekOcrSdkInput:
    return VertexDeepSeekOcrSdkInput.model_validate(
        {
            "document": draw(public_document_strategy()),
            "vertex_project": project,
            "vertex_location": location,
        }
    )


def vertex_recording_targets(environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrRecordingTarget, ...]:
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
                st.builds(
                    _as_vertex_mistral,
                    case_input=mistral_input_strategy(MISTRAL_MODEL),
                    project=st.just(project),
                    location=st.just(location),
                ),
            ),
            invocation=invocation,
            required_inputs=cast(
                tuple[OcrSdkInputBase, ...],
                tuple(
                    _as_vertex_mistral(case_input, project, location)
                    for case_input in required_mistral_inputs(MISTRAL_MODEL)
                ),
            ),
        ),
        OcrRecordingTarget(
            name="vertex-deepseek",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(SearchStrategy[OcrSdkInputBase], vertex_deepseek_input_strategy(project, location)),
            invocation=invocation,
            required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_deepseek_inputs(project, location)),
        ),
    )
