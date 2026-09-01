from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.route_parity.fixture_recorder import ProviderSpec
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrFixtureTarget,
    image_document,
    invoke_with_api_key,
    public_document_strategy,
)
from tests.test_litellm.ocr.fixtures.mistral import (
    MISTRAL_MODEL,
    mistral_input_strategy,
    required_mistral_inputs,
)
from tests.test_litellm.ocr.fixtures.models import (
    MistralOcrSdkInput,
    OcrSdkInputBase,
    VertexDeepSeekOcrSdkInput,
    VertexMistralOcrSdkInput,
)


def _as_vertex_mistral(case_input: MistralOcrSdkInput, project: str, location: str) -> VertexMistralOcrSdkInput:
    values: Final = case_input.model_dump(mode="python", exclude={"boundary", "model", "custom_llm_provider"})
    return VertexMistralOcrSdkInput.model_validate({**values, "vertex_project": project, "vertex_location": location})


def _required_deepseek_inputs(project: str, location: str) -> tuple[VertexDeepSeekOcrSdkInput, ...]:
    document: Final = image_document("invoice 123", 24)
    common: Final = {"document": document, "vertex_project": project, "vertex_location": location}
    cases: Final[tuple[dict[str, object], ...]] = (
        {},
        {"stream": False},
        {"temperature": 0.5},
        {"max_tokens": 256},
        {"top_p": 0.9},
        {"n": 1},
        {"stop": ["END", "STOP"]},
    )
    return tuple(VertexDeepSeekOcrSdkInput.model_validate({**common, **case}) for case in cases)


@st.composite
def vertex_deepseek_input_strategy(draw: DrawFn, project: str, location: str) -> VertexDeepSeekOcrSdkInput:
    optional_params: Final = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "stream": st.just(False),
                "temperature": st.sampled_from((0.0, 0.5, 1.0)),
                "max_tokens": st.sampled_from((1, 256, 1024)),
                "top_p": st.sampled_from((0.1, 0.9, 1.0)),
                "n": st.just(1),
                "stop": st.sampled_from(("END", ["END", "STOP"])),
            },
        )
    )
    return VertexDeepSeekOcrSdkInput.model_validate(
        {
            "document": draw(public_document_strategy()),
            "vertex_project": project,
            "vertex_location": location,
            **optional_params,
        }
    )


class VertexFixtureSource:
    def targets(self, environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrFixtureTarget, ...]:
        api_key: Final = environ.get("VERTEX_AI_API_KEY")
        project: Final = environ.get("VERTEXAI_PROJECT") or environ.get("VERTEX_PROJECT")
        location: Final = environ.get("VERTEXAI_LOCATION") or environ.get("VERTEX_LOCATION") or "us-central1"
        if not api_key or not project:
            return ()
        upstream_base: Final = environ.get("VERTEX_AI_API_BASE") or f"https://{location}-aiplatform.googleapis.com"
        invocation: Final = invoke_with_api_key(client, api_key)
        return (
            OcrFixtureTarget(
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
            OcrFixtureTarget(
                name="vertex-deepseek",
                provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
                strategy=cast(SearchStrategy[OcrSdkInputBase], vertex_deepseek_input_strategy(project, location)),
                invocation=invocation,
                required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_deepseek_inputs(project, location)),
            ),
        )
