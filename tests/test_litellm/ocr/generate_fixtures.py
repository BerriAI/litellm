from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote

from dotenv import load_dotenv
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.route_parity.fixture_generator import (
    FixtureTarget,
    generate_target_fixtures,
    parse_generator_args,
)
from tests.route_parity.fixture_generator import (
    require_targets as require_fixture_targets,
)
from tests.route_parity.fixture_recorder import (
    ProviderSpec,
    fixture_directory,
)
from tests.test_litellm.ocr.fixture_models import (
    AzureDocumentIntelligenceOcrSdkInput,
    AzureMistralOcrSdkInput,
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    MistralDocumentUrlDocument,
    MistralImageUrlDocument,
    MistralOcrSdkInput,
    OcrParityCase,
    OcrSdkInputBase,
    ReductoChunking,
    ReductoDocumentUrlDocument,
    ReductoFormatting,
    ReductoParseLegacySdkInput,
    ReductoParseV3SdkInput,
    ReductoRetrieval,
    ReductoSettings,
    VertexDeepSeekOcrSdkInput,
    VertexMistralOcrSdkInput,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
_VALUE_TEXT: Final = st.just("case-1")
_MISTRAL_MODEL: Final = "mistral/mistral-ocr-latest"
_REDUCTO_API_BASE: Final = "https://platform.reducto.ai"


OcrFixtureTarget = FixtureTarget[OcrSdkInputBase]


def _image_document(text: str, font_size: int) -> MistralImageUrlDocument:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    return MistralImageUrlDocument(type="image_url", image_url=url)


def _fixture_pdf_data_uri() -> str:
    fixture: Final = Path(__file__).resolve().parents[2] / "llm_translation" / "fixtures" / "dummy.pdf"
    encoded: Final = base64.b64encode(fixture.read_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"


def _pdf_document() -> MistralDocumentUrlDocument:
    return MistralDocumentUrlDocument(type="document_url", document_url=_fixture_pdf_data_uri())


def _public_document_strategy() -> SearchStrategy[MistralImageUrlDocument | MistralDocumentUrlDocument]:
    return st.sampled_from((_image_document("invoice 123", 24), _pdf_document()))


def _annotation_format(name: str) -> JsonSchemaResponseFormat:
    return JsonSchemaResponseFormat(
        type="json_schema",
        json_schema=JsonSchemaDefinition(
            name=name,
            description="Extract the visible document fields",
            schema={
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )


def _mistral_confidence_strategy() -> SearchStrategy[str]:
    return st.just("page")


@st.composite
def mistral_input_strategy(draw: DrawFn, model: str) -> MistralOcrSdkInput:
    document: Final = draw(_public_document_strategy())
    annotation: Final = draw(
        st.sampled_from(
            (
                {},
                {"document_annotation_format": _annotation_format("document_title")},
                {
                    "document_annotation_format": _annotation_format("prompted_document_title"),
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
                "bbox_annotation_format": st.just(_annotation_format("bounding_boxes")),
                "extract_header": st.booleans(),
                "extract_footer": st.booleans(),
                "table_format": st.just("markdown"),
                "confidence_scores_granularity": _mistral_confidence_strategy(),
                "include_blocks": st.booleans(),
                "id": _VALUE_TEXT,
            },
        )
    )
    return MistralOcrSdkInput.model_validate(
        {
            "model": model,
            "document": document,
            **annotation,
            **optional_params,
        }
    )


def _as_azure_mistral(case_input: MistralOcrSdkInput, model: str) -> AzureMistralOcrSdkInput:
    values: Final = case_input.model_dump(mode="python", exclude={"boundary", "model", "custom_llm_provider"})
    return AzureMistralOcrSdkInput.model_validate({**values, "model": model})


def _as_vertex_mistral(case_input: MistralOcrSdkInput, project: str, location: str) -> VertexMistralOcrSdkInput:
    values: Final = case_input.model_dump(mode="python", exclude={"boundary", "model", "custom_llm_provider"})
    return VertexMistralOcrSdkInput.model_validate(
        {**values, "vertex_project": project, "vertex_location": location}
    )


def _required_mistral_inputs(model: str) -> tuple[MistralOcrSdkInput, ...]:
    document: Final = _image_document("invoice 123", 24)
    annotation: Final = _annotation_format("document_title")
    bbox_annotation: Final = _annotation_format("bounding_boxes")
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
    return tuple(
        MistralOcrSdkInput.model_validate({"model": model, "document": document, **case}) for case in cases
    )


def _required_azure_document_intelligence_inputs() -> tuple[AzureDocumentIntelligenceOcrSdkInput, ...]:
    document: Final = _pdf_document()
    model: Final = "azure_ai/doc-intelligence/prebuilt-layout"
    return (
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, pages=[0, 1]),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, features=["languages"]),
        AzureDocumentIntelligenceOcrSdkInput(model=model, document=document, req_format="litellm"),
    )


def _required_vertex_deepseek_inputs(project: str, location: str) -> tuple[VertexDeepSeekOcrSdkInput, ...]:
    document: Final = _image_document("invoice 123", 24)
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


def _required_reducto_v3_inputs(
    document: ReductoDocumentUrlDocument,
) -> tuple[ReductoParseV3SdkInput, ...]:
    return (
        ReductoParseV3SdkInput(model="reducto/parse-v3", document=document),
        ReductoParseV3SdkInput(
            model="reducto/parse-v3",
            document=document,
            formatting=ReductoFormatting(table_output_format="md"),
        ),
        ReductoParseV3SdkInput(
            model="reducto/parse-v3",
            document=document,
            retrieval=ReductoRetrieval(chunking=ReductoChunking(chunk_mode="page")),
        ),
        ReductoParseV3SdkInput(
            model="reducto/parse-v3",
            document=document,
            settings=ReductoSettings(return_ocr_data=True),
        ),
    )


@st.composite
def azure_document_intelligence_input_strategy(draw: DrawFn) -> AzureDocumentIntelligenceOcrSdkInput:
    optional_params: Final = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "pages": st.sampled_from(([0], [0, 1], "1-2")),
                "features": st.sampled_from((["languages"], ["keyValuePairs"], "languages,keyValuePairs")),
            },
        )
    )
    return AzureDocumentIntelligenceOcrSdkInput.model_validate(
        {
            "model": draw(
                st.sampled_from(
                    (
                        "azure_ai/doc-intelligence/prebuilt-read",
                        "azure_ai/doc-intelligence/prebuilt-layout",
                        "azure_ai/doc-intelligence/prebuilt-document",
                    )
                )
            ),
            "document": draw(_public_document_strategy()),
            **optional_params,
        }
    )


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
            "document": draw(_public_document_strategy()),
            "vertex_project": project,
            "vertex_location": location,
            **optional_params,
        }
    )


def _reducto_formatting_strategy() -> SearchStrategy[ReductoFormatting]:
    return st.builds(
        ReductoFormatting,
        add_page_markers=st.booleans(),
        table_output_format=st.sampled_from(("html", "json", "md", "jsonbbox", "dynamic", "csv")),
        merge_tables=st.booleans(),
        include=st.sampled_from(
            (
                [],
                ["hyperlinks"],
                ["change_tracking", "highlight", "comments"],
                ["signatures", "ignore_watermarks"],
            )
        ),
    )


def _reducto_chunking_strategy() -> SearchStrategy[ReductoChunking]:
    return st.one_of(
        st.builds(
            ReductoChunking,
            chunk_mode=st.sampled_from(("section", "page", "disabled", "block", "page_sections")),
            chunk_size=st.just(None),
            chunk_overlap=st.just(0),
        ),
        st.builds(
            ReductoChunking,
            chunk_mode=st.just("variable"),
            chunk_size=st.sampled_from((250, 1000, 1500)),
            chunk_overlap=st.sampled_from((0, 32, 128)),
        ),
    )


def _reducto_retrieval_strategy() -> SearchStrategy[ReductoRetrieval]:
    return st.builds(
        ReductoRetrieval,
        chunking=_reducto_chunking_strategy(),
        filter_blocks=st.sampled_from(
            (
                [],
                ["Header"],
                ["Header", "Footer", "Page Number"],
                ["Figure", "Table", "Key Value"],
            )
        ),
        embedding_optimized=st.booleans(),
    )


def _reducto_settings_strategy() -> SearchStrategy[ReductoSettings]:
    return st.builds(
        ReductoSettings,
        ocr_system=st.sampled_from(("standard", "legacy")),
        extraction_mode=st.sampled_from(("ocr", "hybrid")),
        force_url_result=st.just(False),
        return_ocr_data=st.booleans(),
        return_images=st.sampled_from(([], ["figure"], ["table"], ["page"], ["figure", "table", "page"])),
        embed_pdf_metadata=st.booleans(),
        embed_pdf_metadata_dpi=st.sampled_from((50, 100, 250)),
        persist_results=st.just(False),
        timeout=st.sampled_from((None, 300.0, 900.0)),
        page_range=st.sampled_from((None, [1], [1, 2], ["Sheet1"])),
    )


@st.composite
def reducto_v3_input_strategy(
    draw: DrawFn, document: ReductoDocumentUrlDocument | None = None
) -> ReductoParseV3SdkInput:
    model, custom_llm_provider = draw(st.sampled_from((("reducto/parse-v3", None), ("parse-v3", "reducto"))))
    options: Final = draw(
        st.fixed_dictionaries(
            {},
            optional={
                "formatting": _reducto_formatting_strategy(),
                "retrieval": _reducto_retrieval_strategy(),
                "settings": _reducto_settings_strategy(),
            },
        )
    )
    return ReductoParseV3SdkInput.model_validate(
        {
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "document": document
            or ReductoDocumentUrlDocument(type="document_url", document_url="reducto://fixture-document.pdf"),
            **options,
        }
    )


def reducto_legacy_input_strategy(
    document: ReductoDocumentUrlDocument | None = None,
) -> SearchStrategy[ReductoParseLegacySdkInput]:
    selected_document: Final = document or ReductoDocumentUrlDocument(
        type="document_url", document_url="reducto://fixture-document.pdf"
    )
    return st.sampled_from(
        (
            ReductoParseLegacySdkInput(
                model="reducto/parse-legacy",
                document=selected_document,
            ),
            ReductoParseLegacySdkInput(
                model="parse-legacy",
                custom_llm_provider="reducto",
                document=selected_document,
            ),
            ReductoParseLegacySdkInput(
                model="reducto/parse-legacy",
                document=selected_document,
                enhance={},
            ),
        )
    )


def _generate_examples(
    target: OcrFixtureTarget,
    root: Path,
    examples: int,
    concurrency: int,
) -> None:
    generate_target_fixtures(target, root, examples, concurrency, OcrParityCase)


def _mistral_upstream_base(environ: Mapping[str, str]) -> str:
    configured: Final = environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    return configured.removesuffix("/v1")


def _mistral_target(
    environ: Mapping[str, str],
    sdk_call: Callable[..., object],
) -> OcrFixtureTarget | None:
    api_key: Final = environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None

    def invoke(api_base: str, case_input: OcrSdkInputBase) -> object:
        return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

    return OcrFixtureTarget(
        name="mistral-ocr",
        provider_spec=ProviderSpec(upstream_base=_mistral_upstream_base(environ)),
        strategy=cast(SearchStrategy[OcrSdkInputBase], mistral_input_strategy(_MISTRAL_MODEL)),
        invoke=invoke,
        required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_mistral_inputs(_MISTRAL_MODEL)),
    )


def _azure_mistral_target(
    environ: Mapping[str, str], sdk_call: Callable[..., object]
) -> OcrFixtureTarget | None:
    api_key: Final = environ.get("AZURE_AI_API_KEY")
    upstream_base: Final = environ.get("AZURE_AI_API_BASE")
    configured_model: Final = environ.get("AZURE_AI_OCR_MODEL")
    if not api_key or not upstream_base or not configured_model:
        return None
    model: Final = configured_model if configured_model.startswith("azure_ai/") else f"azure_ai/{configured_model}"

    def invoke(api_base: str, case_input: OcrSdkInputBase) -> object:
        return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

    return OcrFixtureTarget(
        name="azure-mistral",
        provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
        strategy=cast(
            SearchStrategy[OcrSdkInputBase],
            mistral_input_strategy(_MISTRAL_MODEL).map(lambda case_input: _as_azure_mistral(case_input, model)),
        ),
        invoke=invoke,
        required_inputs=cast(
            tuple[OcrSdkInputBase, ...],
            tuple(_as_azure_mistral(case_input, model) for case_input in _required_mistral_inputs(_MISTRAL_MODEL)),
        ),
    )


def _azure_document_intelligence_target(
    environ: Mapping[str, str], sdk_call: Callable[..., object]
) -> OcrFixtureTarget | None:
    api_key: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    upstream_base: Final = environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not api_key or not upstream_base:
        return None

    def invoke(api_base: str, case_input: OcrSdkInputBase) -> object:
        return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

    return OcrFixtureTarget(
        name="azure-document-intelligence",
        provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
        strategy=cast(SearchStrategy[OcrSdkInputBase], azure_document_intelligence_input_strategy()),
        invoke=invoke,
        required_inputs=cast(
            tuple[OcrSdkInputBase, ...], _required_azure_document_intelligence_inputs()
        ),
    )


def _vertex_targets(
    environ: Mapping[str, str], sdk_call: Callable[..., object]
) -> tuple[OcrFixtureTarget, ...]:
    api_key: Final = environ.get("VERTEX_AI_API_KEY")
    project: Final = environ.get("VERTEXAI_PROJECT") or environ.get("VERTEX_PROJECT")
    location: Final = environ.get("VERTEXAI_LOCATION") or environ.get("VERTEX_LOCATION") or "us-central1"
    if not api_key or not project:
        return ()
    upstream_base: Final = environ.get("VERTEX_AI_API_BASE") or f"https://{location}-aiplatform.googleapis.com"

    def invoke(api_base: str, case_input: OcrSdkInputBase) -> object:
        return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

    return (
        OcrFixtureTarget(
            name="vertex-mistral",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase],
                st.builds(
                    _as_vertex_mistral,
                    case_input=mistral_input_strategy(_MISTRAL_MODEL),
                    project=st.just(project),
                    location=st.just(location),
                ),
            ),
            invoke=invoke,
            required_inputs=cast(
                tuple[OcrSdkInputBase, ...],
                tuple(
                    _as_vertex_mistral(case_input, project, location)
                    for case_input in _required_mistral_inputs(_MISTRAL_MODEL)
                ),
            ),
        ),
        OcrFixtureTarget(
            name="vertex-deepseek",
            provider_spec=ProviderSpec(upstream_base=upstream_base.rstrip("/")),
            strategy=cast(
                SearchStrategy[OcrSdkInputBase], vertex_deepseek_input_strategy(project, location)
            ),
            invoke=invoke,
            required_inputs=cast(
                tuple[OcrSdkInputBase, ...], _required_vertex_deepseek_inputs(project, location)
            ),
        ),
    )


def _reducto_targets(
    environ: Mapping[str, str], sdk_call: Callable[..., object]
) -> tuple[OcrFixtureTarget, ...]:
    api_key: Final = environ.get("REDUCTO_API_KEY")
    if not api_key:
        return ()
    upstream_base: Final = environ.get("REDUCTO_API_BASE", _REDUCTO_API_BASE).rstrip("/")
    document: Final = ReductoDocumentUrlDocument(type="document_url", document_url=_fixture_pdf_data_uri())

    def invoke(api_base: str, case_input: OcrSdkInputBase) -> object:
        return sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())

    return (
        OcrFixtureTarget(
            name="reducto-v3",
            provider_spec=ProviderSpec(upstream_base=upstream_base),
            strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_v3_input_strategy(document)),
            invoke=invoke,
            required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_reducto_v3_inputs(document)),
        ),
        OcrFixtureTarget(
            name="reducto-legacy",
            provider_spec=ProviderSpec(upstream_base=upstream_base),
            strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_legacy_input_strategy(document)),
            invoke=invoke,
            required_inputs=cast(
                tuple[OcrSdkInputBase, ...],
                (
                    ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=document),
                    ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=document, enhance={}),
                ),
            ),
        ),
    )


def discover_targets(
    environ: Mapping[str, str],
    sdk_call: Callable[..., object],
) -> tuple[OcrFixtureTarget, ...]:
    optional_targets: Final = (
        _mistral_target(environ, sdk_call),
        _azure_mistral_target(environ, sdk_call),
        _azure_document_intelligence_target(environ, sdk_call),
    )
    direct_targets: Final = tuple(target for target in optional_targets if target is not None)
    return (*direct_targets, *_vertex_targets(environ, sdk_call), *_reducto_targets(environ, sdk_call))


def require_targets(targets: tuple[OcrFixtureTarget, ...]) -> tuple[OcrFixtureTarget, ...]:
    return require_fixture_targets(
        targets,
        "No OCR fixture providers are configured. Set a supported provider API key and endpoint",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_generator_args()
    sdk_call: Final = cast(Callable[..., object], litellm.ocr)
    targets: Final = require_targets(discover_targets(os.environ, sdk_call))
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name("fixtures"),
    )
    use_litellm_rust(False, ocr=None, aocr=None)
    for target in targets:
        _generate_examples(target, root, args.examples, args.concurrency)


if __name__ == "__main__":
    main()
