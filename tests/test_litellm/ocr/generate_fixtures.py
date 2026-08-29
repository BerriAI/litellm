from __future__ import annotations

import argparse
import logging
import os
import queue
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote

from dotenv import load_dotenv
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.test_litellm._fixture_recorder import (
    ProviderSpec,
    fixture_directory,
    record_cases,
)
from tests.test_litellm.ocr.fixture_models import (
    JsonSchemaDefinition,
    JsonSchemaResponseFormat,
    MistralImageUrlDocument,
    MistralModel,
    MistralOcrSdkInput,
    ReductoChunking,
    ReductoDocumentUrlDocument,
    ReductoFormatting,
    ReductoParseLegacySdkInput,
    ReductoParseV3SdkInput,
    ReductoRetrieval,
    ReductoSettings,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
LOGGER: Final = logging.getLogger(__name__)
_TEXT: Final = st.just("invoice 123")
_VALUE_TEXT: Final = st.just("case-1")
_FONT_SIZE: Final = st.just(24)
_MISTRAL_MODELS: Final = (
    "mistral/mistral-ocr-2512",
    "mistral/mistral-ocr-4-0",
    "mistral/mistral-ocr-4-1",
    "mistral/mistral-ocr-4",
    "mistral/mistral-ocr-latest",
)


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None
    model: str


def _image_document(text: str, font_size: int) -> MistralImageUrlDocument:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    return MistralImageUrlDocument(type="image_url", image_url=url)


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
    document: Final = draw(st.builds(_image_document, _TEXT, _FONT_SIZE))
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
        force_url_result=st.booleans(),
        return_ocr_data=st.booleans(),
        return_images=st.sampled_from(([], ["figure"], ["table"], ["page"], ["figure", "table", "page"])),
        embed_pdf_metadata=st.booleans(),
        embed_pdf_metadata_dpi=st.sampled_from((50, 100, 250)),
        persist_results=st.just(False),
        timeout=st.sampled_from((None, 300.0, 900.0)),
        page_range=st.sampled_from((None, [1], [1, 2], ["Sheet1"])),
    )


@st.composite
def reducto_v3_input_strategy(draw: DrawFn) -> ReductoParseV3SdkInput:
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
            "document": ReductoDocumentUrlDocument(
                type="document_url",
                document_url="reducto://fixture-document.pdf",
            ),
            **options,
        }
    )


def reducto_legacy_input_strategy() -> SearchStrategy[ReductoParseLegacySdkInput]:
    return st.sampled_from(
        (
            ReductoParseLegacySdkInput(
                model="reducto/parse-legacy",
                document=ReductoDocumentUrlDocument(
                    type="document_url",
                    document_url="reducto://fixture-document.pdf",
                ),
            ),
            ReductoParseLegacySdkInput(
                model="parse-legacy",
                custom_llm_provider="reducto",
                document=ReductoDocumentUrlDocument(
                    type="document_url",
                    document_url="reducto://fixture-document.pdf",
                ),
            ),
        )
    )


def _generate_examples(
    spec: ProviderSpec,
    root: Path,
    examples: int,
    concurrency: int,
    sdk_call: Callable[..., object],
) -> None:
    generated: Final[queue.SimpleQueue[MistralOcrSdkInput | None]] = queue.SimpleQueue()

    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(case_input=mistral_input_strategy(spec.model))
    def generate_case(case_input: MistralOcrSdkInput) -> None:
        generated.put(case_input)

    generate_case()
    generated.put(None)
    case_inputs: Final = tuple(iter(generated.get, None))
    results: Final = record_cases(spec, root, case_inputs, sdk_call, concurrency)
    for result in results:
        LOGGER.info("%s %s", "cached" if result.cache_hit else "recorded", result.case.litellm_input.model)


def _mistral_upstream_base() -> str:
    configured: Final = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    return configured.removesuffix("/v1")


def _parse_args() -> GeneratorArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--model", choices=_MISTRAL_MODELS, default="mistral/mistral-ocr-latest")
    namespace: Final = parser.parse_args()
    return GeneratorArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
        model=cast(MistralModel, namespace.model),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = _parse_args()
    api_key: Final = os.environ.get("MISTRAL_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if api_key is None:
        raise SystemExit("MISTRAL_API_KEY is required")
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name(".fixtures"),
    )
    spec: Final = ProviderSpec(model=args.model, upstream_base=_mistral_upstream_base(), api_key=api_key)
    use_litellm_rust(False, ocr=None, aocr=None)
    _generate_examples(spec, root, args.examples, args.concurrency, cast(Callable[..., object], litellm.ocr))


if __name__ == "__main__":
    main()
