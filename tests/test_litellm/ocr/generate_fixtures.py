from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote

from dotenv import load_dotenv
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.test_litellm._fixture_recorder import (
    ProviderSpec,
    fixture_directory,
    parse_generator_args,
    record_case,
)
from tests.test_litellm.ocr.fixture_models import (
    AnnotationFormat,
    DocumentUrlDocument,
    ImageUrlDocument,
    JsonSchemaDefinition,
    MistralOcrParityInput,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
LOGGER: Final = logging.getLogger(__name__)
_TEXT: Final = st.from_regex(r"[A-Za-z0-9 ]{1,24}", fullmatch=True)
_VALUE_TEXT: Final = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 -_", min_size=1, max_size=32)
_ANNOTATION_FORMAT: Final[SearchStrategy[AnnotationFormat]] = st.builds(
    AnnotationFormat,
    type=st.just("json_schema"),
    json_schema=st.builds(
        JsonSchemaDefinition,
        name=_VALUE_TEXT,
        schema_value=st.just({"type": "object", "properties": {}, "additionalProperties": False}),
        strict=st.one_of(st.none(), st.booleans()),
    ),
)


def _image_document(text: str, font_size: int) -> ImageUrlDocument:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    return ImageUrlDocument(type="image_url", image_url=url)


def _input_strategy(model: str) -> SearchStrategy[MistralOcrParityInput]:
    document_strategy: Final = st.one_of(
        st.builds(_image_document, _TEXT, st.integers(min_value=12, max_value=36)),
        st.just(
            DocumentUrlDocument(
                type="document_url",
                document_url="https://arxiv.org/pdf/2201.04234",
            )
        ),
    )
    input_values: Final = st.fixed_dictionaries(
        {"model": st.just(model), "document": document_strategy},
        optional={
            "pages": st.one_of(
                st.none(),
                st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=5, unique=True),
            ),
            "include_image_base64": st.one_of(st.none(), st.booleans()),
            "image_limit": st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
            "image_min_size": st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)),
            "bbox_annotation_format": st.one_of(st.none(), _ANNOTATION_FORMAT),
            "document_annotation_format": st.one_of(st.none(), _ANNOTATION_FORMAT),
            "document_annotation_prompt": st.one_of(st.none(), _VALUE_TEXT),
            "extract_header": st.one_of(st.none(), st.booleans()),
            "extract_footer": st.one_of(st.none(), st.booleans()),
            "table_format": st.one_of(st.none(), st.sampled_from(("markdown", "html"))),
            "confidence_scores_granularity": st.one_of(
                st.none(), st.sampled_from(("word", "page", "block"))
            ),
            "include_blocks": st.one_of(st.none(), st.booleans()),
            "id": st.one_of(st.none(), _VALUE_TEXT),
        },
    )
    return input_values.map(MistralOcrParityInput.model_validate)


def _generate_examples(
    spec: ProviderSpec,
    root: Path,
    examples: int,
    sdk_call: Callable[..., object],
) -> None:
    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(case_input=_input_strategy(spec.model))
    def generate_case(case_input: MistralOcrParityInput) -> None:
        result: Final = record_case(spec, root, case_input, sdk_call)
        LOGGER.info("%s %s", "cached" if result.cache_hit else "recorded", result.case.input.model)

    generate_case()


def _mistral_upstream_base() -> str:
    configured: Final = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    return configured.removesuffix("/v1")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_generator_args()
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
    _generate_examples(spec, root, args.examples, cast(Callable[..., object], litellm.ocr))


if __name__ == "__main__":
    main()
