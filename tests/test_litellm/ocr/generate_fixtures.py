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
from hypothesis.strategies import SearchStrategy

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.test_litellm._fixture_recorder import (
    ProviderSpec,
    fixture_directory,
    record_cases,
)
from tests.test_litellm.ocr.fixture_models import (
    ImageUrlDocument,
    LiteLLMOcrInput,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
LOGGER: Final = logging.getLogger(__name__)
_TEXT: Final = st.from_regex(r"[A-Za-z0-9 ]{1,24}", fullmatch=True)
_VALUE_TEXT: Final = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 -_", min_size=1, max_size=32)


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None
    model: str


def _image_document(text: str, font_size: int) -> ImageUrlDocument:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    return ImageUrlDocument(type="image_url", image_url=url)


def _mistral_input_strategy(model: str) -> SearchStrategy[LiteLLMOcrInput]:
    document_strategy: Final = st.builds(_image_document, _TEXT, st.integers(min_value=12, max_value=36))
    input_values: Final = st.fixed_dictionaries(
        {"model": st.just(model), "document": document_strategy},
        optional={
            "include_image_base64": st.booleans(),
            "image_limit": st.integers(min_value=1, max_value=100),
            "image_min_size": st.integers(min_value=0, max_value=10_000),
            "extract_header": st.booleans(),
            "extract_footer": st.booleans(),
            "table_format": st.sampled_from(("markdown", "html")),
            "include_blocks": st.booleans(),
            "id": _VALUE_TEXT,
        },
    )
    return input_values.map(LiteLLMOcrInput.model_validate)


def _generate_examples(
    spec: ProviderSpec,
    root: Path,
    examples: int,
    concurrency: int,
    sdk_call: Callable[..., object],
) -> None:
    generated: Final[queue.SimpleQueue[LiteLLMOcrInput | None]] = queue.SimpleQueue()

    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(case_input=_mistral_input_strategy(spec.model))
    def generate_case(case_input: LiteLLMOcrInput) -> None:
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
    parser.add_argument("--model", default="mistral/mistral-ocr-latest")
    namespace: Final = parser.parse_args()
    return GeneratorArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
        model=cast(str, namespace.model),
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
