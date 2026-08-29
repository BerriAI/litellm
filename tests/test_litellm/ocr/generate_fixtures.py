from __future__ import annotations

import base64
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from pydantic import TypeAdapter

import litellm
from litellm.llms.reducto.common import (
    REDUCTO_API_BASE,
    extract_file_id_or_bytes,
    upload_bytes_sync,
)
from tests.test_litellm._fixture_recorder import (
    ProviderSpec,
    RecorderResult,
    fill_missing_responses,
    fixture_directory,
    parse_generator_args,
    record_case,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
JSON_OBJECT: Final = TypeAdapter(dict[str, object])
PROVIDER_NAMES: Final = ("mistral", "reducto")
LOGGER: Final = logging.getLogger(__name__)
_TEXT: Final = st.from_regex(r"[A-Za-z0-9 ]{1,24}", fullmatch=True)
_VALUE_TEXT: Final = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 -_", min_size=1, max_size=32)
_NULLABLE_TEXT: Final = st.one_of(st.none(), _VALUE_TEXT)
_POSITIVE_INTEGER: Final = st.integers(min_value=1, max_value=10_000)
_NON_NEGATIVE_INTEGER: Final = st.integers(min_value=0, max_value=10_000)
_SMALL_NUMBER: Final = st.floats(min_value=0.01, max_value=30.0, allow_nan=False, allow_infinity=False)
_REQUEST_ONLY_IMAGE: Final = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_BLOCK_TYPES: Final = (
    "Header",
    "Footer",
    "Title",
    "Section Header",
    "Page Number",
    "List Item",
    "Figure",
    "Table",
    "Key Value",
    "Text",
    "Comment",
    "Signature",
)


def _optional_object(optional: dict[str, SearchStrategy[object]]) -> SearchStrategy[dict[str, object]]:
    return st.fixed_dictionaries({}, optional=optional)


def _merge_objects(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    return {**left, **right}


def _page_range(start: int, length: int) -> dict[str, object]:
    return {"start": start, "end": start + length}


def _annotation_format(name: str, strict: bool) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "strict": strict,
        },
    }


def _annotation_format_only(value: dict[str, object]) -> dict[str, object]:
    return {"document_annotation_format": value}


def _null_annotation_format() -> dict[str, object]:
    return {"document_annotation_format": None}


def _annotation_format_and_prompt(value: dict[str, object], prompt: str | None) -> dict[str, object]:
    return {"document_annotation_format": value, "document_annotation_prompt": prompt}


def _table_agentic(prompt: str | None, mode: str) -> dict[str, object]:
    return {"scope": "table", "prompt": prompt, "mode": mode}


def _figure_agentic(prompt: str | None, advanced: bool, overlays: bool) -> dict[str, object]:
    return {
        "scope": "figure",
        "prompt": prompt,
        "advanced_chart_agent": advanced,
        "return_overlays": overlays,
    }


def _text_agentic(prompt: str | None) -> dict[str, object]:
    return {"scope": "text", "prompt": prompt}


def _agentic_scope(value: dict[str, object]) -> object:
    return value["scope"]


_PAGE_RANGE: Final = st.builds(
    _page_range, st.integers(min_value=1, max_value=1000), st.integers(min_value=0, max_value=50)
)
_PAGE_RANGE_VALUE: Final = st.one_of(
    _PAGE_RANGE,
    st.lists(_PAGE_RANGE, min_size=1, max_size=3),
    st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=5, unique=True),
    st.lists(_VALUE_TEXT, min_size=1, max_size=3, unique=True),
)
_ANNOTATION_FORMAT: Final[SearchStrategy[dict[str, object]]] = st.builds(_annotation_format, _VALUE_TEXT, st.booleans())
_DOCUMENT_ANNOTATION: Final[SearchStrategy[dict[str, object]]] = st.one_of(
    st.just(dict[str, object]()),
    st.just(_null_annotation_format()),
    st.builds(_annotation_format_only, _ANNOTATION_FORMAT),
    st.builds(_annotation_format_and_prompt, _ANNOTATION_FORMAT, _NULLABLE_TEXT),
)


def _mistral_options(model: str) -> SearchStrategy[dict[str, object]]:
    model_name: Final = model.rsplit("/", 1)[-1]
    confidence_values: Final = ("word", "page") if model_name == "mistral-ocr-4-0" else ("word", "page", "block")
    confidence_option: Final[dict[str, SearchStrategy[object]]] = (
        {}
        if model_name == "mistral-ocr-2512"
        else {"confidence_scores_granularity": st.one_of(st.none(), st.sampled_from(confidence_values))}
    )
    independent: Final = _optional_object(
        {
            "pages": st.one_of(
                st.none(),
                st.lists(_NON_NEGATIVE_INTEGER, min_size=1, max_size=5, unique=True),
                st.sampled_from(("0", "0,1,2", "0-5", "0,2-4")),
            ),
            "include_image_base64": st.one_of(st.none(), st.booleans()),
            "image_limit": st.one_of(st.none(), _POSITIVE_INTEGER),
            "image_min_size": st.one_of(st.none(), _NON_NEGATIVE_INTEGER),
            "bbox_annotation_format": st.one_of(st.none(), _ANNOTATION_FORMAT),
            "extract_header": st.booleans(),
            "extract_footer": st.booleans(),
            "table_format": st.one_of(st.none(), st.sampled_from(("markdown", "html"))),
            "include_blocks": st.booleans(),
            **confidence_option,
        }
    )
    return st.builds(_merge_objects, independent, _DOCUMENT_ANNOTATION)


_AGENTIC_TABLE: Final[SearchStrategy[dict[str, object]]] = st.builds(
    _table_agentic,
    _NULLABLE_TEXT,
    st.sampled_from(("default", "auto", "max")),
)
_AGENTIC_FIGURE: Final[SearchStrategy[dict[str, object]]] = st.builds(
    _figure_agentic,
    _NULLABLE_TEXT,
    st.booleans(),
    st.booleans(),
)
_AGENTIC_TEXT: Final[SearchStrategy[dict[str, object]]] = st.builds(_text_agentic, _NULLABLE_TEXT)
_REDUCTO_ENHANCE: Final = _optional_object(
    {
        "agentic": st.lists(
            st.one_of(_AGENTIC_TABLE, _AGENTIC_FIGURE, _AGENTIC_TEXT),
            max_size=3,
            unique_by=_agentic_scope,
        ),
        "summarize_figures": st.booleans(),
        "intelligent_ordering": st.booleans(),
    }
)
_CHUNKING: Final = _optional_object(
    {
        "chunk_mode": st.sampled_from(("variable", "section", "page", "disabled", "block", "page_sections")),
        "chunk_size": st.one_of(st.none(), _POSITIVE_INTEGER),
        "chunk_overlap": _NON_NEGATIVE_INTEGER,
    }
)
_LEGACY_CHUNKING: Final = _optional_object(
    {
        "chunk_mode": st.sampled_from(("variable", "section", "page", "disabled", "block", "page_sections")),
        "chunk_size": _POSITIVE_INTEGER,
        "chunk_overlap": _NON_NEGATIVE_INTEGER,
    }
)
_REDUCTO_RETRIEVAL: Final = _optional_object(
    {
        "chunking": _CHUNKING,
        "filter_blocks": st.lists(st.sampled_from(_BLOCK_TYPES), max_size=len(_BLOCK_TYPES), unique=True),
        "embedding_optimized": st.booleans(),
    }
)
_REDUCTO_FORMATTING: Final = _optional_object(
    {
        "add_page_markers": st.booleans(),
        "table_output_format": st.sampled_from(("html", "json", "md", "jsonbbox", "dynamic", "csv")),
        "merge_tables": st.booleans(),
        "include": st.lists(
            st.sampled_from(
                ("change_tracking", "highlight", "comments", "hyperlinks", "signatures", "ignore_watermarks")
            ),
            max_size=6,
            unique=True,
        ),
    }
)
_SPLIT_TABLE_SIZE: Final = st.one_of(
    _POSITIVE_INTEGER,
    _optional_object(
        {"row": st.one_of(st.none(), _POSITIVE_INTEGER), "column": st.one_of(st.none(), _POSITIVE_INTEGER)}
    ),
)
_REDUCTO_SPREADSHEET: Final = _optional_object(
    {
        "split_large_tables": _optional_object({"enabled": st.booleans(), "size": _SPLIT_TABLE_SIZE}),
        "include": st.lists(st.sampled_from(("cell_colors", "formula", "dropdowns")), max_size=3, unique=True),
        "clustering": st.sampled_from(("accurate", "fast", "disabled")),
        "exclude": st.lists(
            st.sampled_from(("hidden_sheets", "hidden_rows", "hidden_cols", "styling", "spreadsheet_images")),
            max_size=5,
            unique=True,
        ),
        "max_cell_count": st.one_of(st.none(), _POSITIVE_INTEGER),
    }
)
_TENANT_THROTTLING: Final = st.fixed_dictionaries(
    {"tenant_id": st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=1, max_size=256)},
    optional={"max_share": st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)},
)
_REDUCTO_SETTINGS: Final = _optional_object(
    {
        "ocr_system": st.sampled_from(("standard", "legacy")),
        "extraction_mode": st.sampled_from(("ocr", "hybrid")),
        "force_url_result": st.booleans(),
        "force_file_extension": _NULLABLE_TEXT,
        "return_ocr_data": st.booleans(),
        "return_images": st.lists(st.sampled_from(("figure", "table", "page")), max_size=3, unique=True),
        "embed_pdf_metadata": st.booleans(),
        "embed_pdf_metadata_dpi": st.integers(min_value=50, max_value=250),
        "persist_results": st.booleans(),
        "tenant_throttling": st.one_of(st.none(), _TENANT_THROTTLING),
        "timeout": st.one_of(st.none(), _SMALL_NUMBER),
        "page_range": st.one_of(st.none(), _PAGE_RANGE_VALUE),
        "document_password": _NULLABLE_TEXT,
        "hybrid_vpc": _optional_object({"environment": _NULLABLE_TEXT}),
    }
)
_REDUCTO_V3_OPTIONS: Final = _optional_object(
    {
        "enhance": _REDUCTO_ENHANCE,
        "retrieval": _REDUCTO_RETRIEVAL,
        "formatting": _REDUCTO_FORMATTING,
        "spreadsheet": _REDUCTO_SPREADSHEET,
        "settings": _REDUCTO_SETTINGS,
    }
)
_LEGACY_SUMMARY: Final = _optional_object(
    {"enabled": st.booleans(), "prompt": _VALUE_TEXT, "override": st.booleans(), "advanced_chart_agent": st.booleans()}
)
_REDUCTO_LEGACY_OPTIONS: Final = _optional_object(
    {
        "ocr_mode": st.sampled_from(("standard", "agentic")),
        "extraction_mode": st.sampled_from(("ocr", "metadata", "hybrid")),
        "chunking": _LEGACY_CHUNKING,
        "table_summary": _optional_object({"enabled": st.booleans(), "prompt": _VALUE_TEXT}),
        "figure_summary": _LEGACY_SUMMARY,
        "filter_blocks": st.lists(st.sampled_from(_BLOCK_TYPES), max_size=len(_BLOCK_TYPES), unique=True),
        "force_url_result": st.booleans(),
    }
)
_REDUCTO_LEGACY_ADVANCED: Final = _optional_object(
    {
        "ocr_system": st.sampled_from(("highres", "multilingual", "combined", "reducto", "legacy")),
        "table_output_format": st.sampled_from(("html", "json", "md", "jsonbbox", "dynamic", "ai_json", "csv")),
        "merge_tables": st.booleans(),
        "include_formula_information": st.booleans(),
        "include_color_information": st.booleans(),
        "include_dropdown_information": st.booleans(),
        "continue_hierarchy": st.booleans(),
        "keep_line_breaks": st.booleans(),
        "page_range": _PAGE_RANGE_VALUE,
        "force_file_extension": _VALUE_TEXT,
        "large_table_chunking": _optional_object({"enabled": st.booleans(), "size": _POSITIVE_INTEGER}),
        "spreadsheet_table_clustering": st.sampled_from(("default", "disabled", "intelligent")),
        "max_cell_count": st.one_of(st.none(), _POSITIVE_INTEGER),
        "add_page_markers": st.booleans(),
        "remove_text_formatting": st.booleans(),
        "return_ocr_data": st.booleans(),
        "document_password": _VALUE_TEXT,
        "filter_line_numbers": st.booleans(),
        "read_comments": st.booleans(),
        "persist_results": st.booleans(),
        "exclude_hidden_sheets": st.booleans(),
        "exclude_hidden_rows_cols": st.booleans(),
        "enable_change_tracking": st.booleans(),
        "enable_highlight_detection": st.booleans(),
        "ignore_watermarks": st.booleans(),
    }
)
_REDUCTO_LEGACY_EXPERIMENTAL: Final = _optional_object(
    {
        "enrich": _optional_object(
            {
                "enabled": st.booleans(),
                "mode": st.sampled_from(("standard", "page", "table", "table_auto")),
                "prompt": _VALUE_TEXT,
            }
        ),
        "layout_enrichment": st.booleans(),
        "enable_checkboxes": st.booleans(),
        "enable_equations": st.booleans(),
        "rotate_pages": st.booleans(),
        "rotate_figures": st.booleans(),
        "enable_scripts": st.booleans(),
        "return_figure_images": st.booleans(),
        "return_table_images": st.booleans(),
        "return_page_images": st.booleans(),
        "layout_model": st.sampled_from(("default", "beta")),
        "embed_text_metadata_pdf": st.booleans(),
        "embed_pdf_metadata_dpi": st.integers(min_value=50, max_value=250),
        "detect_signatures": st.booleans(),
        "danger_filter_wide_boxes": st.booleans(),
        "user_specified_timeout_seconds": st.one_of(st.none(), _SMALL_NUMBER),
    }
)
_REDUCTO_LEGACY_ROOT: Final = _optional_object(
    {
        "options": _REDUCTO_LEGACY_OPTIONS,
        "advanced_options": _REDUCTO_LEGACY_ADVANCED,
        "experimental_options": _REDUCTO_LEGACY_EXPERIMENTAL,
        "priority": st.booleans(),
    }
)


def _mistral_upstream_base() -> str:
    configured: Final = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    return configured.removesuffix("/v1")


def _provider_specs(
    selected: tuple[str, ...],
    requests_only: bool = False,
    all_models: bool = False,
) -> tuple[ProviderSpec, ...]:
    mistral_key: Final = os.environ.get("MISTRAL_API_KEY") or os.environ.get("LITELLM_API_KEY")
    reducto_key: Final = os.environ.get("REDUCTO_API_KEY")
    mistral_model: Final = os.environ.get("MISTRAL_OCR_MODEL")
    reducto_model: Final = os.environ.get("REDUCTO_OCR_MODEL")
    mistral_models: Final = (
        (mistral_model,)
        if mistral_model is not None
        else (
            ("mistral/mistral-ocr-2512", "mistral/mistral-ocr-4-0", "mistral/mistral-ocr-4-1")
            if requests_only or all_models
            else ("mistral/mistral-ocr-latest",)
        )
    )
    reducto_models: Final = (
        (reducto_model,) if reducto_model is not None else ("reducto/parse-v3", "reducto/parse-legacy")
    )
    mistral_specs: Final = (
        tuple(
            ProviderSpec(
                name="mistral",
                model=model,
                upstream_base=_mistral_upstream_base(),
                api_key=mistral_key or "request-only-key",
                upstream_model=os.environ.get("MISTRAL_OCR_UPSTREAM_MODEL"),
            )
            for model in mistral_models
        )
        if "mistral" in selected and (mistral_key is not None or requests_only)
        else ()
    )
    reducto_specs: Final = (
        tuple(
            ProviderSpec(
                name="reducto",
                model=model,
                upstream_base=os.environ.get("REDUCTO_API_BASE", REDUCTO_API_BASE),
                api_key=reducto_key or "request-only-key",
            )
            for model in reducto_models
        )
        if "reducto" in selected and (reducto_key is not None or requests_only)
        else ()
    )
    specs: Final = (*mistral_specs, *reducto_specs)
    present: Final = frozenset(spec.name for spec in specs)
    missing: Final = tuple(name for name in selected if name not in present)
    if missing:
        LOGGER.warning("Skipping providers without credentials: %s", ", ".join(missing))
    return specs


def _image_data_uri(text: str, font_size: int) -> str:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    response: Final = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    raw_content_type: Final = cast(object, response.headers.get("content-type", "image/png"))
    content_type: Final = raw_content_type if isinstance(raw_content_type, str) else "image/png"
    media_type: Final = content_type.split(";", 1)[0]
    encoded: Final = base64.b64encode(response.content).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _sdk_kwargs(
    spec: ProviderSpec,
    image_data_uri: str,
    options: dict[str, object],
) -> dict[str, object]:
    return {
        "model": spec.model,
        "document": {"type": "image_url", "image_url": image_data_uri},
        **options,
    }


def _upload_reducto_document(
    spec: ProviderSpec,
    sdk_kwargs: dict[str, object],
    requests_only: bool,
) -> dict[str, object]:
    if spec.name != "reducto":
        return sdk_kwargs
    if requests_only:
        return {**sdk_kwargs, "document": {"type": "image_url", "image_url": "reducto://fixture"}}
    if spec.api_key is None:
        raise ValueError("Reducto response fixture generation requires REDUCTO_API_KEY")
    document: Final = JSON_OBJECT.validate_python(sdk_kwargs["document"])
    image_data_uri: Final = document.get("image_url")
    if not isinstance(image_data_uri, str):
        raise ValueError("Reducto fixture generation requires an image_url data URI")
    _, raw_bytes, mime = extract_file_id_or_bytes(image_data_uri, model=spec.model)
    file_id: Final = upload_bytes_sync(
        raw_bytes=raw_bytes or b"",
        mime=mime,
        api_key=spec.api_key,
        api_base=spec.upstream_base,
    )
    return {**sdk_kwargs, "document": {"type": "image_url", "image_url": file_id}}


def _generate_provider_case(
    spec: ProviderSpec,
    root: Path,
    image_data_uri: str,
    options: dict[str, object],
    requests_only: bool,
    sdk_call: Callable[..., object],
) -> None:
    sdk_kwargs: Final = _upload_reducto_document(spec, _sdk_kwargs(spec, image_data_uri, options), requests_only)
    result: Final = record_case(spec, root, sdk_kwargs, requests_only, sdk_call)
    state: Final = "cached" if result.cache_hit else "recorded request" if requests_only else "recorded"
    LOGGER.info("%s %s %s", state, spec.name, result.request.provider_request.path)


def _generate_provider_examples(
    spec: ProviderSpec,
    root: Path,
    examples: int,
    requests_only: bool,
    sdk_call: Callable[..., object],
) -> None:
    options_strategy: Final = _options_strategy(spec)
    image_strategy: Final = (
        st.just(_REQUEST_ONLY_IMAGE)
        if requests_only
        else st.builds(_image_data_uri, _TEXT, st.integers(min_value=12, max_value=36))
    )

    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(image_data_uri=image_strategy, options=options_strategy)
    def generate_case(image_data_uri: str, options: dict[str, object]) -> None:
        _generate_provider_case(spec, root, image_data_uri, options, requests_only, sdk_call)

    generate_case()


def _options_strategy(spec: ProviderSpec) -> SearchStrategy[dict[str, object]]:
    model: Final = spec.model.rsplit("/", 1)[-1]
    if spec.name == "mistral":
        return _mistral_options(spec.model)
    if model == "parse-v3":
        return _REDUCTO_V3_OPTIONS
    if model == "parse-legacy":
        return _REDUCTO_LEGACY_ROOT
    raise ValueError(f"Unsupported Reducto OCR fixture model: {spec.model}")


def _generate(
    specs: tuple[ProviderSpec, ...],
    root: Path,
    examples: int,
    requests_only: bool,
    sdk_call: Callable[..., object],
) -> None:
    for spec in specs:
        _generate_provider_examples(spec, root, examples, requests_only, sdk_call)


def _log_filled_responses(results: tuple[RecorderResult, ...]) -> None:
    for result in results:
        LOGGER.info("filled response %s %s", result.request.provider, result.request.provider_request.path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_generator_args(PROVIDER_NAMES)
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name(".fixtures"),
    )
    specs: Final = _provider_specs(
        args.providers,
        requests_only=args.requests_only,
        all_models=args.responses_only,
    )
    if not specs:
        raise SystemExit("No selected provider has the required credentials")
    ocr_call: Final = cast(Callable[..., object], litellm.ocr)
    if args.responses_only:
        _log_filled_responses(fill_missing_responses(specs, root, ocr_call))
        return
    _generate(specs, root, args.examples, args.requests_only, ocr_call)


if __name__ == "__main__":
    main()
