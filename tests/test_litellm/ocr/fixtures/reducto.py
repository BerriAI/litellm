from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, SearchStrategy

from tests.route_parity.fixture_recorder import ProviderSpec
from tests.test_litellm.ocr.fixtures.common import (
    OcrFixtureClient,
    OcrFixtureTarget,
    fixture_pdf_data_uri,
    invoke_with_api_key,
)
from tests.test_litellm.ocr.fixtures.models import (
    OcrSdkInputBase,
    ReductoChunking,
    ReductoDocumentUrlDocument,
    ReductoFormatting,
    ReductoParseLegacySdkInput,
    ReductoParseV3SdkInput,
    ReductoRetrieval,
    ReductoSettings,
)

_REDUCTO_API_BASE: Final = "https://platform.reducto.ai"


def _formatting_strategy() -> SearchStrategy[ReductoFormatting]:
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


def _chunking_strategy() -> SearchStrategy[ReductoChunking]:
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


def _retrieval_strategy() -> SearchStrategy[ReductoRetrieval]:
    return st.builds(
        ReductoRetrieval,
        chunking=_chunking_strategy(),
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


def _settings_strategy() -> SearchStrategy[ReductoSettings]:
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
                "formatting": _formatting_strategy(),
                "retrieval": _retrieval_strategy(),
                "settings": _settings_strategy(),
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
            ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=selected_document),
            ReductoParseLegacySdkInput(model="parse-legacy", custom_llm_provider="reducto", document=selected_document),
            ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=selected_document, enhance={}),
        )
    )


def _required_v3_inputs(document: ReductoDocumentUrlDocument) -> tuple[ReductoParseV3SdkInput, ...]:
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


class ReductoFixtureSource:
    def targets(self, environ: Mapping[str, str], client: OcrFixtureClient) -> tuple[OcrFixtureTarget, ...]:
        api_key: Final = environ.get("REDUCTO_API_KEY")
        if not api_key:
            return ()
        upstream_base: Final = environ.get("REDUCTO_API_BASE", _REDUCTO_API_BASE).rstrip("/")
        document: Final = ReductoDocumentUrlDocument(type="document_url", document_url=fixture_pdf_data_uri())
        invocation: Final = invoke_with_api_key(client, api_key)
        return (
            OcrFixtureTarget(
                name="reducto-v3",
                provider_spec=ProviderSpec(upstream_base=upstream_base),
                strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_v3_input_strategy(document)),
                invocation=invocation,
                required_inputs=cast(tuple[OcrSdkInputBase, ...], _required_v3_inputs(document)),
            ),
            OcrFixtureTarget(
                name="reducto-legacy",
                provider_spec=ProviderSpec(upstream_base=upstream_base),
                strategy=cast(SearchStrategy[OcrSdkInputBase], reducto_legacy_input_strategy(document)),
                invocation=invocation,
                required_inputs=cast(
                    tuple[OcrSdkInputBase, ...],
                    (
                        ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=document),
                        ReductoParseLegacySdkInput(model="reducto/parse-legacy", document=document, enhance={}),
                    ),
                ),
            ),
        )
