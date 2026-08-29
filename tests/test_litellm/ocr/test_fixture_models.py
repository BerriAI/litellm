from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from tests.test_litellm.ocr.fixture_models import LiteLLMOcrInput


@pytest.mark.parametrize(
    ("raw_input", "expected_params"),
    (
        (
            {
                "model": "azure_ai/doc-intelligence/prebuilt-layout",
                "document": {"type": "document_url", "document_url": "https://example.com/document.pdf"},
                "pages": "1-3,5",
                "features": ["keyValuePairs", "languages"],
            },
            {"pages": "1-3,5", "features": ["keyValuePairs", "languages"]},
        ),
        (
            {
                "model": "reducto/parse-v3",
                "document": {"type": "document_url", "document_url": "reducto://fixture-file"},
                "formatting": {"table_output_format": "html"},
                "retrieval": {"chunking": {"chunk_mode": "variable"}},
            },
            {
                "formatting": {"table_output_format": "html"},
                "retrieval": {"chunking": {"chunk_mode": "variable"}},
            },
        ),
    ),
)
def test_litellm_ocr_input_preserves_provider_params(
    raw_input: dict[str, object], expected_params: dict[str, object]
) -> None:
    fixture_input: Final = LiteLLMOcrInput.model_validate(raw_input)
    sdk_kwargs: Final = fixture_input.as_sdk_kwargs()
    canonical_input: Final = fixture_input.canonical_input()

    assert {name: sdk_kwargs[name] for name in expected_params} == expected_params
    assert {name: canonical_input[name] for name in expected_params} == expected_params


def test_litellm_ocr_input_rejects_non_json_provider_params() -> None:
    with pytest.raises(ValidationError):
        LiteLLMOcrInput.model_validate(
            {
                "model": "reducto/parse-v3",
                "document": {"type": "document_url", "document_url": "reducto://fixture-file"},
                "settings": object(),
            }
        )
