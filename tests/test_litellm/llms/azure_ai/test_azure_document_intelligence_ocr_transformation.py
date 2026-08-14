from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)


def test_should_encode_azure_document_intelligence_model_id():
    config = AzureDocumentIntelligenceOCRConfig()

    url = config.get_complete_url(
        api_base="https://example.cognitiveservices.azure.com",
        model="prebuilt-layout?x=1#frag",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://example.cognitiveservices.azure.com/documentintelligence/documentModels/prebuilt-layout%3Fx%3D1%23frag:analyze?api-version=2024-11-30"
    )


def test_should_reject_dot_segment_azure_document_intelligence_model_id():
    config = AzureDocumentIntelligenceOCRConfig()

    with pytest.raises(ValueError, match="model_id cannot be a dot path segment"):
        config.get_complete_url(
            api_base="https://example.cognitiveservices.azure.com",
            model="azure_ai/doc-intelligence/..",
            optional_params={},
            litellm_params={},
        )


AZURE_TABLES = [
    {
        "rowCount": 2,
        "columnCount": 2,
        "cells": [
            {"kind": "columnHeader", "rowIndex": 0, "columnIndex": 0, "content": "Item"},
            {"kind": "columnHeader", "rowIndex": 0, "columnIndex": 1, "content": "Price"},
            {"rowIndex": 1, "columnIndex": 0, "content": "Widget"},
            {"rowIndex": 1, "columnIndex": 1, "content": "$100.00"},
        ],
    },
    {
        "rowCount": 1,
        "columnCount": 1,
        "cells": [{"rowIndex": 0, "columnIndex": 0, "content": "Totals"}],
    },
]

AZURE_KEY_VALUE_PAIRS = [
    {"key": {"content": "Invoice No"}, "value": {"content": "INV-12345"}, "confidence": 0.98},
    {"key": {"content": "Total"}, "value": {"content": "$100.00"}, "confidence": 0.95},
]

AZURE_ANALYZE_SUCCEEDED = {
    "status": "succeeded",
    "createdDateTime": "2026-07-02T00:00:00Z",
    "lastUpdatedDateTime": "2026-07-02T00:00:05Z",
    "analyzeResult": {
        "apiVersion": "2024-11-30",
        "modelId": "prebuilt-layout",
        "content": "Invoice\nInvoice No: INV-12345\nTotal: $100.00",
        "pages": [
            {
                "pageNumber": 1,
                "width": 8.5,
                "height": 11,
                "unit": "inch",
                "lines": [
                    {"content": "Invoice"},
                    {"content": "Invoice No: INV-12345"},
                    {"content": "Total: $100.00"},
                ],
            }
        ],
        "tables": AZURE_TABLES,
        "keyValuePairs": AZURE_KEY_VALUE_PAIRS,
    },
}


def _completed_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request("GET", "https://example.cognitiveservices.azure.com/analyzeResults/xyz"),
    )


def _assert_native_fields_preserved(serialized: dict) -> None:
    assert serialized["content"] == "Invoice\nInvoice No: INV-12345\nTotal: $100.00"
    assert serialized["tables"] == AZURE_TABLES
    assert serialized["keyValuePairs"] == AZURE_KEY_VALUE_PAIRS
    assert serialized["object"] == "ocr"
    assert serialized["usage_info"]["pages_processed"] == 1
    assert serialized["pages"][0]["index"] == 0
    assert serialized["pages"][0]["markdown"] == "Invoice\nInvoice No: INV-12345\nTotal: $100.00"
    assert serialized["pages"][0]["dimensions"] == {"width": 816, "height": 1056, "dpi": 96}


def test_transform_ocr_response_preserves_azure_native_fields():
    config = AzureDocumentIntelligenceOCRConfig()

    result = config.transform_ocr_response(
        model="azure_ai/doc-intelligence/prebuilt-layout",
        raw_response=_completed_response(AZURE_ANALYZE_SUCCEEDED),
        logging_obj=MagicMock(),
    )

    _assert_native_fields_preserved(result.model_dump())


@pytest.mark.asyncio
async def test_async_transform_ocr_response_preserves_azure_native_fields():
    config = AzureDocumentIntelligenceOCRConfig()

    result = await config.async_transform_ocr_response(
        model="azure_ai/doc-intelligence/prebuilt-layout",
        raw_response=_completed_response(AZURE_ANALYZE_SUCCEEDED),
        logging_obj=MagicMock(),
    )

    _assert_native_fields_preserved(result.model_dump())


def test_transform_ocr_response_tolerates_missing_native_fields():
    config = AzureDocumentIntelligenceOCRConfig()
    payload = {
        "status": "succeeded",
        "analyzeResult": {
            "pages": [
                {
                    "pageNumber": 1,
                    "width": 8.5,
                    "height": 11,
                    "unit": "inch",
                    "lines": [{"content": "hello"}],
                }
            ],
        },
    }

    result = config.transform_ocr_response(
        model="azure_ai/doc-intelligence/prebuilt-read",
        raw_response=_completed_response(payload),
        logging_obj=MagicMock(),
    )

    serialized = result.model_dump()
    assert serialized["pages"][0]["markdown"] == "hello"
    assert serialized["content"] is None
    assert serialized["tables"] is None
    assert serialized["keyValuePairs"] is None


def test_transform_ocr_response_non_succeeded_status_raises():
    config = AzureDocumentIntelligenceOCRConfig()

    with pytest.raises(ValueError, match="failed with status: failed"):
        config.transform_ocr_response(
            model="azure_ai/doc-intelligence/prebuilt-layout",
            raw_response=_completed_response({"status": "failed"}),
            logging_obj=MagicMock(),
        )


def test_get_supported_ocr_params_includes_features():
    config = AzureDocumentIntelligenceOCRConfig()

    assert config.get_supported_ocr_params("prebuilt-layout") == ["pages", "features"]


@pytest.mark.parametrize(
    "features,expected",
    [
        (["keyValuePairs"], "keyValuePairs"),
        (["keyValuePairs", "languages"], "keyValuePairs,languages"),
        ("keyValuePairs", "keyValuePairs"),
        ("keyValuePairs,languages", "keyValuePairs,languages"),
        ("keyValuePairs, languages", "keyValuePairs,languages"),
    ],
)
def test_map_ocr_params_features(features, expected):
    config = AzureDocumentIntelligenceOCRConfig()

    mapped = config.map_ocr_params({"features": features}, {}, "prebuilt-layout")

    assert mapped == {"features": expected}


def test_map_ocr_params_empty_features_list_omitted():
    config = AzureDocumentIntelligenceOCRConfig()

    assert config.map_ocr_params({"features": []}, {}, "prebuilt-layout") == {}


@pytest.mark.parametrize(
    "features",
    [
        "keyValuePairs&pages=9",
        "key value pairs",
        "",
        [1, 2],
        [["keyValuePairs"]],
        {"feature": "keyValuePairs"},
        5,
    ],
)
def test_map_ocr_params_invalid_features_raises(features):
    config = AzureDocumentIntelligenceOCRConfig()

    with pytest.raises(ValueError, match="Invalid `features`"):
        config.map_ocr_params({"features": features}, {}, "prebuilt-layout")


def test_get_complete_url_appends_features_query():
    config = AzureDocumentIntelligenceOCRConfig()

    url = config.get_complete_url(
        api_base="https://example.cognitiveservices.azure.com",
        model="azure_ai/doc-intelligence/prebuilt-layout",
        optional_params={"features": "keyValuePairs"},
    )

    assert "&features=keyValuePairs" in url


def test_get_complete_url_combines_pages_and_features():
    config = AzureDocumentIntelligenceOCRConfig()

    optional_params = config.map_ocr_params(
        {"pages": [0, 1, 2], "features": ["keyValuePairs", "languages"]},
        {},
        "prebuilt-layout",
    )
    url = config.get_complete_url(
        api_base="https://example.cognitiveservices.azure.com",
        model="prebuilt-layout",
        optional_params=optional_params,
    )

    assert "&pages=1,2,3" in url
    assert "&features=keyValuePairs,languages" in url
