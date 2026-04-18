"""
Test OCR functionality with Azure Document Intelligence API.

Azure Document Intelligence provides advanced document analysis capabilities
using the v4.0 (2024-11-30) API.
"""

import os

import pytest

from base_ocr_unit_tests import BaseOCRTest
from litellm.constants import AZURE_DOCUMENT_INTELLIGENCE_API_VERSION
from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
    AzureDocumentIntelligenceOCRConfig,
)


class TestAzureDocumentIntelligenceOCR(BaseOCRTest):
    """
    Test class for Azure Document Intelligence OCR functionality.

    Inherits from BaseOCRTest and provides Azure Document Intelligence-specific configuration.

    Tests the azure_ai/doc-intelligence/<model> provider route.
    """

    def get_base_ocr_call_args(self) -> dict:
        """
        Return the base OCR call args for Azure Document Intelligence.

        Uses prebuilt-layout model which is closest to Mistral OCR format.
        """
        # Check for required environment variables
        api_key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if not api_key or not endpoint:
            pytest.skip(
                "AZURE_DOCUMENT_INTELLIGENCE_API_KEY and AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "environment variables are required for Azure Document Intelligence tests"
            )

        return {
            "model": "azure_ai/doc-intelligence/prebuilt-layout",
            "api_key": api_key,
            "api_base": endpoint,
        }


class TestAzureDocumentIntelligencePagesParam:
    """
    Unit tests for the Mistral-compatible `pages` parameter translation to
    Azure Document Intelligence's `pages` query string.

    These tests exercise the transformation layer directly and do not
    require Azure credentials or a network call.
    """

    @pytest.fixture
    def cfg(self) -> AzureDocumentIntelligenceOCRConfig:
        return AzureDocumentIntelligenceOCRConfig()

    def test_get_supported_ocr_params_includes_pages(self, cfg):
        assert cfg.get_supported_ocr_params("prebuilt-layout") == ["pages"]

    def test_map_ocr_params_mistral_zero_based_int_list(self, cfg):
        mapped = cfg.map_ocr_params({"pages": [0, 1, 2]}, {}, "prebuilt-layout")
        assert mapped == {"pages": "1,2,3"}

    def test_map_ocr_params_dedupes_and_sorts(self, cfg):
        mapped = cfg.map_ocr_params({"pages": [2, 0, 0, 1]}, {}, "prebuilt-layout")
        assert mapped == {"pages": "1,2,3"}

    def test_map_ocr_params_empty_list_omits_pages(self, cfg):
        mapped = cfg.map_ocr_params({"pages": []}, {}, "prebuilt-layout")
        assert mapped == {}

    def test_map_ocr_params_azure_native_string_range(self, cfg):
        mapped = cfg.map_ocr_params({"pages": "3-9"}, {}, "prebuilt-layout")
        assert mapped == {"pages": "3-9"}

    def test_map_ocr_params_azure_native_string_with_spaces_stripped(self, cfg):
        mapped = cfg.map_ocr_params({"pages": "1-3, 5"}, {}, "prebuilt-layout")
        assert mapped == {"pages": "1-3,5"}

    def test_map_ocr_params_list_of_string_tokens(self, cfg):
        mapped = cfg.map_ocr_params({"pages": ["1", "3-5"]}, {}, "prebuilt-layout")
        assert mapped == {"pages": "1,3-5"}

    def test_map_ocr_params_invalid_string_raises(self, cfg):
        with pytest.raises(ValueError, match="Invalid `pages` string"):
            cfg.map_ocr_params({"pages": "a,b"}, {}, "prebuilt-layout")

    def test_map_ocr_params_negative_index_raises(self, cfg):
        with pytest.raises(ValueError, match="must be >= 0"):
            cfg.map_ocr_params({"pages": [-1]}, {}, "prebuilt-layout")

    def test_map_ocr_params_bool_list_raises(self, cfg):
        with pytest.raises(ValueError, match="must be integers, not booleans"):
            cfg.map_ocr_params({"pages": [True, False]}, {}, "prebuilt-layout")

    def test_map_ocr_params_unsupported_type_raises(self, cfg):
        with pytest.raises(ValueError):
            cfg.map_ocr_params({"pages": 5}, {}, "prebuilt-layout")

    def test_get_complete_url_appends_pages_query(self, cfg):
        url = cfg.get_complete_url(
            api_base="https://example.cognitiveservices.azure.com/",
            model="azure_ai/doc-intelligence/prebuilt-layout",
            optional_params={"pages": "1-3,5"},
        )
        assert (
            f"api-version={AZURE_DOCUMENT_INTELLIGENCE_API_VERSION}" in url
        ), url
        assert "pages=1-3,5" in url, url
        assert "/documentintelligence/documentModels/prebuilt-layout:analyze" in url

    def test_get_complete_url_no_pages_when_optional_params_empty(self, cfg):
        url = cfg.get_complete_url(
            api_base="https://example.cognitiveservices.azure.com",
            model="prebuilt-layout",
            optional_params={},
        )
        assert "pages=" not in url

    def test_transform_ocr_request_does_not_put_pages_in_body(self, cfg):
        req = cfg.transform_ocr_request(
            model="prebuilt-layout",
            document={
                "type": "document_url",
                "document_url": "https://example.com/x.pdf",
            },
            optional_params={"pages": "1,2,3"},
            headers={},
        )
        assert req.data is not None
        assert "pages" not in req.data
        assert req.data.get("urlSource") == "https://example.com/x.pdf"

    def test_end_to_end_mistral_shape_to_azure_query(self, cfg):
        """
        Caller sends Mistral-style `pages: [2,3,4,5,6,7,8]` (0-based,
        meaning human pages 3-9). LiteLLM should turn that into Azure's
        `&pages=3,4,5,6,7,8,9` on the analyze URL, and the body should
        still only contain urlSource.
        """
        non_default_params = {"pages": [2, 3, 4, 5, 6, 7, 8]}
        optional_params = cfg.map_ocr_params(
            non_default_params=non_default_params,
            optional_params={},
            model="prebuilt-layout",
        )
        url = cfg.get_complete_url(
            api_base="https://example.cognitiveservices.azure.com",
            model="prebuilt-layout",
            optional_params=optional_params,
        )
        req = cfg.transform_ocr_request(
            model="prebuilt-layout",
            document={
                "type": "document_url",
                "document_url": "https://example.com/x.pdf",
            },
            optional_params=optional_params,
            headers={},
        )

        assert "pages=3,4,5,6,7,8,9" in url
        assert req.data == {"urlSource": "https://example.com/x.pdf"}

