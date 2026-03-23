"""
Unit tests for MistralOCRConfig transformation.

Tests the supported OCR parameters and their mapping behaviour.
No real API calls are made — all tests are fully mocked/local.
"""
import pytest

from litellm.llms.mistral.ocr.transformation import MistralOCRConfig


@pytest.fixture
def config() -> MistralOCRConfig:
    return MistralOCRConfig()


MODEL = "mistral-ocr-latest"


class TestGetSupportedOcrParams:
    def test_extract_header_in_supported_params(self, config: MistralOCRConfig) -> None:
        """extract_header must be in the Mistral OCR supported params list."""
        supported = config.get_supported_ocr_params(model=MODEL)
        assert "extract_header" in supported

    def test_extract_footer_in_supported_params(self, config: MistralOCRConfig) -> None:
        """extract_footer must be in the Mistral OCR supported params list."""
        supported = config.get_supported_ocr_params(model=MODEL)
        assert "extract_footer" in supported

    def test_existing_params_still_present(self, config: MistralOCRConfig) -> None:
        """Ensure the previously supported params were not accidentally removed."""
        supported = config.get_supported_ocr_params(model=MODEL)
        for param in [
            "pages",
            "include_image_base64",
            "image_limit",
            "image_min_size",
            "bbox_annotation_format",
            "document_annotation_format",
        ]:
            assert param in supported, f"Previously supported param '{param}' is missing"


class TestMapOcrParams:
    def test_extract_header_passed_through(self, config: MistralOCRConfig) -> None:
        """extract_header=True must survive the map_ocr_params filter."""
        result = config.map_ocr_params(
            non_default_params={"extract_header": True},
            optional_params={},
            model=MODEL,
        )
        assert result == {"extract_header": True}

    def test_extract_footer_passed_through(self, config: MistralOCRConfig) -> None:
        """extract_footer=True must survive the map_ocr_params filter."""
        result = config.map_ocr_params(
            non_default_params={"extract_footer": True},
            optional_params={},
            model=MODEL,
        )
        assert result == {"extract_footer": True}

    def test_extract_header_and_footer_together(self, config: MistralOCRConfig) -> None:
        """Both params can be passed together and are both forwarded."""
        result = config.map_ocr_params(
            non_default_params={"extract_header": True, "extract_footer": False},
            optional_params={},
            model=MODEL,
        )
        assert result == {"extract_header": True, "extract_footer": False}

    def test_unknown_param_is_dropped(self, config: MistralOCRConfig) -> None:
        """Parameters not in the supported list must be silently dropped."""
        result = config.map_ocr_params(
            non_default_params={"extract_header": True, "unsupported_param": "value"},
            optional_params={},
            model=MODEL,
        )
        assert "extract_header" in result
        assert "unsupported_param" not in result
