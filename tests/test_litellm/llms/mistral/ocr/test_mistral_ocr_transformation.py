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
            assert (
                param in supported
            ), f"Previously supported param '{param}' is missing"


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


class TestNewSupportedParams:
    """Verify the newly added params are in the supported list."""

    @pytest.mark.parametrize(
        "param_name",
        [
            "table_format",
            "confidence_scores_granularity",
            "document_annotation_prompt",
            "id",
        ],
    )
    def test_new_param_in_supported_list(
        self, config: MistralOCRConfig, param_name: str
    ) -> None:
        supported = config.get_supported_ocr_params(model=MODEL)
        assert param_name in supported


class TestNewParamsMapOcr:
    """Verify the newly added params survive map_ocr_params."""

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("table_format", "html"),
            ("table_format", "markdown"),
            ("confidence_scores_granularity", "word"),
            ("confidence_scores_granularity", "page"),
            ("document_annotation_prompt", "Extract all invoice line items"),
            ("id", "req-123"),
        ],
    )
    def test_new_param_passed_through(
        self, config: MistralOCRConfig, param_name: str, param_value: str
    ) -> None:
        result = config.map_ocr_params(
            non_default_params={param_name: param_value},
            optional_params={},
            model=MODEL,
        )
        assert result == {param_name: param_value}


class TestTransformOcrRequest:
    """Verify params end up in the final request body via transform_ocr_request."""

    SAMPLE_DOCUMENT = {
        "type": "document_url",
        "document_url": "https://example.com/doc.pdf",
    }

    @pytest.mark.parametrize(
        "param_name,param_value",
        [
            ("table_format", "html"),
            ("confidence_scores_granularity", "word"),
            ("document_annotation_prompt", "Extract all invoice line items"),
            ("id", "req-123"),
            ("extract_header", True),
            ("pages", [0, 1]),
        ],
    )
    def test_param_included_in_request_body(
        self, config: MistralOCRConfig, param_name: str, param_value
    ) -> None:
        result = config.transform_ocr_request(
            model=MODEL,
            document=self.SAMPLE_DOCUMENT,
            optional_params={param_name: param_value},
            headers={},
        )
        assert result.data[param_name] == param_value
        assert result.data["model"] == MODEL
        assert result.data["document"] == self.SAMPLE_DOCUMENT
        assert result.files is None

    def test_multiple_new_params_together(self, config: MistralOCRConfig) -> None:
        """Multiple new params can be passed together in a single request."""
        optional_params = {
            "table_format": "html",
            "confidence_scores_granularity": "page",
            "extract_header": True,
        }
        result = config.transform_ocr_request(
            model=MODEL,
            document=self.SAMPLE_DOCUMENT,
            optional_params=optional_params,
            headers={},
        )
        for key, value in optional_params.items():
            assert result.data[key] == value
