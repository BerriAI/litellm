from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from ...shared.reporting.models import SdkFunction
from .mapping_validator import MappingSuite, TestMapping, UnitParityExclusionSpec


OCR_MAPPING: Final = MappingSuite(
    python_scope=(
        "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
        "tests/test_litellm/ocr/test_ocr_azure_document_intelligence_api_base.py",
        "tests/test_litellm/ocr/test_rust_bridge.py",
        "tests/test_litellm/ocr/test_ocr_file_input.py",
        "tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py",
        "tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_cost.py",
        "tests/test_litellm/ocr/test_ocr_native_format.py",
        "tests/test_litellm/llms/ocr/guardrail_translation/test_ocr_guardrail_handler.py",
        "tests/test_litellm/proxy/ocr_endpoints/test_endpoints.py",
    ),
    unit_parity_scope=(
        "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
        "tests/test_litellm/ocr/test_ocr_azure_document_intelligence_api_base.py",
        "tests/test_litellm/ocr/test_rust_bridge.py",
        "tests/test_litellm/ocr/test_ocr_file_input.py",
        "tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py",
        "tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_cost.py",
        "tests/test_litellm/ocr/test_ocr_native_format.py",
        "tests/test_litellm/llms/ocr/guardrail_translation/test_ocr_guardrail_handler.py",
    ),
    unit_parity_exclusions=(
        UnitParityExclusionSpec(
            nodeid="tests/test_litellm/ocr/test_rust_bridge.py::test_use_litellm_rust_toggles_flag",
            reason="This test asserts the process-level backend flag selected by the parity runner.",
        ),
    ),
    rust_scope=(
        "litellm-rust/crates/ai-gateway/src/ocr/common_utils.rs",
        "litellm-rust/crates/ai-gateway/src/ocr/tests.rs",
        "litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs",
        "litellm-rust/crates/core/src/providers/vertex_ai/ocr/transformation.rs",
        "litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs",
        "litellm-rust/crates/ai-gateway/src/integrations/custom_logger/mod.rs",
    ),
    cargo_manifest="litellm-rust/Cargo.toml",
    cargo_filter="ocr",
    mappings=(
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_transform_ocr_response_preserves_azure_native_fields",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_response_normalizes_pages",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_async_transform_ocr_response_preserves_azure_native_fields",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_response_normalizes_pages",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_features",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_url_normalizes_features",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_empty_features_list_omitted",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_url_omits_empty_feature_list",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_invalid_features_raises",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_url_rejects_invalid_features",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_get_complete_url_appends_features_query",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_url_normalizes_features",
        ),
        TestMapping(
            python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_get_complete_url_combines_pages_and_features",
            rust="litellm-rust/crates/core/src/providers/azure_ai/ocr/transformation.rs::document_intelligence_url_combines_pages_and_feature_list",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_extract_header_in_supported_params",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::extract_header_is_a_supported_ocr_param",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_extract_footer_in_supported_params",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::extract_footer_is_a_supported_ocr_param",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_existing_params_still_present",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::existing_ocr_params_remain_supported",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_header_passed_through",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::map_ocr_params_forwards_extract_header",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_footer_passed_through",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::map_ocr_params_forwards_extract_footer",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_header_and_footer_together",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::map_ocr_params_forwards_extract_header_and_footer",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_unknown_param_is_dropped",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::map_ocr_params_drops_unknown_params",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestNewSupportedParams::test_new_param_in_supported_list",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::new_ocr_params_are_supported",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestNewParamsMapOcr::test_new_param_passed_through",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::map_ocr_params_forwards_new_ocr_params",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrRequest::test_param_included_in_request_body",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::transform_ocr_request_includes_each_optional_param",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrRequest::test_multiple_new_params_together",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::transform_ocr_request_includes_multiple_new_params",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrResponseOcr4Fields::test_blocks_and_confidence_scores_preserved",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::transform_ocr_response_preserves_blocks_and_confidence_scores",
        ),
        TestMapping(
            python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrResponseOcr4Fields::test_ocr4_fields_survive_model_dump",
            rust="litellm-rust/crates/core/src/providers/mistral/ocr/transformation.rs::transform_ocr_response_preserves_ocr4_page_fields",
        ),
    ),
)

MAPPING_SUITES: Final[Mapping[SdkFunction, MappingSuite]] = MappingProxyType({"ocr": OCR_MAPPING})
