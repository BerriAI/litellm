from __future__ import annotations

from typing import Final

from ....shared.unit_runners.rust_runner import RustTarget, RustTestIdentity
from ..contracts import (
    MappingSpec,
    PythonFunctionDiscoverySpec,
    RustUnitSpec,
    TestMapping,
    UnitParityExclusionSpec,
    UnitParitySpec,
    UnitTestContract,
)

_CORE_TARGET: Final = RustTarget(package="litellm-core", name="litellm_core", kind="lib")
_GATEWAY_TARGET: Final = RustTarget(
    package="litellm-ai-gateway",
    name="litellm_ai_gateway",
    kind="lib",
)
_AZURE_OCR_TESTS: Final = "providers::azure_ai::ocr::transformation::tests"
_MISTRAL_OCR_TESTS: Final = "providers::mistral::ocr::transformation::tests"


def _rust_test(target: RustTarget, module: str, test: str) -> RustTestIdentity:
    return RustTestIdentity(target=target, name=f"{module}::{test}")


OCR_CONTRACT: Final = UnitTestContract(
    mapping=MappingSpec(
        python_functions=PythonFunctionDiscoverySpec(
            trace_module="tests.rust-python-harness.strategies.trace_parity.sdk.ocr.case",
            trace_spans=(
                "ocr",
                "prepare_ocr_call",
                "ocr_provider_config",
                "supported_ocr_params",
                "map_ocr_params",
                "validate_environment",
                "complete_url",
                "transform_ocr_request",
                "execute_ocr_provider_call",
                "transform_ocr_response",
                "poll_document_intelligence",
            ),
            search_roots=("tests",),
            exclude_roots=(
                "tests/e2e",
                "tests/ocr_tests/test_ocr_mistral.py",
                "tests/rust-python-harness",
            ),
            includes=(
                "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
                "tests/test_litellm/llms/mistral/ocr",
                "tests/test_litellm/llms/ocr",
                "tests/test_litellm/ocr",
                "tests/test_litellm/proxy/ocr_endpoints",
            ),
            exclusions=(
                "tests/ocr_tests/test_ocr_azure_document_intelligence.py::TestAzureDocumentIntelligenceOCR",
                "tests/ocr_tests/test_ocr_vertex_ai.py::TestVertexAIMistralOCR",
                "tests/ocr_tests/test_ocr_vertex_ai.py::TestVertexAIDeepSeekOCR",
            ),
        ),
        rust_targets=(_CORE_TARGET, _GATEWAY_TARGET),
        mappings=(
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_transform_ocr_response_preserves_azure_native_fields",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_response_normalizes_pages"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_features",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_maps_features"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_empty_features_list_omitted",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_url_omits_empty_feature_list"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_invalid_features_raises",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_url_rejects_invalid_features"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_get_complete_url_appends_features_query",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_url_normalizes_features"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_get_complete_url_combines_pages_and_features",
                rust=_rust_test(
                    _CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_url_combines_pages_and_feature_list"
                ),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_extract_header_in_supported_params",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "extract_header_is_a_supported_ocr_param"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_extract_footer_in_supported_params",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "extract_footer_is_a_supported_ocr_param"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestGetSupportedOcrParams::test_existing_params_still_present",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "existing_ocr_params_remain_supported"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_header_passed_through",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "map_ocr_params_forwards_extract_header"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_footer_passed_through",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "map_ocr_params_forwards_extract_footer"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_extract_header_and_footer_together",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "map_ocr_params_forwards_extract_header_and_footer"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestMapOcrParams::test_unknown_param_is_dropped",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "map_ocr_params_drops_unknown_params"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestNewSupportedParams::test_new_param_in_supported_list",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "new_ocr_params_are_supported"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestNewParamsMapOcr::test_new_param_passed_through",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "map_ocr_params_forwards_new_ocr_params"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrRequest::test_param_included_in_request_body",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "transform_ocr_request_includes_each_optional_param"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrRequest::test_multiple_new_params_together",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "transform_ocr_request_includes_multiple_new_params"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrResponseOcr4Fields::test_blocks_and_confidence_scores_preserved",
                rust=_rust_test(
                    _CORE_TARGET, _MISTRAL_OCR_TESTS, "transform_ocr_response_preserves_blocks_and_confidence_scores"
                ),
            ),
            TestMapping(
                python="tests/test_litellm/llms/mistral/ocr/test_mistral_ocr_transformation.py::TestTransformOcrResponseOcr4Fields::test_ocr4_fields_survive_model_dump",
                rust=_rust_test(_CORE_TARGET, _MISTRAL_OCR_TESTS, "transform_ocr_response_preserves_ocr4_page_fields"),
            ),
        ),
    ),
    unit_parity=UnitParitySpec(
        python_selectors=(
            "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
            "tests/test_litellm/llms/mistral/ocr",
            "tests/test_litellm/llms/ocr",
            "tests/test_litellm/ocr",
        ),
        exclusions=(
            UnitParityExclusionSpec(
                nodeid="tests/test_litellm/ocr/test_rust_bridge.py::test_use_litellm_rust_toggles_flag",
                reason="This test asserts the process-level backend flag selected by the parity runner.",
            ),
        ),
    ),
    rust=RustUnitSpec(
        cargo_manifest="litellm-rust/Cargo.toml",
        cargo_filter="ocr",
    ),
)
