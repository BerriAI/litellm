from __future__ import annotations

from typing import Final

from ....shared.unit_runners.rust_runner import RustTarget, RustTestIdentity
from ..contracts import (
    MappingExclusionSpec,
    MappingSpec,
    PythonFunctionDiscoverySpec,
    RustTestFamily,
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
_VERTEX_OCR_TESTS: Final = "providers::vertex_ai::ocr::transformation::tests"
_REDUCTO_OCR_TESTS: Final = "providers::reducto::ocr::tests"
_GATEWAY_OCR_TESTS: Final = "ocr::tests"
_GATEWAY_PREPARE_OCR_TESTS: Final = "ocr::prepare::tests"


def _rust_test(target: RustTarget, module: str, test: str) -> RustTestIdentity:
    return RustTestIdentity(target=target, name=f"{module}::{test}")


def _rust_family(target: RustTarget, module: str, test: str) -> RustTestFamily:
    return RustTestFamily(target=target, name=f"{module}::{test}")


def _test_mappings(target: RustTarget, module: str, pairs: tuple[tuple[str, str], ...]) -> tuple[TestMapping, ...]:
    return tuple(TestMapping(python=python, rust=_rust_test(target, module, test)) for python, test in pairs)


_AZURE_TRANSFORM_FILE: Final = "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py"
_AZURE_PAGES_FILE: Final = "tests/ocr_tests/test_ocr_azure_document_intelligence.py"
_AZURE_BASE_FILE: Final = "tests/test_litellm/ocr/test_ocr_azure_document_intelligence_api_base.py"
_RUST_BRIDGE_FILE: Final = "tests/test_litellm/ocr/test_rust_bridge.py"

_AZURE_PORT_MAPPINGS: Final = _test_mappings(
    _CORE_TARGET,
    _AZURE_OCR_TESTS,
    (
        (
            f"{_AZURE_TRANSFORM_FILE}::test_should_encode_azure_document_intelligence_model_id",
            "azure_document_intelligence_model_id_is_encoded",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_should_reject_dot_segment_azure_document_intelligence_model_id",
            "azure_document_intelligence_dot_segment_model_id_is_rejected",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_async_transform_ocr_response_preserves_azure_native_fields",
            "document_intelligence_async_response_preserves_normalized_fields",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_transform_ocr_response_tolerates_missing_native_fields",
            "document_intelligence_response_tolerates_missing_native_fields",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_transform_ocr_response_non_succeeded_status_raises",
            "document_intelligence_non_succeeded_status_is_rejected",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_get_supported_ocr_params_includes_features",
            "document_intelligence_supported_params_include_features",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_transform_ocr_response_native_format_carries_raw_operation",
            "document_intelligence_native_format_carries_raw_operation",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_async_transform_ocr_response_native_format_carries_raw_operation",
            "document_intelligence_async_native_format_carries_raw_operation",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_map_ocr_params_rejects_unknown_req_format_as_bad_request",
            "document_intelligence_rejects_unknown_req_format",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_get_complete_url_omits_req_format_query_param",
            "document_intelligence_url_omits_req_format",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_validate_environment_uses_subscription_key",
            "document_intelligence_validate_environment_uses_subscription_key",
        ),
        (
            f"{_AZURE_TRANSFORM_FILE}::test_validate_environment_falls_back_to_entra_token",
            "document_intelligence_validate_environment_falls_back_to_entra_token",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_get_supported_ocr_params_includes_pages_and_features",
            "document_intelligence_supported_params_include_pages_features_and_req_format",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_mistral_zero_based_int_list",
            "document_intelligence_maps_zero_based_page_list",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_dedupes_and_sorts",
            "document_intelligence_page_mapping_dedupes_and_sorts",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_empty_list_omits_pages",
            "document_intelligence_page_mapping_omits_empty_list",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_azure_native_string_range",
            "document_intelligence_page_mapping_accepts_native_range",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_azure_native_string_with_spaces_stripped",
            "document_intelligence_page_mapping_strips_spaces",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_list_of_string_tokens",
            "document_intelligence_page_mapping_accepts_string_tokens",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_invalid_string_raises",
            "document_intelligence_page_mapping_rejects_invalid_string",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_negative_index_raises",
            "document_intelligence_page_mapping_rejects_negative_index",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_bool_list_raises",
            "document_intelligence_page_mapping_rejects_bool_list",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_map_ocr_params_unsupported_type_raises",
            "document_intelligence_page_mapping_rejects_unsupported_type",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_get_complete_url_appends_pages_query",
            "document_intelligence_url_appends_pages_query",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_get_complete_url_no_pages_when_optional_params_empty",
            "document_intelligence_url_has_no_pages_when_params_are_empty",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_transform_ocr_request_does_not_put_pages_in_body",
            "document_intelligence_request_keeps_pages_out_of_body",
        ),
        (
            f"{_AZURE_PAGES_FILE}::TestAzureDocumentIntelligencePagesParam::test_end_to_end_mistral_shape_to_azure_query",
            "document_intelligence_mistral_pages_flow_to_query_only",
        ),
        (
            "tests/test_litellm/llms/azure_ai/test_azure_ai_entra_auth.py::test_ocr_authenticates_with_entra_token",
            "azure_ai_ocr_authenticates_with_entra_token",
        ),
        (
            f"{_AZURE_BASE_FILE}::TestDocIntelligenceApiBaseResolution::test_generic_azure_ai_base_does_not_hijack_doc_intelligence",
            "document_intelligence_endpoint_ignores_generic_azure_ai_base",
        ),
        (
            f"{_AZURE_BASE_FILE}::TestDocIntelligenceApiBaseResolution::test_explicit_api_base_is_honoured_for_doc_intelligence",
            "document_intelligence_endpoint_honors_explicit_api_base",
        ),
        (
            f"{_AZURE_BASE_FILE}::TestDocIntelligenceApiBaseResolution::test_generic_azure_ai_base_still_applies_to_mistral_ocr",
            "azure_ai_mistral_ocr_uses_generic_api_base",
        ),
    ),
)

_REDUCTO_PORT_MAPPINGS: Final = _test_mappings(
    _CORE_TARGET,
    _REDUCTO_OCR_TESTS,
    (
        (
            "tests/test_litellm/llms/reducto/test_parse_v3.py::test_parse_v3_reducto_id_passthrough_skips_upload",
            "test_parse_v3_reducto_id_passthrough_skips_upload",
        ),
        (
            "tests/test_litellm/llms/reducto/test_parse_legacy.py::test_parse_legacy_wraps_enhance_under_options",
            "test_parse_legacy_wraps_enhance_under_options",
        ),
        (
            "tests/test_litellm/llms/reducto/test_upload.py::test_parse_v3_image_data_uri_upload_uses_image_mime",
            "test_parse_v3_image_data_uri_upload_uses_image_mime",
        ),
        (
            "tests/test_litellm/llms/reducto/test_upload.py::test_parse_v3_uses_programmatic_api_key_over_env",
            "test_parse_v3_uses_programmatic_api_key_over_env",
        ),
    ),
)

_REDUCTO_GATEWAY_MAPPING: Final = TestMapping(
    python="tests/test_litellm/llms/reducto/test_parse_v3.py::test_parse_v3_file_upload_and_response_mapping",
    rust=_rust_test(_GATEWAY_TARGET, _GATEWAY_OCR_TESTS, "reducto_file_upload_then_parse_maps_response"),
)

_GATEWAY_PORT_MAPPINGS: Final = _test_mappings(
    _GATEWAY_TARGET,
    _GATEWAY_PREPARE_OCR_TESTS,
    (
        (
            "tests/test_litellm/ocr/test_ocr_native_format.py::test_native_format_rejected_for_provider_without_support_as_bad_request",
            "native_format_rejected_for_provider_without_support_as_bad_request",
        ),
        (
            "tests/test_litellm/ocr/test_ocr_native_format.py::test_unknown_format_rejected_for_provider_without_support_as_bad_request",
            "unknown_format_rejected_for_provider_without_support_as_bad_request",
        ),
    ),
)

_HOST_ONLY_BRIDGE_EXCLUSIONS: Final = tuple(
    MappingExclusionSpec(nodeid=f"{_RUST_BRIDGE_FILE}::{test}", reason=reason)
    for test, reason in (
        ("test_ocr_routes_to_rust_when_enabled", "Python selects and invokes the native bridge."),
        ("test_ocr_routes_azure_ai_to_rust_when_enabled", "Python resolves provider arguments before the bridge."),
        ("test_ocr_rust_path_converts_file_document_before_bridge", "Python converts file inputs before the bridge."),
        (
            "test_ocr_exception_type_uses_resolved_provider_context",
            "Python wraps bridge exceptions into public errors.",
        ),
        ("test_aocr_routes_to_async_rust_when_enabled", "Python selects and invokes the async native bridge."),
        ("test_aocr_exception_type_uses_resolved_provider_context", "Python wraps async bridge exceptions."),
        ("test_ocr_forwards_timeout_to_rust", "Python converts and forwards explicit timeouts."),
        ("test_ocr_passes_default_request_timeout_to_rust", "Python supplies its process-level default timeout."),
        ("test_ocr_falls_back_to_python_when_bridge_unavailable", "Python owns fallback when the extension is absent."),
    )
)

_FAMILY_PORT_MAPPINGS: Final = (
    TestMapping(
        python=f"{_AZURE_TRANSFORM_FILE}::test_transform_ocr_response_default_format_omits_raw_operation",
        rust=_rust_family(
            _CORE_TARGET,
            _AZURE_OCR_TESTS,
            "document_intelligence_default_format_omits_raw_operation",
        ),
    ),
    TestMapping(
        python=f"{_AZURE_TRANSFORM_FILE}::test_map_ocr_params_passes_through_req_format",
        rust=_rust_family(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_maps_req_format"),
    ),
    TestMapping(
        python="tests/ocr_tests/test_ocr_vertex_ai.py::test_deepseek_request_uses_single_provider_namespace",
        rust=_rust_family(
            _CORE_TARGET,
            _VERTEX_OCR_TESTS,
            "vertex_deepseek_request_uses_single_provider_namespace",
        ),
    ),
    TestMapping(
        python="tests/test_litellm/llms/reducto/test_upload.py::test_parse_v3_rejects_plain_http_urls",
        rust=_rust_family(_CORE_TARGET, _REDUCTO_OCR_TESTS, "test_parse_v3_rejects_plain_http_urls"),
    ),
)


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
                rust=_rust_family(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_maps_features"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_empty_features_list_omitted",
                rust=_rust_test(_CORE_TARGET, _AZURE_OCR_TESTS, "document_intelligence_url_omits_empty_feature_list"),
            ),
            TestMapping(
                python="tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py::test_map_ocr_params_invalid_features_raises",
                rust=_rust_family(
                    _CORE_TARGET,
                    _AZURE_OCR_TESTS,
                    "document_intelligence_mapping_rejects_invalid_features",
                ),
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
            *_AZURE_PORT_MAPPINGS,
            *_REDUCTO_PORT_MAPPINGS,
            _REDUCTO_GATEWAY_MAPPING,
            *_GATEWAY_PORT_MAPPINGS,
            *_FAMILY_PORT_MAPPINGS,
        ),
        exclusions=_HOST_ONLY_BRIDGE_EXCLUSIONS,
        require_complete=True,
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
                nodeid="tests/test_litellm/ocr/test_rust_bridge.py::test_rust_toggles_flag",
                reason="This test asserts the process-level backend flag selected by the parity runner.",
            ),
        ),
    ),
    rust=RustUnitSpec(
        cargo_manifest="litellm-rust/Cargo.toml",
        cargo_filter="ocr",
    ),
)
