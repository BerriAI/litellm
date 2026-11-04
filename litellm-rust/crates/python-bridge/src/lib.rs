mod cache;
mod callable;
mod coercion;
mod credentials;
mod db;
mod diagnostics;
mod errors;
mod execution;
mod http;
mod logger;
mod marshal;
mod model_registry;
mod preflight;
mod python_settings;
mod routes;
mod secrets;
mod tokenizer;

#[pymodule(gil_used = true)]
mod _native {
    #[pymodule_export]
    use litellm_host_python::{ForkedAfterNativeRuntimeStarted, ProcessReservedForForking};
    use pyo3::{prelude::*, types::PyModule};

    #[pymodule_export]
    use crate::db::NativeDatabase;
    #[cfg(feature = "panic-test")]
    #[pymodule_export]
    use crate::diagnostics::_panic_for_test;
    #[pymodule_export]
    use crate::diagnostics::{gil_stats, process_state_started, reserve_process_for_forking};
    #[pymodule_export]
    use crate::errors::{RustBridgeDeclined, RustUpstreamError};
    #[pymodule_export]
    use crate::logger::NativeDiagnosticProcessor;
    #[pymodule_export]
    use crate::model_registry::{
        AliasWarning, ModelCostMapFetch, ModelCountProblem, PyProvenance, PyRefresh,
        bundled_model_cost_map, cost_map_omits_token_price, count_model_entries,
        expand_model_aliases, find_model_cost_key, git_blob_id, install_model_cost_map_provenance,
        invalidate_model_cost_keys, live_row_or, model_cost_map_provenance, model_count_problem,
        registration_row, stringified_number, take_fallback_generalization_rules,
        update_dictionary, with_pricing_overrides, without_overridden_price_variants,
        without_synthesized_costs,
    };
    #[pymodule_export]
    use crate::routes::audio_transcription::{atranscription, transcription};
    #[pymodule_export]
    use crate::routes::chat_completions::{acompletion, completion};
    #[pymodule_export]
    use crate::routes::clickhouse_spend::{
        NativeClickHouseSpendConfig, NativeClickHouseSpendStorage,
    };
    #[pymodule_export]
    use crate::routes::embeddings::{aembedding, embedding};
    #[pymodule_export]
    use crate::routes::messages::{amessages, messages};
    #[pymodule_export]
    use crate::routes::ocr::{aocr, ocr, ocr_health_check_document, ocr_passthrough_response};
    #[pymodule_export]
    use crate::routes::responses::{ResponsesWebSocketConnection, aresponses, responses};
    #[pymodule_export]
    use crate::routes::token_counter::TokenCounter;
    #[pymodule_export]
    use crate::routes::traces::trace_encode_error;
    #[cfg(feature = "huggingface")]
    #[pymodule_export]
    use crate::tokenizer::HuggingFaceEncoding;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = module.py();
        let dict = module.dict();
        dict.set_item(
            "_SecretManagerRuntime",
            py.get_type::<crate::secrets::runtime::NativeSecretManager>(),
        )
    }
}

use pyo3::prelude::*;

#[cfg(test)]
pub(crate) fn native_module(py: Python<'_>) -> Bound<'_, PyModule> {
    pyo3::wrap_pymodule!(_native)(py).into_bound(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let mut expected = vec![
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ForkedAfterNativeRuntimeStarted",
                "ProcessReservedForForking",
                "ocr",
                "aocr",
                "ocr_health_check_document",
                "ocr_passthrough_response",
                "embedding",
                "aembedding",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "completion",
                "acompletion",
                "responses",
                "aresponses",
                "ResponsesWebSocketConnection",
                "NativeDiagnosticProcessor",
                "NativeClickHouseSpendConfig",
                "NativeClickHouseSpendStorage",
                "NativeDatabase",
                "trace_encode_error",
                "bundled_model_cost_map",
                "count_model_entries",
                "model_count_problem",
                "ModelCountProblem",
                "expand_model_aliases",
                "AliasWarning",
                "git_blob_id",
                "ModelCostMapFetch",
                "ModelCostMapProvenance",
                "ModelCostMapRefresh",
                "find_model_cost_key",
                "cost_map_omits_token_price",
                "live_row_or",
                "invalidate_model_cost_keys",
                "install_model_cost_map_provenance",
                "model_cost_map_provenance",
                "registration_row",
                "stringified_number",
                "take_fallback_generalization_rules",
                "update_dictionary",
                "with_pricing_overrides",
                "without_overridden_price_variants",
                "without_synthesized_costs",
                "TokenCounter",
                "Tokenizer",
                "gil_stats",
                "process_state_started",
                "reserve_process_for_forking",
            ];
            #[cfg(feature = "huggingface")]
            expected.push("HuggingFaceEncoding");
            expected.sort_unstable();

            let mut public_names: Vec<String> = native_module(py)
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            public_names.sort_unstable();
            assert_eq!(public_names, expected);
        });
    }
}
