mod cache;
mod callable;
mod coercion;
mod credentials;
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
        CatalogRefresher, FetchedRegistry, InvalidProvider, ModelCostLayers, ModelInfoView,
        ModelNotFound, ModelRegistry, PriceLine, Priced, RegistrationTarget, RegistryNotModified,
        RegistryUnavailable, ResolvedModel, RowError, RowErrors, Unpriced, UsageInput, WhenInput,
        fetch_once, fetch_with_retry, invalidate_model_cost_index, model_info, model_row_errors,
        price, registration_target, resolve_model, write_rows,
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
        crate::model_registry::init_model_key_index(module)?;
        let py = module.py();
        let dict = module.dict();
        dict.set_item(
            "NativeCacheHandle",
            py.get_type::<crate::cache::NativeCacheHandle>(),
        )?;
        dict.set_item(
            "_SecretManagerRuntime",
            py.get_type::<crate::secrets::runtime::NativeSecretManager>(),
        )?;
        let typing = py.import("typing")?;
        let literal = typing.getattr("Literal")?;
        let kinds = pyo3::types::PyTuple::new(
            py,
            <crate::model_registry::RowErrorKind as strum::VariantArray>::VARIANTS
                .iter()
                .map(|kind| literal.get_item(<&'static str>::from(*kind)))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        module.add("RowErrorKind", typing.getattr("Union")?.get_item(kinds)?)
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
    fn registry_rows_install_without_replacing_the_live_dict() {
        Python::initialize();
        Python::attach(|py| {
            let scope = pyo3::types::PyDict::new(py);
            scope.set_item("native", native_module(py)).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
registry = native.ModelRegistry.parse(b'{"model": {"litellm_provider": "openai", "input_cost_per_token": 0}}')
live = {"old": {}}
alias = live
assert native.write_rows(live, registry.served(), ["old", "missing"]) is None
assert live is alias
assert live == {"model": {"litellm_provider": "openai", "input_cost_per_token": 0}}
"#), Some(&scope), Some(&scope)).unwrap();
        });
    }

    #[rstest::rstest]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let mut expected = vec![
                "NativeCacheHandle",
                "CatalogRefresher",
                "FetchedRegistry",
                "RegistryNotModified",
                "RegistryUnavailable",
                "fetch_once",
                "fetch_with_retry",
                "ModelRegistry",
                "ModelCostLayers",
                "RowError",
                "RowErrorKind",
                "model_row_errors",
                "write_rows",
                "ResolvedModel",
                "ModelNotFound",
                "InvalidProvider",
                "RegistrationTarget",
                "RowErrors",
                "ModelInfoView",
                "UsageInput",
                "WhenInput",
                "PriceLine",
                "Priced",
                "Unpriced",
                "resolve_model",
                "registration_target",
                "model_info",
                "price",
                "invalidate_model_cost_index",
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
                "trace_encode_error",
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
