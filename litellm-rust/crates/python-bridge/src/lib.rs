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
    use pyo3::{
        prelude::*,
        types::{PyModule, PyTuple},
    };

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
    use crate::model_registry::bundled_model_cost_map;
    #[pymodule_export]
    use crate::model_registry::{CatalogHandle, Installed, Parsed, RowError, Unavailable};
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
            "NativeCacheHandle",
            py.get_type::<crate::cache::NativeCacheHandle>(),
        )?;
        dict.set_item(
            "_SecretManagerRuntime",
            py.get_type::<crate::secrets::runtime::NativeSecretManager>(),
        )?;
        let typing = py.import("typing")?;
        let literal = typing.getattr("Literal")?;
        let row_error_kinds = PyTuple::new(
            py,
            <crate::model_registry::RowErrorKind as strum::VariantArray>::VARIANTS
                .iter()
                .map(|kind| literal.get_item(<&'static str>::from(*kind)))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let row_error_kind = typing.getattr("Union")?.get_item(row_error_kinds)?;
        module.add("RowErrorKind", row_error_kind)
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

    fn singleton_literal_values(py: Python<'_>, alias: &Bound<'_, PyAny>) -> (bool, Vec<String>) {
        let typing = py.import("typing").unwrap();
        let literal = typing.getattr("Literal").unwrap();
        let components = typing.call_method1("get_args", (alias,)).unwrap();
        let inspected = components
            .try_iter()
            .unwrap()
            .map(|component| {
                let component = component.unwrap();
                let origin = typing.call_method1("get_origin", (&component,)).unwrap();
                let arguments = typing.call_method1("get_args", (&component,)).unwrap();
                let values = arguments.extract::<Vec<String>>().unwrap();
                (
                    origin.is(&literal) && values.len() == 1,
                    values.first().cloned(),
                )
            })
            .collect::<Vec<_>>();
        (
            inspected
                .iter()
                .all(|(is_singleton_literal, _)| *is_singleton_literal),
            inspected
                .into_iter()
                .filter_map(|(_, value)| value)
                .collect(),
        )
    }

    #[rstest::rstest]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let mut expected = vec![
                "NativeCacheHandle",
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
                "Catalog",
                "Parsed",
                "Installed",
                "Unavailable",
                "RowError",
                "RowErrorKind",
                "NativeClickHouseSpendConfig",
                "NativeClickHouseSpendStorage",
                "trace_encode_error",
                "bundled_model_cost_map",
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

    #[rstest::rstest]
    fn row_error_kind_is_a_union_of_singleton_literals() {
        Python::initialize();
        Python::attach(|py| {
            let module = native_module(py);
            let typing = py.import("typing").unwrap();
            let union = typing.getattr("Union").unwrap();
            let alias = module.getattr("RowErrorKind").unwrap();
            let origin = typing.call_method1("get_origin", (alias.clone(),)).unwrap();
            assert!(origin.is(&union));
            let (all_singleton_literals, actual) = singleton_literal_values(py, &alias);
            assert!(all_singleton_literals);
            assert_eq!(
                actual,
                Vec::from([
                    "json".to_owned(),
                    "empty".to_owned(),
                    "below_minimum".to_owned(),
                    "shrunk".to_owned(),
                    "not_an_object".to_owned(),
                    "invalid".to_owned(),
                    "unknown_block_key".to_owned(),
                    "missing_provider".to_owned(),
                    "unknown_timezone".to_owned(),
                    "alias_shadows_model".to_owned(),
                    "alias_taken".to_owned(),
                    "duplicate_price".to_owned(),
                    "suffix_inside_block".to_owned(),
                    "tier_range".to_owned(),
                    "tier_reversed".to_owned(),
                    "tier_gap".to_owned(),
                    "mixed_tier_basis".to_owned(),
                    "not_priced_by_results".to_owned(),
                    "bands_from_two_sources".to_owned(),
                    "not_an_off_peak_rate".to_owned(),
                    "no_off_peak_prices".to_owned(),
                    "no_off_peak_hours".to_owned(),
                    "empty_list".to_owned(),
                    "invalid_date".to_owned(),
                ])
            );
        });
    }
}
