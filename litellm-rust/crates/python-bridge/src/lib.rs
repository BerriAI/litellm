mod cache;
mod callable;
mod coercion;
mod credentials;
mod diagnostics;
mod errors;
mod execution;
mod http;
mod lifecycle;
mod logger;
mod marshal;
mod preflight;
mod python_settings;
mod routes;
mod secrets;
mod tokenizer;

#[pymodule(gil_used = true)]
mod _native {
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
    use crate::routes::audio_transcription::{atranscription, transcription};
    #[pymodule_export]
    use crate::routes::chat_completions::{
        achat_completions, acompletion, chat_completions, completion,
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
    use crate::routes::traces::{
        NativeTraceConfig, NativeTraceStorage, trace_encode_error, trace_span_rows,
    };
    #[cfg(feature = "huggingface")]
    #[pymodule_export]
    use crate::tokenizer::HuggingFaceEncoding;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;
    #[pymodule_export]
    use litellm_host_python::{ForkedAfterNativeRuntimeStarted, ProcessReservedForForking};
    use pyo3::{prelude::*, types::PyModule};

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
                "chat_completions",
                "achat_completions",
                "completion",
                "acompletion",
                "responses",
                "aresponses",
                "ResponsesWebSocketConnection",
                "NativeDiagnosticProcessor",
                "NativeTraceConfig",
                "NativeTraceStorage",
                "trace_encode_error",
                "trace_span_rows",
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
