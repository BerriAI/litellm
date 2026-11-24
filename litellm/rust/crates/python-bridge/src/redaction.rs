//! Secret redaction as Python classes. Python compiles a redactor once and calls it for each log record, and Rust
//! code that redacts takes the same instance.

use std::sync::Arc;

use litellm_host_python::release_gil;
use litellm_redaction::TokenRedactor;
use litellm_tracing::{DiagnosticInput, Policy, Processor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

type NativeDiagnosticOutput = (String, Option<String>, Option<String>, Vec<String>, bool);

#[pyclass]
pub(crate) struct NativeTokenRedactor {
    inner: Arc<TokenRedactor>,
}

impl NativeTokenRedactor {
    /// The compiled redactor, for Rust code that redacts with the same instance.
    pub(crate) fn shared(&self) -> Arc<TokenRedactor> {
        Arc::clone(&self.inner)
    }
}

#[pymethods]
impl NativeTokenRedactor {
    #[new]
    fn new(minimum_custom_key_length: usize, word_ranges: Vec<(u32, u32)>) -> Self {
        Self {
            inner: Arc::new(TokenRedactor::new(minimum_custom_key_length, word_ranges)),
        }
    }

    fn redact(
        &self,
        py: Python<'_>,
        text: PyBackedStr,
        replacement: PyBackedStr,
    ) -> Option<String> {
        release_gil(py, || self.inner.redact(&text, &replacement))
    }
}

#[pyclass]
pub(crate) struct NativeDiagnosticProcessor {
    inner: Processor,
}

#[pymethods]
impl NativeDiagnosticProcessor {
    #[new]
    fn new(minimum_custom_key_length: usize) -> Self {
        Self {
            inner: Processor::new(minimum_custom_key_length),
        }
    }

    fn redact_text(&self, text: &str) -> PyResult<String> {
        self.inner.redact_text(text).map_err(processing_error)
    }

    fn redact_structured_text(&self, key: Option<&str>, text: &str) -> PyResult<String> {
        self.inner
            .redact_structured_text(key, text)
            .map_err(processing_error)
    }

    fn redact_client_message(&self, text: &str) -> PyResult<String> {
        self.inner
            .redact_client_message(text)
            .map_err(processing_error)
    }

    #[pyo3(signature = (message, exception, stack, leaves, policy))]
    fn process_diagnostic(
        &self,
        message: String,
        exception: Option<String>,
        stack: Option<String>,
        leaves: Vec<(Option<String>, String)>,
        policy: (bool, i64, i64),
    ) -> PyResult<NativeDiagnosticOutput> {
        let input = DiagnosticInput {
            message,
            exception,
            stack,
            leaves,
        };
        let policy = Policy {
            redact: policy.0,
            base64_limit: policy.1,
            text_limit: policy.2,
        };
        self.inner
            .process_diagnostic(&input, policy)
            .map(|output| {
                (
                    output.message,
                    output.exception,
                    output.stack,
                    output.leaves,
                    output.changed,
                )
            })
            .map_err(processing_error)
    }

    fn scrub_access_arguments(&self, arguments: Vec<String>) -> PyResult<Vec<String>> {
        self.inner
            .scrub_access_arguments(&arguments)
            .map_err(processing_error)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("diagnostic processing failed")]
struct ProcessingError(#[source] fancy_regex::Error);

fn processing_error(error: fancy_regex::Error) -> PyErr {
    PyRuntimeError::new_err(litellm_error::Report::new(&ProcessingError(error)).to_string())
}
