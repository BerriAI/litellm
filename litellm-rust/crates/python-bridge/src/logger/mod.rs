mod execution;
mod machine;

pub(crate) use execution::{run_async, run_async_value, run_sync, run_sync_value};
pub(crate) use machine::LoggedMachine;

use litellm_host_python::Pythonized;
use litellm_tracing::{DiagnosticInput, Level, Logger, Metadata, Policy, Processor, Record, Sink};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.logger";
type NativeDiagnosticOutput = (String, Option<String>, Option<String>, Vec<String>, bool);

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

fn processing_error(_: fancy_regex::Error) -> PyErr {
    PyRuntimeError::new_err("diagnostic processing failed")
}

struct PythonSink {
    correlation: (String, String),
}

fn level(level: &Level) -> u8 {
    match *level {
        Level::ERROR => 40,
        Level::WARN => 30,
        Level::INFO => 20,
        Level::DEBUG | Level::TRACE => 10,
    }
}

fn report<T: Default>(py: Python<'_>, result: PyResult<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            error.write_unraisable(py, None);
            T::default()
        }
    }
}

impl Sink for PythonSink {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        if !metadata.target().starts_with("litellm_") && !metadata.target().starts_with("_native::")
        {
            return false;
        }
        Python::try_attach(|py| {
            report(
                py,
                py.import(MODULE)
                    .and_then(|module| module.call_method1("enabled", (level(metadata.level()),)))
                    .and_then(|enabled| enabled.extract()),
            )
        })
        .unwrap_or(false)
    }

    fn emit(&self, record: &Record) {
        Python::try_attach(|py| {
            report(
                py,
                py.import(MODULE).and_then(|module| {
                    module
                        .call_method1(
                            "emit",
                            (
                                level(record.metadata.level()),
                                &record.message,
                                record.metadata.file().unwrap_or_default(),
                                record.metadata.line().unwrap_or_default(),
                                record.metadata.target(),
                                Pythonized(&record.fields),
                                (&self.correlation.0, &self.correlation.1),
                            ),
                        )
                        .map(|_| ())
                }),
            );
        });
    }
}

pub(crate) fn capture(py: Python<'_>) -> Logger {
    report(
        py,
        py.import(MODULE)
            .and_then(|module| module.call_method0("context"))
            .and_then(|value| value.extract())
            .map(|correlation| Logger::new(PythonSink { correlation })),
    )
}

#[cfg(test)]
mod tests;
