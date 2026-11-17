mod machine;

pub(crate) use machine::LoggedMachine;

use std::sync::atomic::{AtomicU8, Ordering};

use litellm_host_python::{Pythonized, release_gil};
use litellm_tracing::{
    DiagnosticInput, Level, Logger, Metadata, Policy, Processor, Record, Sink, TokenRedactor,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;

const MODULE: &str = "litellm.rust_bridge.host.logger";
type NativeDiagnosticOutput = (String, Option<String>, Option<String>, Vec<String>, bool);

#[pyclass]
pub(crate) struct NativeTokenRedactor {
    inner: TokenRedactor,
}

#[pymethods]
impl NativeTokenRedactor {
    #[new]
    fn new(minimum_custom_key_length: usize, word_ranges: Vec<(u32, u32)>) -> Self {
        Self {
            inner: TokenRedactor::new(minimum_custom_key_length, word_ranges),
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

struct PythonSink {
    correlation: (String, String),
    levels: Option<Levels>,
}

#[derive(Default)]
struct Levels([AtomicU8; 4]);

const UNKNOWN: u8 = 0;
const DISABLED: u8 = 1;
const ENABLED: u8 = 2;

impl Levels {
    fn enabled(&self, level: u8, query: impl FnOnce() -> bool) -> bool {
        let slot = &self.0[usize::from(level / 10 - 1)];
        match slot.load(Ordering::Relaxed) {
            UNKNOWN => {
                let enabled = query();
                slot.store(if enabled { ENABLED } else { DISABLED }, Ordering::Relaxed);
                enabled
            }
            state => state == ENABLED,
        }
    }
}

fn python_enabled(level: u8) -> bool {
    Python::try_attach(|py| {
        report(
            py,
            py.import(MODULE)
                .and_then(|module| module.call_method1("enabled", (level,)))
                .and_then(|enabled| enabled.extract()),
        )
    })
    .unwrap_or(false)
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
        let level = level(metadata.level());
        match &self.levels {
            Some(levels) => levels.enabled(level, || python_enabled(level)),
            None => python_enabled(level),
        }
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
    logger(py, None)
}

pub(crate) fn capture_call(py: Python<'_>) -> Logger {
    logger(py, Some(Levels::default()))
}

fn logger(py: Python<'_>, levels: Option<Levels>) -> Logger {
    report(
        py,
        py.import(MODULE)
            .and_then(|module| module.call_method0("context"))
            .and_then(|value| value.extract())
            .map(|correlation| {
                Logger::new(PythonSink {
                    correlation,
                    levels,
                })
            }),
    )
}

#[cfg(test)]
mod tests;
