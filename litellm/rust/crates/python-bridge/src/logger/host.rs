//! Every call into `litellm.rust_bridge.host.logger`, Python's side of this module. Each function is one function
//! there. A failure in Python is reported as unraisable and never reaches the Rust code that logged.

use litellm_host_python::Pythonized;
use litellm_tracing::{Level, Record};
use pyo3::prelude::*;
use serde_json::{Map, Value};

const MODULE: &str = "litellm.rust_bridge.host.logger";

/// The targets `litellm.rust_bridge.host.logger` logs on the `LiteLLM Proxy` logger.
const PROXY_TARGETS: [&str; 1] = ["litellm_callbacks_parallel_request_limiter"];

pub(super) fn proxy_target(target: &str) -> bool {
    PROXY_TARGETS
        .iter()
        .any(|prefix| target.starts_with(prefix))
}

/// The Python `logging` number for a Rust level.
pub(super) fn level(level: &Level) -> u8 {
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

/// The session and trace ids of the request the calling thread is serving.
pub(super) fn context(py: Python<'_>) -> Option<(String, String)> {
    report(
        py,
        py.import(MODULE)
            .and_then(|module| module.call_method0("context"))
            .and_then(|value| value.extract())
            .map(Some),
    )
}

/// Whether the Python logger for `target` takes records at `level`.
pub(super) fn enabled(level: u8, target: &str) -> bool {
    Python::try_attach(|py| {
        report(
            py,
            py.import(MODULE)
                .and_then(|module| module.call_method1("enabled", (level, target)))
                .and_then(|enabled| enabled.extract()),
        )
    })
    .unwrap_or(false)
}

/// One log line as Python's `emit` takes it.
pub(super) struct Line<'a> {
    pub(super) level: u8,
    pub(super) message: &'a str,
    pub(super) file: &'a str,
    pub(super) line: u32,
    pub(super) target: &'a str,
    pub(super) fields: &'a Map<String, Value>,
}

impl<'a> From<&'a Record> for Line<'a> {
    fn from(record: &'a Record) -> Self {
        Self {
            level: level(record.metadata.level()),
            message: &record.message,
            file: record.metadata.file().unwrap_or_default(),
            line: record.metadata.line().unwrap_or_default(),
            target: record.metadata.target(),
            fields: &record.fields,
        }
    }
}

/// Logs one line through Python's filters and handlers.
pub(super) fn emit(line: &Line<'_>, correlation: (&str, &str)) {
    Python::try_attach(|py| {
        report(
            py,
            py.import(MODULE).and_then(|module| {
                module
                    .call_method1(
                        "emit",
                        (
                            line.level,
                            line.message,
                            line.file,
                            line.line,
                            line.target,
                            Pythonized(line.fields),
                            correlation,
                        ),
                    )
                    .map(|_| ())
            }),
        );
    });
}
