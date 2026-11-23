//! Every call into `litellm.rust_bridge.host.logger`, Python's side of this module. Each function is one function
//! there. A failure in Python is reported as unraisable and never reaches the Rust code that logged.

use litellm_host_python::Pythonized;
use litellm_tracing::{Level, Record};
use pyo3::prelude::*;

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

/// Logs one record through Python's filters and handlers.
pub(super) fn emit(record: &Record, correlation: (&str, &str)) {
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
                            correlation,
                        ),
                    )
                    .map(|_| ())
            }),
        );
    });
}
