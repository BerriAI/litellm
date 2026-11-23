//! Every call into `litellm.rust_bridge.host.logger`, Python's side of this module. Each function is one function
//! there. A failure in Python is reported as unraisable and never reaches the Rust code that logged.

use litellm_host_python::Pythonized;
use litellm_tracing::{Level, Record};
use pyo3::prelude::*;
use serde_json::{Map, Value};
use strum::{EnumCount, VariantArray};

use super::route::PythonLogger;

const MODULE: &str = "litellm.rust_bridge.host.logger";

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

/// What a call reads from Python when it starts.
pub(super) struct Context {
    /// The session and trace ids of the request the calling thread is serving.
    pub(super) correlation: (String, String),
    /// The lowest level each Python logger takes, in `PythonLogger` order.
    pub(super) floors: [u8; PythonLogger::COUNT],
}

pub(super) fn context(py: Python<'_>) -> Option<Context> {
    let loggers: Vec<&str> = PythonLogger::VARIANTS
        .iter()
        .map(|logger| <&str>::from(*logger))
        .collect();
    report(
        py,
        py.import(MODULE)
            .and_then(|module| module.call_method1("context", (loggers,)))
            .and_then(|value| value.extract())
            .map(|(correlation, floors)| {
                Some(Context {
                    correlation,
                    floors,
                })
            }),
    )
}

/// One log line as Python's `emit` takes it.
pub(super) struct Line<'a> {
    pub(super) logger: PythonLogger,
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
            logger: PythonLogger::of(record.metadata.target()),
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
                            <&str>::from(line.logger),
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
