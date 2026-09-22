mod execution;
mod machine;

pub(crate) use execution::{run_async, run_async_value, run_sync, run_sync_value};
pub(crate) use machine::LoggedMachine;

use litellm_host_python::Pythonized;
use litellm_logger::{Level, Logger, Metadata, Record, Sink};
use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.logger";

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
