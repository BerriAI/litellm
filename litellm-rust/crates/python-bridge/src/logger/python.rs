use std::sync::{
    OnceLock,
    atomic::{AtomicU32, Ordering},
};

use litellm_tracing::{Diagnostics, DiagnosticsConfig, Level, Logger, Metadata, Record, Sink};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

use litellm_host_python::{Pythonized, from_json_argument};

const MODULE: &str = "litellm.rust_bridge.logger";

#[derive(Default)]
struct PythonDiagnosticRuntime {
    pid: AtomicU32,
    diagnostics: Diagnostics,
}

impl PythonDiagnosticRuntime {
    fn check_process(&self) -> PyResult<()> {
        let owner = self.pid.load(Ordering::Acquire);
        if owner != 0 && owner != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "configure diagnostic exporters after worker processes fork",
            ));
        }
        Ok(())
    }
}

struct SharedSink(&'static PythonDiagnosticRuntime);

impl Sink for SharedSink {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        if metadata.target() != "litellm.diagnostics"
            && !metadata.target().starts_with("litellm_")
            && !metadata.target().starts_with("_native::")
        {
            return false;
        }
        (self.0.check_process().is_ok() && self.0.diagnostics.enabled(metadata))
            || PythonSink.enabled(metadata)
    }

    fn emit(&self, record: &Record) {
        if self.0.check_process().is_ok() {
            self.0.diagnostics.emit(record);
        }
        PythonSink.emit(record);
    }
}

fn runtime() -> &'static PythonDiagnosticRuntime {
    static RUNTIME: OnceLock<PythonDiagnosticRuntime> = OnceLock::new();
    RUNTIME.get_or_init(PythonDiagnosticRuntime::default)
}

pub(super) fn logger() -> Logger {
    static LOGGER: OnceLock<Logger> = OnceLock::new();
    LOGGER
        .get_or_init(|| Logger::new(SharedSink(runtime())))
        .clone()
}

fn level(value: i32) -> Level {
    match value {
        40.. => Level::ERROR,
        30..40 => Level::WARN,
        20..30 => Level::INFO,
        10..20 => Level::DEBUG,
        _ => Level::TRACE,
    }
}

#[pyclass]
pub(crate) struct NativeDiagnosticLogger;

#[pymethods]
impl NativeDiagnosticLogger {
    #[new]
    fn new() -> Self {
        Self
    }

    fn payload_shapes_enabled(&self) -> PyResult<bool> {
        runtime().check_process()?;
        Ok(logger().scope(litellm_tracing::payload::enabled))
    }

    #[pyo3(signature = (value, *, nodes=4096, depth=16, paths=256, bytes=16384))]
    fn extract_payload_shape(
        &self,
        value: &Bound<'_, PyAny>,
        nodes: usize,
        depth: usize,
        paths: usize,
        bytes: usize,
    ) -> PyResult<(Vec<String>, bool)> {
        let shape = super::shape::extract(
            value,
            litellm_tracing::ShapeLimits {
                nodes: nodes.min(4096),
                depth: depth.min(16),
                paths: paths.min(256),
                bytes: bytes.min(16384),
            },
        )?;
        Ok((shape.field_paths, shape.truncated))
    }

    fn emit_payload_shape(&self, py: Python<'_>, event: &str) -> PyResult<()> {
        runtime().check_process()?;
        let event: litellm_tracing::payload::PayloadEvent = from_json_argument(event)
            .map_err(|_| PyValueError::new_err("invalid payload shape event"))?;
        py.detach(|| logger().scope(|| event.emit()));
        Ok(())
    }

    fn active(&self) -> PyResult<bool> {
        runtime().check_process()?;
        Ok(runtime().diagnostics.active())
    }

    fn emit(&self, py: Python<'_>, severity: i32, message: String, fields: &str) -> PyResult<()> {
        runtime().check_process()?;
        let fields = from_json_argument(fields)
            .map_err(|_| PyValueError::new_err("diagnostic fields must be a JSON object"))?;
        py.detach(|| logger().emit(level(severity), &message, fields));
        Ok(())
    }

    fn configure(&self, py: Python<'_>, configuration: &str) -> PyResult<()> {
        runtime().check_process()?;
        let config: DiagnosticsConfig = from_json_argument(configuration)
            .map_err(|_| PyValueError::new_err("invalid diagnostic configuration"))?;
        let config = config
            .resolve(&|name| std::env::var(name).ok())
            .map_err(|_| {
                PyValueError::new_err("invalid or unavailable diagnostic configuration")
            })?;
        if config.enabled && !config.destinations.is_empty() {
            litellm_host_python::enter_native()?;
            runtime().pid.store(std::process::id(), Ordering::Release);
        }
        py.detach(|| runtime().diagnostics.configure(config))
            .map_err(|_| PyRuntimeError::new_err("could not configure diagnostic export"))
    }

    fn force_flush(&self, py: Python<'_>) -> PyResult<()> {
        runtime().check_process()?;
        py.detach(|| runtime().diagnostics.force_flush())
            .map_err(|_| PyRuntimeError::new_err("diagnostic exporter flush failed"))
    }

    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        runtime().check_process()?;
        py.detach(|| runtime().diagnostics.shutdown())
            .map_err(|_| PyRuntimeError::new_err("diagnostic exporter shutdown failed"))
    }
}

struct PythonSink;

fn python_level(level: &Level) -> u8 {
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
        if metadata.target() == litellm_tracing::payload::TARGET {
            return false;
        }
        if !metadata.target().starts_with("litellm_") && !metadata.target().starts_with("_native::")
        {
            return false;
        }
        Python::try_attach(|py| {
            report(
                py,
                py.import(MODULE)
                    .and_then(|module| {
                        module.call_method1("enabled", (python_level(metadata.level()),))
                    })
                    .and_then(|enabled| enabled.extract()),
            )
        })
        .unwrap_or(false)
    }

    fn emit(&self, record: &Record) {
        if record.metadata.target() == "litellm.diagnostics"
            || record.metadata.target() == litellm_tracing::payload::TARGET
        {
            return;
        }
        let mut fields = record.fields.clone();
        let session = fields.remove("session_id").unwrap_or_default();
        let trace = fields.remove("trace_id").unwrap_or_default();
        Python::try_attach(|py| {
            report(
                py,
                py.import(MODULE).and_then(|module| {
                    module
                        .call_method1(
                            "emit",
                            (
                                python_level(record.metadata.level()),
                                &record.message,
                                record.metadata.file().unwrap_or_default(),
                                record.metadata.line().unwrap_or_default(),
                                record.metadata.target(),
                                Pythonized(&fields),
                                (
                                    session.as_str().unwrap_or_default(),
                                    trace.as_str().unwrap_or_default(),
                                ),
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
            .and_then(|value| value.extract::<(String, String)>())
            .map(|(session, trace)| {
                logger().with_fields(serde_json::Map::from_iter([
                    ("session_id".to_owned(), session.into()),
                    ("trace_id".to_owned(), trace.into()),
                ]))
            }),
    )
}
