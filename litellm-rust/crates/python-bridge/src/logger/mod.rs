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
mod tests {
    use std::{process::Command, task::Poll};

    use litellm_host::{
        MachineFault,
        machine::{HostFailure, Interrupted, Machine, MachineStep, Step},
        protocol::Protocol,
    };

    use pyo3::{prelude::*, types::PyDict};

    struct DiagnosticMachine;

    #[derive(Clone, Debug)]
    struct DiagnosticError(String);

    impl From<MachineFault> for DiagnosticError {
        fn from(fault: MachineFault) -> Self {
            Self(fault.to_string())
        }
    }

    impl Protocol for DiagnosticMachine {
        type Response = ();
        type Error = DiagnosticError;
        type Projection = ();
        type Op = ();
        type Chunk = ();
        type StreamHead = ();
    }

    impl Machine for DiagnosticMachine {
        type Protocol = Self;
        type Complete = ();

        fn resume(&mut self) -> Step<'_, Self> {
            litellm_tracing::warn!("machine started");
            Box::pin(async {
                tokio::task::yield_now().await;
                litellm_tracing::warn!("machine warning");
                Ok(MachineStep::Complete(()))
            })
        }

        fn interrupt(&mut self, _: HostFailure<DiagnosticError>) -> Interrupted<'_, Self> {
            Box::pin(async {
                litellm_tracing::warn!("machine interrupted");
                Ok(())
            })
        }
    }

    #[pyfunction]
    fn machine_warning(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let mut machine = super::LoggedMachine::new(DiagnosticMachine);
        let mut future = Box::pin(async move {
            machine
                .resume()
                .await
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.0))?;
            machine
                .interrupt(HostFailure::Error(DiagnosticError("stop".into())))
                .await
                .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.0))
        });
        assert!(matches!(
            litellm_host_python::poll_async_value(py, future.as_mut())?,
            Poll::Pending
        ));
        litellm_host_python::run_async_value(py, future)
    }

    #[pyfunction]
    fn warning(py: Python<'_>) {
        super::capture(py).scope(|| {
            litellm_tracing::warn!(attempt = 3, retry = true, "native warning");
        });
    }

    #[pyfunction]
    fn levels(py: Python<'_>) {
        super::capture(py).scope(|| {
            litellm_tracing::trace!("trace");
            litellm_tracing::debug!("debug");
            litellm_tracing::info!("info");
            litellm_tracing::warn!("warn");
            litellm_tracing::error!("error");
            litellm_tracing::warn!(target: "unrelated_transport", "private wire data");
        });
    }

    #[pyfunction]
    fn asynchronous_warning(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        super::run_async_value(py, async {
            tokio::task::yield_now().await;
            litellm_tracing::warn!("async warning");
            Ok(())
        })
    }

    #[pyfunction]
    fn synchronous_warning(py: Python<'_>) -> PyResult<()> {
        super::run_sync_value(py, async {
            tokio::task::yield_now().await;
            litellm_tracing::warn!("sync warning");
            Ok(())
        })
    }

    #[pyfunction]
    fn synchronous_failure(py: Python<'_>) -> PyResult<()> {
        super::run_sync_value(py, async {
            litellm_tracing::warn!("failure diagnostic");
            Err(pyo3::exceptions::PyValueError::new_err("request failed"))
        })
    }

    #[pyfunction]
    fn http_warning(py: Python<'_>) -> PyResult<()> {
        crate::http::call_config(py, &PyDict::new(py), false).map(|_| ())
    }

    #[test]
    fn native_events_reach_python_with_levels_context_reentry_and_http_deduplication() {
        if std::env::var_os("LITELLM_LOGGER_TEST_PROCESS").is_none() {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    std::thread::current().name().unwrap(),
                    "--nocapture",
                ])
                .env("LITELLM_LOGGER_TEST_PROCESS", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item(
                    "repo_root",
                    concat!(env!("CARGO_MANIFEST_DIR"), "/../../.."),
                )
                .unwrap();
            locals
                .set_item(
                    "machine_warning",
                    wrap_pyfunction!(machine_warning, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item(
                    "synchronous_failure",
                    wrap_pyfunction!(synchronous_failure, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item("levels", wrap_pyfunction!(levels, py).unwrap())
                .unwrap();
            locals
                .set_item("warning", wrap_pyfunction!(warning, py).unwrap())
                .unwrap();
            locals
                .set_item(
                    "asynchronous_warning",
                    wrap_pyfunction!(asynchronous_warning, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item(
                    "synchronous_warning",
                    wrap_pyfunction!(synchronous_warning, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item("http_warning", wrap_pyfunction!(http_warning, py).unwrap())
                .unwrap();
            let importable = py
            .eval(
                c"__import__('importlib.util', fromlist=['util']).find_spec('dotenv') is not None",
                Some(&locals),
                Some(&locals),
            )
            .unwrap()
            .is_truthy()
            .unwrap();
            if !importable {
                eprintln!(
                    "SKIP: litellm package dependencies are not importable in this interpreter"
                );
                return;
            }
            py.run(c"
import asyncio
import logging
import sys
sys.path.insert(0, repo_root)
import litellm
from litellm._logging import verbose_logger, session_id_var, trace_id_var

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record)
        warning()

class Broken(logging.Handler):
    def emit(self, record):
        raise ValueError('handler failed')

capture = Capture()
old_handlers = verbose_logger.handlers
old_level = verbose_logger.level
old_correlation = litellm.request_correlation_in_logs
old_curve = litellm.ssl_ecdh_curve
old_unraisable = sys.unraisablehook
failures = []
try:
    verbose_logger.handlers = [capture]
    litellm.request_correlation_in_logs = True
    verbose_logger.setLevel(logging.ERROR)
    warning()
    assert capture.records == []
    verbose_logger.setLevel(logging.WARNING)
    warning()
    assert len(capture.records) == 1
    record = capture.records[0]
    assert record.getMessage() == 'native warning'
    assert record.levelno == logging.WARNING
    assert record.rust_fields == {'attempt': 3, 'retry': True}
    assert record.pathname.endswith('logger/mod.rs')
    assert record.lineno > 0
    assert record.rust_target.endswith('logger::tests')
    verbose_logger.setLevel(logging.ERROR)
    warning()
    assert len(capture.records) == 1
    verbose_logger.setLevel(logging.WARNING)

    async def request(name):
        session = session_id_var.set(name)
        trace = trace_id_var.set('trace-' + name)
        try:
            await asynchronous_warning()
            await machine_warning()
            synchronous_warning()
            assert session_id_var.get() == name
            assert trace_id_var.get() == 'trace-' + name
        finally:
            trace_id_var.reset(trace)
            session_id_var.reset(session)

    async def concurrent():
        await asyncio.gather(request('first'), request('second'))

    asyncio.run(concurrent())
    assert sorted((r.getMessage(), r.session_id, r.trace_id) for r in capture.records[1:]) == sorted(
        (message, name, 'trace-' + name)
        for name in ('first', 'second')
        for message in ('async warning', 'sync warning', 'machine started', 'machine warning', 'machine interrupted')
    )

    verbose_logger.setLevel(logging.DEBUG)
    before_levels = len(capture.records)
    levels()
    assert [(r.getMessage(), r.levelno) for r in capture.records[before_levels:]] == [
        ('trace', logging.DEBUG), ('debug', logging.DEBUG), ('info', logging.INFO),
        ('warn', logging.WARNING), ('error', logging.ERROR),
    ]

    before = len(capture.records)
    litellm.ssl_ecdh_curve = 'logger-test-unsupported-curve'
    http_warning()
    http_warning()
    assert len(capture.records) == before + 1
    assert 'logger-test-unsupported-curve' in capture.records[-1].getMessage()
    assert capture.records[-1].pathname.endswith('http.rs')

    verbose_logger.handlers = [Broken()]
    sys.unraisablehook = failures.append
    warning()
    assert len(failures) == 1
    assert str(failures[0].exc_value) == 'handler failed'
    try:
        synchronous_failure()
    except ValueError as error:
        assert str(error) == 'request failed'
    else:
        raise AssertionError('request failure was lost')
    assert len(failures) == 2
finally:
    sys.unraisablehook = old_unraisable
    verbose_logger.handlers = old_handlers
    verbose_logger.setLevel(old_level)
    litellm.request_correlation_in_logs = old_correlation
    litellm.ssl_ecdh_curve = old_curve
", Some(&locals), Some(&locals)).unwrap();
        });
    }
}
