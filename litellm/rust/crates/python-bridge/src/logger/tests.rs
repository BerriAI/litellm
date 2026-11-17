use std::{cell::Cell, process::Command, task::Poll};

use litellm_host::{
    machine::{HostFailure, Interrupted, Machine, MachineStep, Step},
    protocol::Protocol,
};

use pyo3::{prelude::*, types::PyDict};

struct DiagnosticMachine;

impl Protocol for DiagnosticMachine {
    type Response = ();
    type Error = String;
    type Request = ();
    type HostCall = ();
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

    fn interrupt(&mut self, _: HostFailure<String>) -> Interrupted<'_, Self> {
        Box::pin(async {
            litellm_tracing::warn!("machine interrupted");
            Ok(())
        })
    }
}

struct ChunkMachine {
    raise_level: bool,
}

impl Machine for ChunkMachine {
    type Protocol = DiagnosticMachine;
    type Complete = ();

    fn resume(&mut self) -> Step<'_, Self> {
        let raise_level = self.raise_level;
        Box::pin(async move {
            for index in 0..3 {
                litellm_tracing::debug!(index, "stream chunk");
                if raise_level {
                    Python::attach(|py| {
                        py.import("litellm._logging")?
                            .getattr("verbose_logger")?
                            .call_method1("setLevel", (10,))
                            .map(|_| ())
                    })
                    .map_err(|error| error.to_string())?;
                }
                tokio::task::yield_now().await;
            }
            Ok(MachineStep::Complete(()))
        })
    }

    fn interrupt(&mut self, _: HostFailure<String>) -> Interrupted<'_, Self> {
        Box::pin(async { Ok(()) })
    }
}

#[pyfunction]
fn machine_chunks(py: Python<'_>, raise_level: bool) -> PyResult<Bound<'_, PyAny>> {
    let mut machine = super::LoggedMachine::new(ChunkMachine { raise_level });
    litellm_host_python::run_async_value(py, async move {
        machine
            .resume()
            .await
            .map(|_| ())
            .map_err(pyo3::exceptions::PyValueError::new_err)
    })
}

#[pyfunction]
fn machine_warning(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let mut machine = super::LoggedMachine::new(DiagnosticMachine);
    let mut future = Box::pin(async move {
        machine
            .resume()
            .await
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        machine
            .interrupt(HostFailure::Error("stop".into()))
            .await
            .map_err(pyo3::exceptions::PyValueError::new_err)
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

#[tracing::instrument(name = "litellm.route", skip_all, fields(route = "fixture", outcome))]
async fn traced_operation(_secret: &str) -> PyResult<()> {
    tokio::task::yield_now().await;
    tracing::info!("inside route");
    tracing::Span::current().record("outcome", "success");
    Ok(())
}

#[pyfunction]
fn span_warning(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    crate::execution::run_async_value(py, traced_operation("private-key-sentinel"))
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
    crate::execution::run_async_value(py, async {
        tokio::task::yield_now().await;
        litellm_tracing::warn!("async warning");
        Ok(())
    })
}

#[pyfunction]
fn synchronous_warning(py: Python<'_>) -> PyResult<()> {
    crate::execution::run_sync_value(py, async {
        tokio::task::yield_now().await;
        litellm_tracing::warn!("sync warning");
        Ok(())
    })
}

#[pyfunction]
fn synchronous_failure(py: Python<'_>) -> PyResult<()> {
    crate::execution::run_sync_value(py, async {
        litellm_tracing::warn!("failure diagnostic");
        Err(pyo3::exceptions::PyValueError::new_err("request failed"))
    })
}

#[pyfunction]
fn http_warning(py: Python<'_>) -> PyResult<()> {
    crate::http::call_config(py, &PyDict::new(py), false).map(|_| ())
}

#[rstest::rstest]
#[case::debug(10)]
#[case::info(20)]
#[case::warning(30)]
#[case::error(40)]
fn call_levels_ask_python_once_per_level(#[case] level: u8, #[values(true, false)] enabled: bool) {
    let levels = super::Levels::default();
    let queries = Cell::new(0);
    let query = || {
        queries.set(queries.get() + 1);
        enabled
    };
    assert_eq!(levels.enabled(level, query), enabled);
    assert_eq!(levels.enabled(level, query), enabled);
    assert_eq!(levels.enabled(level, query), enabled);
    assert_eq!(queries.get(), 1);
}

#[rstest::rstest]
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
                concat!(env!("CARGO_MANIFEST_DIR"), "/../../../.."),
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
                "machine_chunks",
                wrap_pyfunction!(machine_chunks, py).unwrap(),
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
            .set_item("span_warning", wrap_pyfunction!(span_warning, py).unwrap())
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
        locals.set_item("native", crate::native_module(py)).unwrap();
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
            eprintln!("SKIP: litellm package dependencies are not importable in this interpreter");
            return;
        }
        py.run(c"
import asyncio
import logging
import sys
sys.path.insert(0, repo_root)
sys.modules['litellm.rust_bridge._native'] = native
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
    assert record.pathname.endswith('logger/tests.rs')
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

    import litellm.rust_bridge.host.logger as host_logger
    original_enabled = host_logger.enabled
    queries = []
    def counting_enabled(level):
        queries.append(level)
        return original_enabled(level)
    host_logger.enabled = counting_enabled
    async def chunks(raise_level):
        await machine_chunks(raise_level)
    try:
        verbose_logger.setLevel(logging.WARNING)
        before_chunks = len(capture.records)
        asyncio.run(chunks(False))
        assert capture.records[before_chunks:] == []
        assert queries == [logging.DEBUG]

        queries.clear()
        verbose_logger.setLevel(logging.DEBUG)
        asyncio.run(chunks(False))
        assert [(r.getMessage(), r.levelno, r.rust_fields) for r in capture.records[before_chunks:]] == [
            ('stream chunk', logging.DEBUG, {'index': 0}),
            ('stream chunk', logging.DEBUG, {'index': 1}),
            ('stream chunk', logging.DEBUG, {'index': 2}),
        ]
        assert queries == [logging.DEBUG] * 4

        queries.clear()
        verbose_logger.setLevel(logging.WARNING)
        before_raise = len(capture.records)
        asyncio.run(chunks(True))
        assert capture.records[before_raise:] == []
        assert queries == [logging.DEBUG]
    finally:
        host_logger.enabled = original_enabled
        verbose_logger.setLevel(logging.DEBUG)

    before_spans = len(capture.records)
    async def traced_request():
        session = session_id_var.set('span-session')
        trace = trace_id_var.set('span-trace')
        try:
            await span_warning()
        finally:
            trace_id_var.reset(trace)
            session_id_var.reset(session)
    asyncio.run(traced_request())
    span_records = capture.records[before_spans:]
    assert [r.getMessage() for r in span_records] == ['inside route', 'span closed']
    assert all(r.session_id == 'span-session' and r.trace_id == 'span-trace' for r in span_records)
    assert all(r.rust_fields['route'] == 'fixture' for r in span_records)
    assert span_records[1].rust_fields['outcome'] == 'success'
    assert span_records[1].rust_fields['span_name'] == 'litellm.route'
    assert span_records[1].rust_fields['duration_ms'] >= 0
    assert 'private-key-sentinel' not in repr([r.rust_fields for r in span_records])

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
