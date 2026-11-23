//! The one sink every Rust event in the process goes through, installed when the extension loads.
//!
//! A call into the bridge opens a span that carries its request's session and trace ids, so an event logged anywhere
//! inside that call, on any thread, is delivered with them. An event outside every call has none. An event on a
//! runtime task is queued and delivered from the blocking pool, so a Python handler never runs on an async worker.
//! An event on any other thread, such as the Python thread of a sync call, is delivered there and then.

use std::{cell::Cell, sync::OnceLock};

use litellm_tracing::{Level, Logger, Metadata, Record, Sink};
use pyo3::prelude::*;
use serde_json::{Map, Value};
use tokio::runtime::Handle;

use super::{
    background::{Background, Delivery},
    host::{self, Line},
    levels::Floors,
    route::PythonLogger,
};

const CALL: &str = "litellm.call";
const SESSION: &str = "litellm.session_id";
const TRACE: &str = "litellm.trace_id";
const CAPACITY: usize = 1024;
const BYTE_BUDGET: usize = 64 * 1024 * 1024;

static FLOORS: Floors = Floors::new();
static PROCESS: OnceLock<Logger> = OnceLock::new();

thread_local! {
    static DELIVERING: Cell<bool> = const { Cell::new(false) };
}

/// Marks this thread as running Python log handlers, so a handler that logs through Rust is not queued again.
struct Delivering;

impl Delivering {
    fn enter() -> Self {
        DELIVERING.set(true);
        Self
    }
}

impl Drop for Delivering {
    fn drop(&mut self) {
        DELIVERING.set(false);
    }
}

fn litellm_target(target: &str) -> bool {
    target.starts_with("litellm_") || target.starts_with("_native::")
}

fn call_span(metadata: &Metadata<'_>) -> bool {
    metadata.is_span() && metadata.name() == CALL
}

fn text(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or_default()
}

/// Logs one record through Python, with the ids its call span carries taken out of the fields.
fn deliver(record: &Record) {
    let correlation = (
        text(record.fields.get(SESSION)),
        text(record.fields.get(TRACE)),
    );
    let fields: Map<String, Value> = record
        .fields
        .iter()
        .filter(|(name, _)| name.as_str() != SESSION && name.as_str() != TRACE)
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    host::emit(
        &Line {
            fields: &fields,
            ..Line::from(record)
        },
        correlation,
    );
}

struct ToPython;

impl Delivery for ToPython {
    fn record(&self, record: &Record) {
        let _delivering = Delivering::enter();
        deliver(record);
    }

    fn lost(&self, count: u64) {
        let _delivering = Delivering::enter();
        host::emit(
            &Line {
                logger: PythonLogger::Sdk,
                level: host::level(&Level::WARN),
                message: &format!(
                    "{count} native log lines were dropped because the Python logger fell behind"
                ),
                file: file!(),
                line: line!(),
                target: module_path!(),
                fields: &Map::new(),
            },
            ("", ""),
        );
    }
}

struct Process {
    background: Background<ToPython>,
}

impl Process {
    fn accepts(floors: &Floors, metadata: &Metadata<'_>) -> bool {
        let target = metadata.target();
        litellm_target(target)
            && floors.enabled(PythonLogger::of(target), host::level(metadata.level()))
    }
}

/// The native runtime, when this thread is running one of its tasks. `block_on` on a Python thread is not a task, so
/// a sync call logs on its own thread. Asking for the runtime claims nothing new: a task means it already started.
fn runtime_task() -> Option<Handle> {
    (tokio::task::try_id().is_some() && litellm_host_python::runtime_started())
        .then(litellm_host_python::runtime_handle)
        .and_then(Result::ok)
}

impl Sink for Process {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        if DELIVERING.get() {
            return false;
        }
        call_span(metadata) || Self::accepts(&FLOORS, metadata)
    }

    fn emit(&self, record: &Record) {
        if call_span(record.metadata) {
            return;
        }
        match runtime_task() {
            Some(runtime) => self.background.push(&runtime, record.clone()),
            None => deliver(record),
        }
    }
}

pub(crate) fn process() -> &'static Logger {
    PROCESS.get_or_init(|| {
        Logger::new(Process {
            background: Background::new(ToPython, CAPACITY, BYTE_BUDGET),
        })
    })
}

/// The logger of the call the calling thread is making. Work scoped to it logs with that call's request ids, and
/// reading them also refreshes the level each Python logger takes.
pub(crate) fn capture(py: Python<'_>) -> Logger {
    let Some(context) = host::context(py) else {
        return Logger::default();
    };
    FLOORS.refresh(context.floors);
    let (session, trace) = context.correlation;
    process().within(|| tracing::info_span!(CALL, { SESSION } = session, { TRACE } = trace))
}

#[cfg(test)]
mod tests {
    use std::{process::Command, task::Poll};

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
            tracing::warn!("machine started");
            Box::pin(async {
                tokio::task::yield_now().await;
                tracing::warn!("machine warning");
                Ok(MachineStep::Complete(()))
            })
        }

        fn interrupt(&mut self, _: HostFailure<String>) -> Interrupted<'_, Self> {
            Box::pin(async {
                tracing::warn!("machine interrupted");
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
                    tracing::debug!(index, "stream chunk");
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
        let mut machine = crate::logger::LoggedMachine::new(ChunkMachine { raise_level });
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
        let mut machine = crate::logger::LoggedMachine::new(DiagnosticMachine);
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
            tracing::warn!(attempt = 3, retry = true, "native warning");
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
            tracing::trace!("trace");
            tracing::debug!("debug");
            tracing::info!("info");
            tracing::warn!("warn");
            tracing::error!("error");
            tracing::warn!(target: "unrelated_transport", "private wire data");
        });
    }

    #[pyfunction]
    fn limiter_warning(py: Python<'_>) {
        super::capture(py).scope(|| {
            tracing::warn!(
                target: "litellm_callbacks_parallel_request_limiter::hosted",
                "limiter warning"
            );
        });
    }

    #[pyfunction]
    fn asynchronous_warning(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async {
            tokio::task::yield_now().await;
            tracing::warn!("async warning");
            Ok(())
        })
    }

    #[pyfunction]
    fn synchronous_warning(py: Python<'_>) -> PyResult<()> {
        crate::execution::run_sync_value(py, async {
            tokio::task::yield_now().await;
            tracing::warn!("sync warning");
            Ok(())
        })
    }

    #[pyfunction]
    fn synchronous_failure(py: Python<'_>) -> PyResult<()> {
        crate::execution::run_sync_value(py, async {
            tracing::warn!("failure diagnostic");
            Err(pyo3::exceptions::PyValueError::new_err("request failed"))
        })
    }

    #[pyfunction]
    fn http_warning(py: Python<'_>) -> PyResult<()> {
        crate::http::call_config(py, &PyDict::new(py), false).map(|_| ())
    }

    #[pyfunction]
    fn messages_warning(py: Python<'_>) -> PyResult<()> {
        crate::http::messages_client(py, &PyDict::new(py), false).map(|_| ())
    }

    /// Whether the calling thread may wait on the runtime. A blocking-pool thread may, and an async worker panics.
    #[pyfunction]
    fn may_wait_on_the_runtime() -> bool {
        std::panic::catch_unwind(|| {
            tokio::runtime::Handle::try_current().is_ok_and(|runtime| {
                runtime.block_on(async {});
                true
            })
        })
        .unwrap_or(false)
    }

    #[pyfunction]
    fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        crate::execution::run_async_value(py, async {
            tracing::error!(
                target: "litellm_callbacks_parallel_request_limiter::sentinel",
                "log sentinel"
            );
            Ok(())
        })
    }

    fn event(level: litellm_tracing::Level, target: &'static str) -> bool {
        let floors = super::Floors::new();
        floors.refresh([20, 40]);
        let callsite = tracing::callsite::Identifier(&CALLSITE);
        let fields = tracing::field::FieldSet::new(&[], callsite);
        let metadata = litellm_tracing::Metadata::new(
            "event",
            target,
            level,
            None,
            None,
            None,
            fields,
            tracing::metadata::Kind::EVENT,
        );
        super::Process::accepts(&floors, &metadata)
    }

    struct Callsite;
    static CALLSITE: Callsite = Callsite;

    impl tracing::Callsite for Callsite {
        fn set_interest(&self, _: tracing::subscriber::Interest) {}

        fn metadata(&self) -> &litellm_tracing::Metadata<'_> {
            unreachable!("the filter reads the metadata it is handed")
        }
    }

    #[rstest::rstest]
    #[case::at_the_sdk_floor(
        litellm_tracing::Level::INFO,
        "litellm_redis_store::cache_invalidation",
        true
    )]
    #[case::below_the_sdk_floor(litellm_tracing::Level::DEBUG, "_native::redis::events", false)]
    #[case::at_the_proxy_floor(
        litellm_tracing::Level::ERROR,
        "litellm_callbacks_parallel_request_limiter::hosted",
        true
    )]
    #[case::below_the_proxy_floor(
        litellm_tracing::Level::WARN,
        "litellm_callbacks_parallel_request_limiter::hosted",
        false
    )]
    #[case::dependency(litellm_tracing::Level::ERROR, "hyper::proto::h1", false)]
    fn an_event_is_logged_when_its_python_logger_takes_its_level(
        #[case] level: litellm_tracing::Level,
        #[case] target: &'static str,
        #[case] accepted: bool,
    ) {
        assert_eq!(event(level, target), accepted);
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
                .set_item("sentinel", wrap_pyfunction!(sentinel, py).unwrap())
                .unwrap();
            locals
                .set_item(
                    "may_wait_on_the_runtime",
                    wrap_pyfunction!(may_wait_on_the_runtime, py).unwrap(),
                )
                .unwrap();
            locals
                .set_item(
                    "limiter_warning",
                    wrap_pyfunction!(limiter_warning, py).unwrap(),
                )
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
            locals
                .set_item(
                    "messages_warning",
                    wrap_pyfunction!(messages_warning, py).unwrap(),
                )
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
                eprintln!(
                    "SKIP: litellm package dependencies are not importable in this interpreter"
                );
                return;
            }
            py.run(c"
import asyncio
import logging
import sys
import time
sys.path.insert(0, repo_root)
sys.modules['litellm.rust_bridge._native'] = native
import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger, session_id_var, trace_id_var

class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
        self.async_lines_off_the_workers = []
    def emit(self, record):
        self.records.append(record)
        if record.getMessage() == 'async warning':
            self.async_lines_off_the_workers.append(may_wait_on_the_runtime())
        warning()

class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record)

class Broken(logging.Handler):
    def emit(self, record):
        raise ValueError('handler failed')

capture = Capture()
proxy_records = Records()
old_handlers = verbose_logger.handlers
old_level = verbose_logger.level
old_proxy_level = verbose_proxy_logger.level
old_correlation = litellm.request_correlation_in_logs
old_curve = litellm.ssl_ecdh_curve
old_unraisable = sys.unraisablehook
failures = []

async def queue_sentinel():
    await sentinel()

def drain():
    # The queue delivers in order, so the sentinel arriving means every line queued before it did.
    seen = len(proxy_records.records)
    asyncio.run(queue_sentinel())
    deadline = time.monotonic() + 10
    while len(proxy_records.records) == seen:
        assert time.monotonic() < deadline, 'queued log lines were never delivered'
        time.sleep(0.005)
    assert proxy_records.records[-1].getMessage() == 'log sentinel'

try:
    verbose_logger.handlers = [capture]
    verbose_proxy_logger.addHandler(proxy_records)
    verbose_proxy_logger.setLevel(logging.WARNING)
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
    assert record.pathname.endswith('logger/process.rs')
    assert record.lineno > 0
    assert record.rust_target.endswith('logger::process::tests')
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

    loop_thread = __import__('threading').get_ident()
    asyncio.run(concurrent())
    drain()
    assert sorted((r.getMessage(), r.session_id, r.trace_id) for r in capture.records[1:]) == sorted(
        (message, name, 'trace-' + name)
        for name in ('first', 'second')
        for message in ('async warning', 'sync warning', 'machine started', 'machine warning', 'machine interrupted')
    )
    assert all(r.rust_fields == {} for r in capture.records[1:])
    assert capture.async_lines_off_the_workers == [True, True]
    assert {r.thread == loop_thread for r in capture.records[1:] if r.getMessage() == 'sync warning'} == {True}

    verbose_logger.setLevel(logging.DEBUG)
    before_levels = len(capture.records)
    levels()
    assert [(r.getMessage(), r.levelno) for r in capture.records[before_levels:]] == [
        ('trace', logging.DEBUG), ('debug', logging.DEBUG), ('info', logging.INFO),
        ('warn', logging.WARNING), ('error', logging.ERROR),
    ]

    before_limiter = len(capture.records)
    limiter_warning()
    assert [(r.name, r.getMessage()) for r in proxy_records.records if r.getMessage() != 'log sentinel'] == [
        ('LiteLLM Proxy', 'limiter warning')
    ]
    assert capture.records[before_limiter:] == []
    verbose_proxy_logger.setLevel(logging.ERROR)
    limiter_warning()
    assert len([r for r in proxy_records.records if r.getMessage() == 'limiter warning']) == 1

    async def chunks(raise_level):
        await machine_chunks(raise_level)
    try:
        verbose_logger.setLevel(logging.WARNING)
        before_chunks = len(capture.records)
        asyncio.run(chunks(False))
        drain()
        assert capture.records[before_chunks:] == []

        verbose_logger.setLevel(logging.DEBUG)
        asyncio.run(chunks(False))
        drain()
        assert [(r.getMessage(), r.levelno, r.rust_fields) for r in capture.records[before_chunks:]] == [
            ('stream chunk', logging.DEBUG, {'index': 0}),
            ('stream chunk', logging.DEBUG, {'index': 1}),
            ('stream chunk', logging.DEBUG, {'index': 2}),
        ]

        # A level lowered while a call runs is read by the next call that starts, not by the running one.
        verbose_logger.setLevel(logging.WARNING)
        warning()
        before_raise = len(capture.records)
        asyncio.run(chunks(True))
        assert verbose_logger.level == logging.DEBUG
        verbose_logger.setLevel(logging.WARNING)
        drain()
        assert capture.records[before_raise:] == []
    finally:
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
    drain()
    span_records = capture.records[before_spans:]
    assert [r.getMessage() for r in span_records] == ['inside route', 'span closed']
    assert all(r.session_id == 'span-session' and r.trace_id == 'span-trace' for r in span_records)
    assert span_records[0].rust_fields == {'route': 'fixture'}
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

    before = len(capture.records)
    litellm.ssl_ecdh_curve = 'logger-test-messages-curve'
    messages_warning()
    messages_warning()
    assert len(capture.records) == before + 1
    assert 'logger-test-messages-curve' in capture.records[-1].getMessage()

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
    verbose_proxy_logger.removeHandler(proxy_records)
    verbose_proxy_logger.setLevel(old_proxy_level)
    litellm.request_correlation_in_logs = old_correlation
    litellm.ssl_ecdh_curve = old_curve
", Some(&locals), Some(&locals)).unwrap();
        });
    }
}
