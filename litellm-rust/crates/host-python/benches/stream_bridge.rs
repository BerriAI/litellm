//! The Python/Rust stream bridge floor: one native call streams `CHUNKS` chunks to an asyncio
//! consumer with nothing else on the path (no HTTP, no routing, no logging), so the per-chunk
//! cost is the driver, the `Execution` handle and the future bridge alone.
//!
//! `native/ready` delivers every chunk from a future that is complete on its first poll, so the
//! driver never hands Python an awaitable. `native/pending` yields to Tokio before each chunk,
//! so every read suspends through the future bridge (a spawned task, a Python future and a
//! `call_soon_threadsafe` completion). `native/sync` is the blocking form. `python/*` runs the
//! same consumer loop over a pure-Python async generator, with `asyncio.sleep(0)` standing in
//! for the pending case.
//!
//! `stream_bridge` and `chunk_bytes` measure wall time. `stream_bridge_cpu` measures process CPU
//! time across every thread, which is what the proxy benchmark saw grow: a mode whose CPU time
//! exceeds its wall time is burning Tokio worker or blocking-pool threads behind the consumer.
//!
//! Run with `cargo bench -p litellm-host-python --bench stream_bridge`.

use std::{
    convert::Infallible, ffi::CString, fmt, hint::black_box, sync::OnceLock, time::Duration,
};

use bytes::Bytes;
use criterion::{
    BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::{Measurement, ValueFormatter},
};
use litellm_host::{
    event::{RequestContext, Timing, WireRequest},
    host::Demand,
    machine::{CallMachine, MachineFault},
    protocol::Protocol,
};
use litellm_host_python::{
    InvokeError, LifecycleEvent, LifecycleStep, ProtocolHost, PythonLifecycle, missing_state,
    release_count, run_call,
};
use pyo3::{
    exceptions::PyRuntimeError,
    ffi::c_str,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};

const CHUNKS: usize = 1_500;

/// One `content_block_delta` frame as the paced proxy benchmark's mock emitted it.
fn delta_frame() -> Bytes {
    Bytes::from_static(
        concat!(
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",",
            "\"text\":\"1499:1758790000000000000 The stream continues.\\n\"}}\n\n",
        )
        .as_bytes(),
    )
}

#[derive(Clone, Debug)]
struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<MachineFault> for Error {
    fn from(fault: MachineFault) -> Self {
        Self(format!("{fault:?}"))
    }
}

struct Stream;

impl Protocol for Stream {
    type Response = ();
    type Error = Error;
    type Projection = ();
    type Op = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Native {
    Ready,
    Pending,
    Sync,
}

impl Native {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Pending => "pending",
            Self::Sync => "sync",
        }
    }
}

fn machine(native: Native, chunks: usize, payload: Bytes) -> CallMachine<Stream> {
    CallMachine::new(move |host| {
        Box::pin(async move {
            host.project().await?;
            if host.open(()).await? == Demand::Detached {
                return Ok(());
            }
            for _ in 0..chunks {
                if native == Native::Pending {
                    tokio::task::yield_now().await;
                }
                if host.deliver(payload.clone()).await? == Demand::Detached {
                    break;
                }
            }
            Ok(())
        })
    })
}

struct BytesHost;

impl ProtocolHost for BytesHost {
    type Protocol = Stream;
    type Failure = PyErr;

    fn project(&mut self, _: Python<'_>, _: &Bound<'_, PyDict>) -> Result<(), InvokeError<Error>> {
        Ok(())
    }

    fn invoke(&mut self, _: Python<'_>, op: Infallible) -> Result<(), InvokeError<Error>> {
        match op {}
    }

    fn complete(&mut self, py: Python<'_>, (): ()) -> PyResult<Py<PyAny>> {
        Ok(py.None())
    }

    fn head(&mut self, py: Python<'_>, (): ()) -> PyResult<Py<PyAny>> {
        Ok(py.None())
    }

    fn chunk(&mut self, py: Python<'_>, chunk: Bytes) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, &chunk).into_any().unbind())
    }

    fn classify(&self, _: Python<'_>, error: Error) -> PyResult<PyErr> {
        Ok(PyRuntimeError::new_err(error.0))
    }

    fn host_error(error: &PyErr) -> Error {
        Error(error.to_string())
    }

    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

struct Silent;

impl PythonLifecycle for Silent {
    fn begin(&mut self, _: Python<'_>, arguments: Py<PyDict>, _: f64) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Arguments(arguments))
    }

    fn before_send(
        &mut self,
        _: Python<'_>,
        wire: Box<WireRequest>,
        _: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Wire(wire))
    }

    fn after_success(
        &mut self,
        _: Python<'_>,
        response: Py<PyAny>,
        _: Timing,
    ) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Response(response))
    }

    fn emit(&mut self, _: Python<'_>, _: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        Ok(LifecycleStep::Done)
    }

    fn opened(&mut self, _: Python<'_>) -> PyResult<()> {
        Ok(())
    }

    fn delivered(&mut self, _: Python<'_>, _: &Py<PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn resume(&mut self, _: Python<'_>, _: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        Err(missing_state())
    }

    fn close(&mut self, _: Python<'_>) {}

    fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

fn install_harness(py: Python<'_>) -> Py<PyModule> {
    py.run(
        c_str!(
            r#"
import sys
import types

sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm.rust_bridge', types.ModuleType('litellm.rust_bridge'))
"#
        ),
        None,
        None,
    )
    .expect("package stubs should install");
    let lifecycle = CString::new(include_str!("../../../../litellm/rust_bridge/lifecycle.py"))
        .expect("lifecycle.py has no interior NUL");
    PyModule::from_code(
        py,
        &lifecycle,
        c_str!("lifecycle.py"),
        c_str!("litellm.rust_bridge.lifecycle"),
    )
    .expect("the lifecycle module should load");
    let harness =
        CString::new(include_str!("stream_bridge.py")).expect("harness has no interior NUL");
    PyModule::from_code(
        py,
        &harness,
        c_str!("stream_bridge.py"),
        c_str!("stream_bridge_bench"),
    )
    .expect("the bench harness should load")
    .unbind()
}

fn harness(py: Python<'_>) -> Bound<'_, PyModule> {
    static HARNESS: OnceLock<Py<PyModule>> = OnceLock::new();
    HARNESS.get_or_init(|| install_harness(py)).bind(py).clone()
}

fn handed(py: Python<'_>, native: Native, payload: &Bytes) -> Py<PyAny> {
    run_call(
        py,
        machine(native, CHUNKS, payload.clone()),
        BytesHost,
        Box::new(Silent),
        PyDict::new(py).unbind(),
        native != Native::Sync,
    )
    .expect("the native call should start")
}

fn consume_native(
    py: Python<'_>,
    harness: &Bound<'_, PyModule>,
    native: Native,
    payload: &Bytes,
) -> usize {
    let handed = handed(py, native, payload);
    let count = match native {
        Native::Sync => handed
            .bind(py)
            .try_iter()
            .expect("a sync stream iterates")
            .filter(|chunk| chunk.is_ok())
            .count(),
        Native::Ready | Native::Pending => harness
            .call_method1("consume", (handed,))
            .expect("the async stream should drain")
            .extract()
            .expect("consume returns the chunk count"),
    };
    assert_eq!(
        count,
        CHUNKS,
        "{} delivered the wrong chunk count",
        native.label()
    );
    count
}

fn consume_python(
    py: Python<'_>,
    harness: &Bound<'_, PyModule>,
    payload: &Bytes,
    pending: bool,
) -> usize {
    let count: usize = harness
        .call_method1(
            "consume_python",
            (CHUNKS, PyBytes::new(py, payload), pending),
        )
        .expect("the Python stream should drain")
        .extract()
        .expect("consume_python returns the chunk count");
    assert_eq!(
        count, CHUNKS,
        "python pending={pending} delivered the wrong chunk count"
    );
    count
}

/// Proves each mode exercises the path its name claims before anything is timed: `ready` must
/// never hand Python an awaitable, `pending` must do so once per chunk. Resumes are the
/// driver's interpreter releases, one per machine poll.
fn report_steps(py: Python<'_>, harness: &Bound<'_, PyModule>, payload: &Bytes) {
    for native in [Native::Ready, Native::Pending] {
        let before = release_count();
        let (chunks, awaits): (usize, usize) = harness
            .call_method1("count_steps", (handed(py, native, payload),))
            .expect("the counting drain should finish")
            .extract()
            .expect("count_steps returns (chunks, awaits)");
        let resumes = release_count() - before;
        eprintln!(
            "native/{}: chunks={chunks} awaits={awaits} resumes={resumes}",
            native.label()
        );
        assert_eq!(chunks, CHUNKS);
        let expected_awaits = match native {
            Native::Ready => 0,
            Native::Pending => CHUNKS,
            Native::Sync => unreachable!("sync mode never suspends"),
        };
        assert_eq!(
            awaits,
            expected_awaits,
            "native/{} suspended unexpectedly",
            native.label()
        );
    }
}

fn bench_modes<M: Measurement>(c: &mut Criterion<M>, name: &str) {
    Python::initialize();
    Python::attach(|py| {
        let harness = harness(py);
        let payload = delta_frame();
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Elements(CHUNKS as u64));
        for native in [Native::Ready, Native::Pending, Native::Sync] {
            group.bench_function(BenchmarkId::new("native", native.label()), |b| {
                b.iter(|| black_box(consume_native(py, &harness, native, &payload)));
            });
        }
        for (label, pending) in [("ready", false), ("pending", true)] {
            group.bench_function(BenchmarkId::new("python", label), |b| {
                b.iter(|| black_box(consume_python(py, &harness, &payload, pending)));
            });
        }
        group.finish();
    });
}

fn stream_bridge(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| report_steps(py, &harness(py), &delta_frame()));
    bench_modes(c, "stream_bridge");
}

fn stream_bridge_cpu(c: &mut Criterion<ProcessCpu>) {
    bench_modes(c, "stream_bridge_cpu");
}

fn chunk_bytes(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let harness = harness(py);
        let payloads = [
            ("delta_frame", delta_frame()),
            ("1_KiB", Bytes::from(vec![b'x'; 1024])),
            ("16_KiB", Bytes::from(vec![b'x'; 16 * 1024])),
            ("256_KiB", Bytes::from(vec![b'x'; 256 * 1024])),
        ];
        let mut group = c.benchmark_group("chunk_bytes");
        for (label, payload) in &payloads {
            group.throughput(Throughput::Bytes((payload.len() * CHUNKS) as u64));
            group.bench_with_input(
                BenchmarkId::new("native/ready", label),
                payload,
                |b, payload| {
                    b.iter(|| black_box(consume_native(py, &harness, Native::Ready, payload)));
                },
            );
        }
        group.finish();
    });
}

/// Process CPU time across every thread, read through `time.process_time_ns` so the bench
/// needs no platform bindings of its own.
struct ProcessCpu;

fn process_cpu_now() -> Duration {
    Python::attach(|py| {
        let nanos: u64 = py
            .import("time")
            .and_then(|time| time.call_method0("process_time_ns"))
            .and_then(|nanos| nanos.extract())
            .expect("time.process_time_ns should be readable");
        Duration::from_nanos(nanos)
    })
}

impl Measurement for ProcessCpu {
    type Intermediate = Duration;
    type Value = Duration;

    fn start(&self) -> Duration {
        process_cpu_now()
    }

    fn end(&self, started: Duration) -> Duration {
        process_cpu_now().saturating_sub(started)
    }

    fn add(&self, left: &Duration, right: &Duration) -> Duration {
        *left + *right
    }

    fn zero(&self) -> Duration {
        Duration::ZERO
    }

    fn to_f64(&self, value: &Duration) -> f64 {
        value.as_nanos() as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &CpuFormatter
    }
}

struct CpuFormatter;

fn nanos_unit(typical: f64) -> (f64, &'static str) {
    if typical < 1e3 {
        (1.0, "cpu ns")
    } else if typical < 1e6 {
        (1e-3, "cpu µs")
    } else if typical < 1e9 {
        (1e-6, "cpu ms")
    } else {
        (1e-9, "cpu s")
    }
}

impl ValueFormatter for CpuFormatter {
    fn scale_values(&self, typical: f64, values: &mut [f64]) -> &'static str {
        let (factor, unit) = nanos_unit(typical);
        for value in values.iter_mut() {
            *value *= factor;
        }
        unit
    }

    fn scale_throughputs(
        &self,
        _: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        let (count, per_unit, unit) = match *throughput {
            Throughput::Elements(elements) => (elements as f64, 1e3, "Kelem/cpu s"),
            Throughput::Bytes(bytes) => (bytes as f64, 1024.0 * 1024.0, "MiB/cpu s"),
            Throughput::BytesDecimal(bytes) => (bytes as f64, 1e6, "MB/cpu s"),
            Throughput::Bits(bits) => (bits as f64, 1e6, "Mb/cpu s"),
            Throughput::ElementsAndBytes { elements, .. } => (elements as f64, 1e3, "Kelem/cpu s"),
        };
        for value in values.iter_mut() {
            *value = count * 1e9 / *value / per_unit;
        }
        unit
    }

    fn scale_for_machines(&self, _: &mut [f64]) -> &'static str {
        "ns"
    }
}

criterion_group!(wall, stream_bridge, chunk_bytes);
criterion_group! {
    name = cpu;
    config = Criterion::default().with_measurement(ProcessCpu);
    targets = stream_bridge_cpu
}
criterion_main!(wall, cpu);
