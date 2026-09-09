use std::ffi::CString;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::task::Poll;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use futures_util::stream::poll_fn;
use litellm_python_interop::{AsyncByteStream, ByteStreamReader, PythonByteStream, to_py_bytes};
use pyo3::prelude::*;
use pyo3::types::{PyBytesMethods, PyModule};
use serde_json::json;

const SIZES: [usize; 4] = [64, 256, 4096, 65536];
const CHUNKS: usize = 256;

fn source(payload: &Bytes, pending: bool) -> PythonByteStream {
    let mut chunks = vec![payload.clone(); CHUNKS].into_iter();
    let mut yielded = false;
    Box::pin(poll_fn(move |cx| {
        if pending && !yielded && chunks.len() > 0 {
            yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        yielded = false;
        Poll::Ready(chunks.next().map(Ok))
    }))
}

fn bench_streaming(c: &mut Criterion) {
    Python::initialize();
    let runtime = pyo3_async_runtimes::tokio::get_runtime();
    let fixture = Python::attach(|py| {
        let source = CString::new(include_str!("support/streaming.py")).unwrap();
        PyModule::from_code(py, &source, c"streaming_bench.py", c"streaming_bench")
            .unwrap()
            .unbind()
    });
    for size in SIZES {
        let payload = Bytes::from(vec![0x61; size]);
        let mut conversion = c.benchmark_group("conversion");
        conversion.throughput(Throughput::Bytes(size as u64));
        Python::attach(|py| {
            for legacy in [false, true] {
                conversion.bench_with_input(
                    BenchmarkId::new(if legacy { "legacy_vec" } else { "direct" }, size),
                    &payload,
                    |b, bytes| {
                        b.iter(|| {
                            let result = if legacy {
                                let temporary = black_box(bytes).to_vec();
                                to_py_bytes(py, &temporary)
                            } else {
                                to_py_bytes(py, black_box(bytes))
                            };
                            assert_eq!(result.as_bytes().len(), size);
                            black_box(result);
                        });
                    },
                );
            }
        });
        conversion.finish();
        for pending in [false, true] {
            let mode = if pending { "pending" } else { "ready" };
            let mut reader = c.benchmark_group("reader");
            reader.throughput(Throughput::Bytes((size * CHUNKS) as u64));
            reader.bench_with_input(BenchmarkId::new(mode, size), &payload, |b, bytes| {
                b.iter_batched(
                    || ByteStreamReader::new(source(bytes, pending), Box::pin(async { Ok(()) })),
                    |reader| {
                        let counts = runtime.block_on(async {
                            let mut count = 0;
                            let mut total = 0;
                            while let Some(bytes) = reader.next_chunk().await.unwrap() {
                                count += 1;
                                total += black_box(bytes).len();
                            }
                            (count, total)
                        });
                        assert_eq!(counts, (CHUNKS, CHUNKS * size));
                    },
                    BatchSize::PerIteration,
                );
            });
            reader.finish();
            let mut python = c.benchmark_group("python_async_for");
            python.throughput(Throughput::Bytes((size * CHUNKS) as u64));
            python.bench_with_input(BenchmarkId::new(mode, size), &payload, |b, bytes| {
                b.iter_batched(
                    || {
                        Python::attach(|py| {
                            Py::new(
                                py,
                                AsyncByteStream::new(
                                    source(bytes, pending),
                                    Box::pin(async { Ok(()) }),
                                ),
                            )
                            .unwrap()
                        })
                    },
                    |iterator| {
                        Python::attach(|py| {
                            let fixture = fixture.bind(py);
                            let coroutine = fixture
                                .getattr("consume")
                                .unwrap()
                                .call1((iterator,))
                                .unwrap();
                            let counts: (usize, usize) = fixture
                                .getattr("loop")
                                .unwrap()
                                .call_method1("run_until_complete", (coroutine,))
                                .unwrap()
                                .extract()
                                .unwrap();
                            assert_eq!(counts, (CHUNKS, CHUNKS * size));
                        })
                    },
                    BatchSize::PerIteration,
                );
            });
            python.finish();
        }
    }
    Python::attach(|py| {
        fixture
            .bind(py)
            .getattr("loop")
            .unwrap()
            .call_method0("close")
            .unwrap();
    });
}

fn summarize(directory: &Path, started: SystemTime) {
    let mut rows =
        String::from("benchmark,ns_per_workload,ns_per_chunk,chunks_per_second,bytes_per_second\n");
    for (group, modes, chunks) in [
        ("conversion", ["direct", "legacy_vec"], 1),
        ("reader", ["ready", "pending"], CHUNKS),
        ("python_async_for", ["ready", "pending"], CHUNKS),
    ] {
        for mode in modes {
            for size in SIZES {
                let id = format!("{group}/{mode}/{size}");
                let path = directory.join(&id).join("new/estimates.json");
                let Ok(metadata) = path.metadata() else {
                    continue;
                };
                if metadata.modified().unwrap() < started {
                    continue;
                }
                let estimates: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                let ns = estimates["mean"]["point_estimate"].as_f64().unwrap();
                let per_chunk = ns / chunks as f64;
                rows.push_str(&format!(
                    "{id},{ns:.3},{per_chunk:.3},{:.3},{:.3}\n",
                    1e9 / per_chunk,
                    size as f64 * 1e9 / per_chunk
                ));
            }
        }
    }
    if rows.lines().count() > 1 {
        let path = directory.join("streaming-summary.csv");
        std::fs::write(&path, rows).unwrap();
        eprintln!("Handoff overhead summary: {}", path.display());
    }
}

fn main() {
    let started = SystemTime::now();
    Python::initialize();
    let directory = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
        .join("criterion");
    std::fs::create_dir_all(&directory).unwrap();
    let metadata = Python::attach(|py| {
        let sys = py.import("sys").unwrap();
        let sysconfig = py.import("sysconfig").unwrap();
        let rustc = Command::new("rustc").arg("-Vv").output().unwrap();
        json!({
            "python": sys.getattr("version").unwrap().extract::<String>().unwrap(),
            "python_executable": sys.getattr("executable").unwrap().extract::<String>().unwrap(),
            "pyo3_python": std::env::var_os("PYO3_PYTHON").map(|value| value.to_string_lossy().into_owned()),
            "gil_disabled_build": sysconfig.call_method1("get_config_var", ("Py_GIL_DISABLED",)).unwrap().extract::<Option<i64>>().unwrap().unwrap_or(0),
            "gil_enabled": sys.getattr("_is_gil_enabled").and_then(|function| function.call0()).and_then(|value| value.extract::<bool>()).unwrap_or(true),
            "rustc": String::from_utf8(rustc.stdout).unwrap(),
            "architecture": std::env::consts::ARCH,
            "os": std::env::consts::OS,
            "abi_target": "abi3-py310 (run with --features pyo3/abi3-py310)",
            "chunks_per_stream": CHUNKS,
            "chunk_sizes": SIZES,
            "arguments": std::env::args().collect::<Vec<_>>(),
        })
    });
    let metadata_path = directory.join(format!(
        "streaming-metadata-{}.json",
        started
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    eprintln!("Streaming handoff metadata: {}", metadata_path.display());
    let mut criterion = Criterion::default()
        .output_directory(&directory)
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();
    bench_streaming(&mut criterion);
    criterion.final_summary();
    summarize(&directory, started);
}
