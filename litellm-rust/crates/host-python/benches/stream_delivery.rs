#[path = "../tests/support/streaming.rs"]
mod streaming;

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use pyo3::{prelude::*, types::PyDict};

const CHUNKS: usize = 256;

fn stream_delivery(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let namespace = PyDict::new(py);
        streaming::install(py, &namespace).expect("stream helper should install");
        py.run(
            c"
import asyncio
import queue
import threading
import time

loop = asyncio.new_event_loop()
requests = queue.SimpleQueue()
shutdown = object()

def deliver():
    while True:
        item = requests.get()
        if item is shutdown:
            return
        value, future = item
        loop.call_soon_threadsafe(future.set_result, value)

delivery_thread = threading.Thread(target=deliver, daemon=True)
delivery_thread.start()

async def consume_native(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

async def deliver_one(value):
    future = loop.create_future()
    requests.put((value, future))
    return await future

async def consume_floor(count):
    return [await deliver_one(value) for value in range(count)]

def measure_native(count):
    start = time.perf_counter_ns()
    loop.run_until_complete(consume_native(count))
    return time.perf_counter_ns() - start

def measure_floor(count):
    start = time.perf_counter_ns()
    loop.run_until_complete(consume_floor(count))
    return time.perf_counter_ns() - start

def spin(stop):
    while not stop.is_set():
        sum(range(1000))

def start_spinner():
    stop = threading.Event()
    thread = threading.Thread(target=spin, args=(stop,), daemon=True)
    thread.start()
    return stop, thread

def stop_spinner(state):
    stop, thread = state
    stop.set()
    thread.join()

def finish():
    requests.put(shutdown)
    delivery_thread.join()
    loop.close()
",
            Some(&namespace),
            Some(&namespace),
        )
        .expect("benchmark helpers should initialize");

        let measure_native = namespace
            .get_item("measure_native")
            .expect("native measurement should exist")
            .expect("native measurement should be set")
            .unbind();
        let measure_floor = namespace
            .get_item("measure_floor")
            .expect("floor measurement should exist")
            .expect("floor measurement should be set")
            .unbind();
        let start_spinner = namespace
            .get_item("start_spinner")
            .expect("spinner starter should exist")
            .expect("spinner starter should be set")
            .unbind();
        let stop_spinner = namespace
            .get_item("stop_spinner")
            .expect("spinner stopper should exist")
            .expect("spinner stopper should be set")
            .unbind();
        let finish = namespace
            .get_item("finish")
            .expect("benchmark cleanup should exist")
            .expect("benchmark cleanup should be set")
            .unbind();

        let mut group = c.benchmark_group("stream_delivery");
        group.throughput(Throughput::Elements(CHUNKS as u64));
        bench_measurement(&mut group, py, "native_stream", &measure_native);
        bench_measurement(&mut group, py, "call_soon_threadsafe_floor", &measure_floor);

        let native_spinner = start_spinner
            .bind(py)
            .call0()
            .expect("spinner should start")
            .unbind();
        bench_measurement(
            &mut group,
            py,
            "native_stream_gil_contended",
            &measure_native,
        );
        stop_spinner
            .bind(py)
            .call1((native_spinner,))
            .expect("spinner should stop");

        let floor_spinner = start_spinner
            .bind(py)
            .call0()
            .expect("spinner should start")
            .unbind();
        bench_measurement(
            &mut group,
            py,
            "call_soon_threadsafe_floor_gil_contended",
            &measure_floor,
        );
        stop_spinner
            .bind(py)
            .call1((floor_spinner,))
            .expect("spinner should stop");
        group.finish();
        finish
            .bind(py)
            .call0()
            .expect("benchmark helpers should stop");
    });
}

fn bench_measurement(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    py: Python<'_>,
    name: &str,
    measurement: &Py<PyAny>,
) {
    group.bench_function(name, |bencher| {
        bencher.iter_custom(|iterations| {
            let nanoseconds = (0..iterations)
                .map(|_| {
                    measurement
                        .bind(py)
                        .call1((CHUNKS,))
                        .expect("measurement should complete")
                        .extract::<u64>()
                        .expect("measurement should return nanoseconds")
                })
                .sum();
            Duration::from_nanos(nanoseconds)
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(10));
    targets = stream_delivery
}
criterion_main!(benches);
