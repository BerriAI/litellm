#[path = "support/streaming.rs"]
mod streaming;

use std::sync::mpsc::{self, Sender};

use pyo3::{prelude::*, types::PyDict};
use rstest::{fixture, rstest};

struct BlockingPool {
    release: Sender<()>,
}

#[fixture]
#[once]
fn blocking_pool() -> BlockingPool {
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    builder.max_blocking_threads(1);
    pyo3_async_runtimes::tokio::init(builder);
    let (release, release_rx) = mpsc::channel();
    let (started_tx, started_rx) = mpsc::channel();
    runtime().spawn_blocking(move || {
        started_tx
            .send(())
            .expect("blocker startup should be observed");
        let _ = release_rx.recv();
    });
    started_rx.recv().expect("blocking thread should start");
    BlockingPool { release }
}

#[expect(
    clippy::disallowed_methods,
    reason = "the test must occupy its configured blocking pool"
)]
fn runtime() -> &'static tokio::runtime::Runtime {
    pyo3_async_runtimes::tokio::get_runtime()
}

#[rstest]
fn stream_chunks_arrive_while_the_blocking_pool_is_busy(
    #[from(blocking_pool)] blocker: &BlockingPool,
) {
    Python::initialize();
    let outcome = Python::attach(|py| {
        let namespace = PyDict::new(py);
        streaming::install(py, &namespace)?;
        py.run(
            c"
import asyncio

async def consume():
    stream = await native_stream(8)
    return [chunk async for chunk in stream]

received = asyncio.run(asyncio.wait_for(consume(), 5))
assert received == list(range(8))
",
            Some(&namespace),
            Some(&namespace),
        )
    });
    blocker
        .release
        .send(())
        .expect("blocking thread should still be occupied");
    outcome.expect("stream consumption should complete while the pool is busy");
}
