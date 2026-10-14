#[path = "support/streaming.rs"]
mod streaming;

use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

#[rstest]
fn stream_chunks_do_not_allocate_an_asyncio_future_each() {
    Python::initialize();
    Python::attach(|py| {
        let namespace = PyDict::new(py);
        streaming::install(py, &namespace).expect("stream helper should install");
        py.run(
            c"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

class CountingLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__()
        self.future_count = 0

    def create_future(self):
        self.future_count += 1
        return super().create_future()

def count_futures(chunks):
    loop = CountingLoop()
    try:
        loop.run_until_complete(consume(chunks))
        return loop.future_count
    finally:
        loop.close()

one_chunk_futures = count_futures(1)
sixty_four_chunk_futures = count_futures(64)
",
            Some(&namespace),
            Some(&namespace),
        )
        .expect("streams should complete");
        let one_chunk_futures: usize = namespace
            .get_item("one_chunk_futures")
            .expect("counter should exist")
            .expect("counter should be set")
            .extract()
            .expect("counter should be an integer");
        let sixty_four_chunk_futures: usize = namespace
            .get_item("sixty_four_chunk_futures")
            .expect("counter should exist")
            .expect("counter should be set")
            .extract()
            .expect("counter should be an integer");

        assert_eq!(
            one_chunk_futures, sixty_four_chunk_futures,
            "future allocations grew from {one_chunk_futures} to {sixty_four_chunk_futures}"
        );
    });
}
