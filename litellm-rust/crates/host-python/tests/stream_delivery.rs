#[path = "support/streaming.rs"]
mod streaming;

use pyo3::{prelude::*, types::PyDict};
use rstest::rstest;

fn run_scenario(py: Python<'_>, source: &str) {
    let namespace = PyDict::new(py);
    streaming::install(py, &namespace).expect("stream helper should install");
    let source = std::ffi::CString::new(source).expect("scenario source has no null bytes");
    py.run(source.as_c_str(), Some(&namespace), Some(&namespace))
        .expect("stream scenario should pass");
}

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

#[rstest]
#[case::concurrent_streams(
    r#"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

async def main():
    first, second = await asyncio.gather(consume(16), consume(16))
    assert first == list(range(16))
    assert second == list(range(16))

asyncio.run(main())
"#
)]
#[case::cancel_pending_stream(
    r#"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

async def main():
    task = asyncio.create_task(consume(64))
    await asyncio.sleep(0)
    assert not task.done()
    task.cancel("test cancellation")
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("consumer task was not cancelled")
    assert await consume(4) == list(range(4))

asyncio.run(main())
"#
)]
#[case::add_reader_unavailable(
    r#"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

class NoReaderLoop(asyncio.SelectorEventLoop):
    def add_reader(self, fd, callback, *args):
        raise NotImplementedError

loop = NoReaderLoop()
try:
    assert loop.run_until_complete(consume(8)) == list(range(8))
finally:
    loop.close()
"#
)]
#[case::add_reader_unavailable_reuses_loop(
    r#"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

class NoReaderLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__()
        self.reader_attempts = 0

    def add_reader(self, fd, callback, *args):
        self.reader_attempts += 1
        raise NotImplementedError

loop = NoReaderLoop()
try:
    assert loop.run_until_complete(consume(8)) == list(range(8))
    assert loop.run_until_complete(consume(4)) == list(range(4))
    assert loop.reader_attempts == 1
finally:
    loop.close()
"#
)]
#[case::sequential_asyncio_runs(
    r#"
import asyncio

async def consume(count):
    stream = await native_stream(count)
    return [chunk async for chunk in stream]

assert asyncio.run(consume(8)) == list(range(8))
assert asyncio.run(consume(8)) == list(range(8))
"#
)]
#[case::mapped_error_after_pending(
    r#"
import asyncio

async def main():
    stream = await native_error_stream()
    try:
        await anext(stream)
    except RuntimeError as error:
        assert str(error) == "synthetic streaming error"
    else:
        raise AssertionError("stream error was not raised")

asyncio.run(main())
"#
)]
fn streams_preserve_pending_step_behavior(#[case] source: &str) {
    Python::initialize();
    Python::attach(|py| run_scenario(py, source));
}
