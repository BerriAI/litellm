use std::alloc::{GlobalAlloc, Layout, System};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use bytes::Bytes;
use criterion::measurement::{Measurement, ValueFormatter};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use futures_util::TryStreamExt;
use litellm_framing::Framer;
use litellm_framing::aws_event_stream::AwsEventStreamFramer;
use litellm_framing::sse::SseFramer;
use tokio::runtime::Runtime;

struct CountingAllocator;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: This allocator delegates the unchanged layout to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from the delegated `System` allocation.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: The pointer and layout came from `System`; the new size is unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Allocations;

impl Measurement for Allocations {
    type Intermediate = u64;
    type Value = u64;

    fn start(&self) -> u64 {
        ALLOCATIONS.load(Ordering::Relaxed)
    }

    fn end(&self, start: u64) -> u64 {
        ALLOCATIONS.load(Ordering::Relaxed) - start
    }

    fn add(&self, left: &u64, right: &u64) -> u64 {
        left + right
    }

    fn zero(&self) -> u64 {
        0
    }

    fn to_f64(&self, value: &u64) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &AllocationFormatter
    }
}

struct AllocationFormatter;

impl ValueFormatter for AllocationFormatter {
    fn scale_values(&self, _typical: f64, _values: &mut [f64]) -> &'static str {
        "allocs"
    }

    fn scale_throughputs(
        &self,
        _typical: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        let Throughput::Elements(frames) = throughput else {
            return "allocs";
        };
        for value in values {
            *value /= *frames as f64;
        }
        "allocs/frame"
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "allocs"
    }
}

const FRAMES: usize = 1_000;
const SPLIT_CHUNK_BYTES: usize = 64;
const DELTA: &str = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" The quick brown fox jumps over the lazy dog and keeps running."}}"#;

fn bedrock_wire() -> Vec<u8> {
    let message = Message::new(Bytes::from(format!(r#"{{"bytes":"{DELTA}"}}"#)))
        .add_header(Header::new(
            ":event-type",
            HeaderValue::String("chunk".into()),
        ))
        .add_header(Header::new(
            ":content-type",
            HeaderValue::String("application/json".into()),
        ))
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String("event".into()),
        ));
    let mut frame = Vec::new();
    write_message_to(&message, &mut frame).expect("frame encodes");
    frame.repeat(FRAMES)
}

fn sse_wire() -> Vec<u8> {
    format!("event: content_block_delta\ndata: {DELTA}\n\n")
        .into_bytes()
        .repeat(FRAMES)
}

fn chunked(wire: &[u8], chunk_bytes: usize) -> Vec<Bytes> {
    wire.chunks(chunk_bytes)
        .map(Bytes::copy_from_slice)
        .collect()
}

fn chunkings(wire: &[u8]) -> [(&'static str, usize); 2] {
    [
        ("frame_per_chunk", wire.len() / FRAMES),
        ("64_byte_chunks", SPLIT_CHUNK_BYTES),
    ]
}

fn framing<M: Measurement>(c: &mut Criterion<M>, group_name: &str) {
    let runtime = Runtime::new().expect("runtime builds");
    let bedrock = bedrock_wire();
    let sse = sse_wire();
    let mut group = c.benchmark_group(group_name);
    group.throughput(Throughput::Elements(FRAMES as u64));

    for (label, chunk_bytes) in chunkings(&bedrock) {
        group.bench_with_input(
            BenchmarkId::new("aws_event_stream", label),
            &chunk_bytes,
            |b, &chunk_bytes| {
                b.iter_batched(
                    || chunked(&bedrock, chunk_bytes),
                    |chunks| {
                        let input =
                            futures_util::stream::iter(chunks.into_iter().map(Ok::<_, io::Error>));
                        runtime
                            .block_on(AwsEventStreamFramer.frame(input).try_collect::<Vec<_>>())
                            .expect("bedrock frames decode")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    for (label, chunk_bytes) in chunkings(&sse) {
        group.bench_with_input(
            BenchmarkId::new("sse", label),
            &chunk_bytes,
            |b, &chunk_bytes| {
                b.iter_batched(
                    || chunked(&sse, chunk_bytes),
                    |chunks| {
                        let input =
                            futures_util::stream::iter(chunks.into_iter().map(Ok::<_, io::Error>));
                        runtime
                            .block_on(SseFramer.frame(input).try_collect::<Vec<_>>())
                            .expect("sse events decode")
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn wall_time(c: &mut Criterion) {
    framing(c, "framing_wall_time");
}

fn allocations(c: &mut Criterion<Allocations>) {
    framing(c, "framing_allocations");
}

criterion_group!(name = time; config = Criterion::default(); targets = wall_time);
criterion_group!(name = allocs; config = Criterion::default().with_measurement(Allocations); targets = allocations);
criterion_main!(time, allocs);
