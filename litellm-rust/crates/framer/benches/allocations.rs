use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use bytes::Bytes;
use futures_util::{Stream, TryStreamExt};
use litellm_framing::Framer;
use litellm_framing::aws_event_stream::AwsEventStreamFramer;
use litellm_framing::sse::SseFramer;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: This allocator delegates the unchanged layout to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The pointer and layout came from the delegated `System` allocation.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        // SAFETY: The pointer and layout came from `System`; the new size is unchanged.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const FRAMES: usize = 1_000;
const SPLIT_CHUNK_BYTES: usize = 64;
const DELTA: &str = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" The quick brown fox jumps over the lazy dog and keeps running."}}"#;

#[derive(Clone, Copy)]
struct AllocationCount {
    allocations: usize,
    bytes: usize,
}

impl AllocationCount {
    fn assert_max(self, label: &str, maximum: Self) {
        eprintln!(
            "{label}: {} allocations, {} bytes ({:.1} allocations/frame)",
            self.allocations,
            self.bytes,
            self.allocations as f64 / FRAMES as f64
        );
        assert!(
            self.allocations <= maximum.allocations,
            "{label} allocation count exceeded {}",
            maximum.allocations
        );
        assert!(
            self.bytes <= maximum.bytes,
            "{label} allocated bytes exceeded {}",
            maximum.bytes
        );
    }
}

fn measure(operation: impl FnOnce()) -> AllocationCount {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    operation();
    AllocationCount {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
    }
}

fn bedrock_wire() -> Vec<u8> {
    let message = Message::new(Bytes::from(format!(r#"{{"bytes":"{DELTA}"}}"#)))
        .add_header(Header::new(":event-type", HeaderValue::String("chunk".into())))
        .add_header(Header::new(
            ":content-type",
            HeaderValue::String("application/json".into()),
        ))
        .add_header(Header::new(":message-type", HeaderValue::String("event".into())));
    let mut frame = Vec::new();
    write_message_to(&message, &mut frame).expect("frame encodes");
    frame.repeat(FRAMES)
}

fn sse_wire() -> Vec<u8> {
    format!("event: content_block_delta\ndata: {DELTA}\n\n")
        .into_bytes()
        .repeat(FRAMES)
}

fn chunked(wire: &[u8], chunk_bytes: usize) -> impl Stream<Item = Result<Bytes, io::Error>> + Send {
    let chunks: Vec<Bytes> = wire.chunks(chunk_bytes).map(Bytes::copy_from_slice).collect();
    futures_util::stream::iter(chunks.into_iter().map(Ok))
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime builds");
    let bedrock = bedrock_wire();
    let bedrock_frame_bytes = bedrock.len() / FRAMES;
    let sse = sse_wire();
    let sse_frame_bytes = sse.len() / FRAMES;

    let aligned = chunked(&bedrock, bedrock_frame_bytes);
    measure(|| {
        black_box(
            runtime
                .block_on(AwsEventStreamFramer.frame(aligned).try_collect::<Vec<_>>())
                .expect("aligned bedrock frames decode"),
        );
    })
    .assert_max(
        "aws event stream, one frame per chunk",
        AllocationCount {
            allocations: 16_100,
            bytes: 1_070_000,
        },
    );

    let split = chunked(&bedrock, SPLIT_CHUNK_BYTES);
    measure(|| {
        black_box(
            runtime
                .block_on(AwsEventStreamFramer.frame(split).try_collect::<Vec<_>>())
                .expect("split bedrock frames decode"),
        );
    })
    .assert_max(
        "aws event stream, 64-byte chunks",
        AllocationCount {
            allocations: 16_100,
            bytes: 1_070_000,
        },
    );

    let aligned = chunked(&sse, sse_frame_bytes);
    measure(|| {
        black_box(
            runtime
                .block_on(SseFramer.frame(aligned).try_collect::<Vec<_>>())
                .expect("aligned sse frames decode"),
        );
    })
    .assert_max(
        "sse, one event per chunk",
        AllocationCount {
            allocations: 2_100,
            bytes: 350_000,
        },
    );

    let split = chunked(&sse, SPLIT_CHUNK_BYTES);
    measure(|| {
        black_box(
            runtime
                .block_on(SseFramer.frame(split).try_collect::<Vec<_>>())
                .expect("split sse frames decode"),
        );
    })
    .assert_max(
        "sse, 64-byte chunks",
        AllocationCount {
            allocations: 2_100,
            bytes: 350_000,
        },
    );
}
