use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_token_counter::{CountableRequest, TokenCounter};

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

#[derive(Clone, Copy)]
struct AllocationCount {
    allocations: usize,
    bytes: usize,
}

impl AllocationCount {
    fn assert_max(self, label: &str, maximum: Self) {
        eprintln!(
            "{label}: {} allocations, {} bytes",
            self.allocations, self.bytes
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

const TOKENIZER_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
));
const OBJECT_BODY: &[u8] = br#"{"model":"claude-sonnet-4-5","input":{"text":"caf\u00e9","n":3,"ok":true,"list":[1,"a",{"z":[]}]}}"#;
const INTEGER_BODY: &[u8] = br#"{"input":[-9223372036854775808,0,18446744073709551615]}"#;

fn measure(operation: impl FnOnce()) -> AllocationCount {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    operation();
    AllocationCount {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        bytes: BYTES.load(Ordering::Relaxed),
    }
}

fn main() {
    measure(|| {
        black_box(CountableRequest::parse(OBJECT_BODY).expect("object request parses"));
    })
    .assert_max(
        "parse object request",
        AllocationCount {
            allocations: 16,
            bytes: 1_900,
        },
    );

    let counter = TokenCounter::from_json(TOKENIZER_JSON).expect("tokenizer loads");
    let object = CountableRequest::parse(OBJECT_BODY).expect("object request parses");
    counter
        .count_request(&object)
        .expect("object warmup succeeds");
    measure(|| {
        black_box(
            counter
                .count_request(black_box(&object))
                .expect("object counts"),
        );
    })
    .assert_max(
        "count object request",
        AllocationCount {
            allocations: 74,
            bytes: 2_200,
        },
    );

    let integers = CountableRequest::parse(INTEGER_BODY).expect("integer request parses");
    counter
        .count_request(&integers)
        .expect("integer warmup succeeds");
    measure(|| {
        black_box(
            counter
                .count_request(black_box(&integers))
                .expect("integers count"),
        );
    })
    .assert_max(
        "count integer list",
        AllocationCount {
            allocations: 26,
            bytes: 1_050,
        },
    );
}
