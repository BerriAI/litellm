// mirrors: crates/cost/tests/generate_python_fixtures.py (catalog embed invariants)

#![allow(clippy::disallowed_types)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_cost::pricing::{Metric, ModelPricing, ServiceTier};
use litellm_cost::wire::RawCatalogEntry;
use rstest::rstest;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn parsed_pricing() -> ModelPricing {
    let entry: RawCatalogEntry = serde_json::from_value(serde_json::json!({
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "cache_read_input_token_cost": 3e-7,
        "cache_creation_input_token_cost_above_1hr": 4e-6,
        "input_cost_per_token_priority": 6e-6,
        "input_cost_per_token_batches": 1.5e-6,
        "input_cost_per_token_above_272k_tokens": 1.5e-6,
        "input_cost_per_token_above_272k_tokens_flex": 7.5e-7,
        "cache_read_input_token_cost_above_272k_tokens_batches": 7.5e-8,
        "provider_specific_entry": {"us": 1.1},
        "search_context_cost_per_query": {"search_context_size_medium": 3e-6},
        "litellm_provider": "synthetic",
        "max_tokens": 8192
    }))
    .expect("synthetic entry parses");
    entry.pricing()
}

#[rstest]
fn model_pricing_lookups_allocate_nothing_on_the_request_path() {
    let pricing = parsed_pricing();
    let tiers = [
        ServiceTier::Priority,
        ServiceTier::Flex,
        ServiceTier::Auto,
        ServiceTier::Fast,
        ServiceTier::Ultrafast,
    ];
    ALLOCATIONS.store(0, Ordering::Relaxed);
    let mut sink = 0.0_f64;
    for metric in [
        Metric::InputPerToken,
        Metric::OutputPerToken,
        Metric::CacheReadToken,
        Metric::CacheCreationToken1hr,
        Metric::OutputReasoningToken,
    ] {
        sink += pricing.rate(metric).value().unwrap_or(0.0);
        for tier in tiers {
            sink += pricing.tier_rate(metric, tier).value().unwrap_or(0.0);
        }
        sink += pricing.batch_rate(metric).value().unwrap_or(0.0);
    }
    for threshold in &pricing.thresholds {
        sink += threshold.above_tokens as f64;
        for metric in [Metric::InputPerToken, Metric::CacheCreationToken1hr] {
            sink += threshold
                .standard
                .get(&metric)
                .and_then(|rate| rate.value())
                .unwrap_or(0.0);
            for tier in tiers {
                sink += threshold
                    .tiers
                    .get(&(metric, tier))
                    .and_then(|rate| rate.value())
                    .unwrap_or(0.0);
            }
        }
    }
    sink += pricing
        .provider_specific_entry
        .get("us")
        .and_then(|rate| rate.value())
        .unwrap_or(0.0);
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0, "sink {sink}");
}
