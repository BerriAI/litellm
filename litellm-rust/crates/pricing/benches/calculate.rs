use criterion::{Criterion, criterion_group, criterion_main};
use litellm_pricing::{
    Pricing, PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, Usage, calculate,
};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let pricing = Pricing {
        standard: Rates {
            input: Rate::Value(0.000002),
            output: Rate::Value(0.000008),
            cache_read: Rate::Value(0.0000005),
            cache_write: Rate::Missing,
            cache_write_1h: Rate::Missing,
        },
        tiers: &[],
        thresholds: &[],
        off_peak: None,
    };
    let request = Request {
        usage: Usage {
            prompt_tokens: 1000,
            completion_tokens: 200,
            cache_read_tokens: 250,
            cache_write_tokens: 0,
            cache_write_5m_tokens: None,
            cache_write_1h_tokens: None,
            prompt_convention: PromptConvention::IncludesCache,
        },
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
    };
    c.bench_function("native_text_cache_calculation", |b| {
        b.iter(|| black_box(calculate(black_box(&pricing), black_box(&request)).unwrap()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
