use criterion::{Criterion, criterion_group, criterion_main};
use litellm_cost::{
    Pricing, PromptConvention, Rate, Rates, Request, ServiceTier, ThresholdPolicy, ThresholdRates,
    Usage, calculate, compile,
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
    let plan = compile(&pricing).unwrap();
    c.bench_function("native_compiled_calculation", |b| {
        b.iter(|| black_box(plan.calculate(black_box(&request)).unwrap()))
    });
    c.bench_function("native_full_wrapper", |b| {
        b.iter(|| black_box(calculate(black_box(&pricing), black_box(&request)).unwrap()))
    });
    c.bench_function("native_rate_compilation", |b| {
        b.iter(|| black_box(compile(black_box(&pricing)).unwrap()))
    });
    let threshold = ThresholdRates {
        above_prompt_tokens: 1000,
        standard: Rates {
            input: Rate::Value(0.000004),
            output: Rate::Value(0.000016),
            ..Rates::EMPTY
        },
        tiers: &[],
    };
    let threshold_pricing = Pricing {
        thresholds: &[threshold],
        ..pricing
    };
    let threshold_plan = compile(&threshold_pricing).unwrap();
    c.bench_function("native_threshold_boundary", |b| {
        b.iter(|| black_box(threshold_plan.calculate(black_box(&request)).unwrap()))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
