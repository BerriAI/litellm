use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litellm_token_counter::TokenCounter;
use tokenizers::Tokenizer;

const TOKENIZER_JSON: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
));
const FULL_CONTEXT_TOKENS: usize = 1_000_000;
const LARGE_PROMPT_UNIT: &str = "The quick brown fox jumps over the lazy dog. 0123456789\n";

fn full_context_input(tokenizer: &Tokenizer) -> String {
    let unit_tokens = tokenizer
        .encode_fast(LARGE_PROMPT_UNIT, false)
        .expect("reference tokenizer should encode")
        .len();
    let input = LARGE_PROMPT_UNIT.repeat(FULL_CONTEXT_TOKENS.div_ceil(unit_tokens));
    let actual_tokens = tokenizer
        .encode_fast(input.as_str(), true)
        .expect("reference tokenizer should encode")
        .len();
    assert!((FULL_CONTEXT_TOKENS..FULL_CONTEXT_TOKENS + unit_tokens).contains(&actual_tokens));
    input
}

fn inputs(tokenizer: &Tokenizer) -> Vec<(&'static str, String)> {
    vec![
        (
            "ascii_chat",
            "User: Summarize the benefits of statistical benchmarking.\nAssistant:".repeat(8),
        ),
        (
            "unicode_nfkc",
            "Ａ quick café résumé: مرحبا 世界 🙂 ﬁ Ⅳ.\n".repeat(16),
        ),
        ("large_prompt", LARGE_PROMPT_UNIT.repeat(256)),
        ("full_context_1m_tokens", full_context_input(tokenizer)),
    ]
}

fn token_counter(c: &mut Criterion) {
    let counter = TokenCounter::from_json(TOKENIZER_JSON).expect("token counter should load");
    let tokenizer = TOKENIZER_JSON
        .parse::<Tokenizer>()
        .expect("reference tokenizer should load");
    let mut group = c.benchmark_group("anthropic_token_counter");

    for (name, input) in inputs(&tokenizer) {
        let expected = tokenizer
            .encode_fast(input.as_str(), true)
            .expect("reference tokenizer should encode")
            .len();
        let actual = counter
            .count_text(input.as_str())
            .expect("benchmark path should count");
        assert_eq!(
            actual, expected,
            "benchmark paths should produce the same count"
        );
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("byte_level_fast_path", name),
            &input,
            |b, input| {
                b.iter(|| {
                    counter
                        .count_text(black_box(input.as_str()))
                        .expect("fast path should count")
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("full_encoder", name),
            &input,
            |b, input| {
                b.iter(|| {
                    tokenizer
                        .encode_fast(black_box(input.as_str()), true)
                        .expect("reference tokenizer should encode")
                        .len()
                })
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4));
    targets = token_counter
}
criterion_main!(benches);
