//! Throughput of each format on a cached chat completion, and `literal_eval` cost by nesting.
//!
//! `literal_eval/nesting` guards against backtracking: the `py_literal` grammar this parser
//! replaced doubled its time per nested `[` or `{` (105 ms at depth 16), so cost must stay
//! linear in depth for every container shape.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litellm_python_compat::{Value, json, literal::literal_eval, pickle, repr::repr};

/// `str(entry)` for the `{timestamp, response}` envelope Python's sync Redis path writes.
fn cached_completion() -> String {
    let choices: Vec<String> = (0..4)
        .map(|index| {
            format!(
                "{{'finish_reason': 'stop', 'index': {index}, 'message': {{'content': \
                 'Benchmarks compare the same workload under controlled conditions, so a \
                 change in time reflects the code rather than the environment. café 日本 \
                 {index}', 'role': 'assistant', 'tool_calls': None, 'function_call': None}}, \
                 'logprobs': None}}"
            )
        })
        .collect();
    format!(
        "{{'timestamp': 1726000000.123, 'response': {{'id': 'chatcmpl-9x1', 'created': \
         1726000000, 'model': 'gpt-4o-2024-08-06', 'object': 'chat.completion', \
         'system_fingerprint': 'fp_1', 'choices': [{}], 'usage': {{'completion_tokens': 120, \
         'prompt_tokens': 42, 'total_tokens': 162, 'completion_tokens_details': None}}}}}}",
        choices.join(", ")
    )
}

fn formats(c: &mut Criterion) {
    let text = cached_completion();
    let value = literal_eval(&text).expect("benchmark payload is a literal");
    let pickled = pickle::dumps(&value).expect("benchmark payload pickles");
    let dumped = json::dumps(&value).expect("benchmark payload is JSON serializable");

    let mut group = c.benchmark_group("cached_completion");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("literal_eval", |b| {
        b.iter(|| literal_eval(black_box(&text)))
    });
    group.bench_function("repr", |b| b.iter(|| repr(black_box(&value))));
    group.bench_function("json_dumps", |b| b.iter(|| json::dumps(black_box(&value))));
    group.bench_function("to_json", |b| b.iter(|| json::to_json(black_box(&value))));
    group.bench_function("from_json", |b| {
        b.iter_batched(
            || serde_json::from_str::<serde_json::Value>(&dumped).expect("dumps output parses"),
            json::from_json,
            criterion::BatchSize::SmallInput,
        )
    });
    group.bench_function("pickle_dumps", |b| {
        b.iter(|| pickle::dumps(black_box(&value)))
    });
    group.bench_function("pickle_loads", |b| {
        b.iter(|| pickle::loads(black_box(&pickled)))
    });
    group.finish();
}

fn nested(open: &str, close: &str, depth: usize) -> String {
    format!("{}1{}", open.repeat(depth), close.repeat(depth))
}

fn literal_nesting(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_eval/nesting");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for depth in [4, 16, 64, 128] {
        for (shape, open, close) in [
            ("list", "[", "]"),
            ("dict", "{'a': ", "}"),
            ("tuple", "(", ",)"),
        ] {
            let text = nested(open, close, depth);
            assert!(matches!(
                literal_eval(&text),
                Ok(Value::List(_) | Value::Dict(_) | Value::Tuple(_))
            ));
            group.bench_with_input(BenchmarkId::new(shape, depth), &text, |b, text| {
                b.iter(|| literal_eval(black_box(text)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, formats, literal_nesting);
criterion_main!(benches);
