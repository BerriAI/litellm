//! Throughput of each format on a cached chat completion, and `literal_eval` cost by nesting.
//!
//! Run one group with `cargo bench -p litellm-python-compat -- cached_completion`, and compare
//! against a stored run with `--save-baseline <name>` / `--baseline <name>`.
//!
//! `literal_eval/nesting` guards against backtracking: the `py_literal` grammar this parser
//! replaced doubled its time per nested `[` or `{` (105 ms at depth 16), so cost must stay
//! linear in depth for every container shape.

use std::{hint::black_box, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
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

/// Every text format, measured against the source bytes it reads or writes.
fn text_formats(group: &mut BenchmarkGroup<'_, WallTime>, text: &str, value: &Value) {
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("literal_eval", |bencher| {
        bencher.iter(|| literal_eval(black_box(text)))
    });
    group.bench_function("repr", |bencher| bencher.iter(|| repr(black_box(value))));
    group.bench_function("json_dumps", |bencher| {
        bencher.iter(|| json::dumps(black_box(value)))
    });
    group.bench_function("to_json", |bencher| {
        bencher.iter(|| json::to_json(black_box(value)))
    });
}

/// Pickle, measured against its own encoding rather than the source text.
fn binary_formats(group: &mut BenchmarkGroup<'_, WallTime>, value: &Value, pickled: &[u8]) {
    group.throughput(Throughput::Bytes(pickled.len() as u64));
    group.bench_function("pickle_dumps", |bencher| {
        bencher.iter(|| pickle::dumps(black_box(value)))
    });
    group.bench_function("pickle_loads", |bencher| {
        bencher.iter(|| pickle::loads(black_box(pickled)))
    });
}

fn formats(c: &mut Criterion) {
    let text = cached_completion();
    let value = literal_eval(&text).expect("benchmark payload is a literal");
    let pickled = pickle::dumps(&value).expect("benchmark payload pickles");
    let dumped = json::dumps(&value).expect("benchmark payload is JSON serializable");

    let mut group = c.benchmark_group("cached_completion");
    text_formats(&mut group, &text, &value);
    binary_formats(&mut group, &value, &pickled);
    // `from_json` consumes its input, so each iteration gets a freshly parsed one.
    group.throughput(Throughput::Bytes(dumped.len() as u64));
    group.bench_function("from_json", |bencher| {
        bencher.iter_batched(
            || serde_json::from_str::<serde_json::Value>(&dumped).expect("dumps output parses"),
            json::from_json,
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// One nesting level of each container shape, as `(name, open, close)`.
const SHAPES: [(&str, &str, &str); 3] = [
    ("list", "[", "]"),
    ("dict", "{'a': ", "}"),
    ("tuple", "(", ",)"),
];

fn literal_nesting(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_eval/nesting");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    for depth in [4, 16, 64, 128] {
        for (shape, open, close) in SHAPES {
            let text = format!("{}1{}", open.repeat(depth), close.repeat(depth));
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(BenchmarkId::new(shape, depth), &text, |bencher, text| {
                bencher.iter(|| literal_eval(black_box(text)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, formats, literal_nesting);
criterion_main!(benches);
