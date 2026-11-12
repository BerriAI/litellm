use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use litellm_host_python::{from_py, json_from_py, json_to_py, to_py};
use pyo3::prelude::*;
use serde_json::{Value, json};

const BODIES: &[(&str, usize)] = &[("2k_tokens", 0), ("50k_tokens", 36)];
const FILE_BYTES: usize = 5_400;

fn tool(name: &str) -> Value {
    json!({
        "name": name,
        "description": format!("{name} a file or pattern in the workspace and return what it finds."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path inside the workspace."},
                "pattern": {"type": "string", "description": "Text or glob to look for."},
                "limit": {"type": "integer", "description": "Most results to return."},
            },
            "required": ["path"],
        },
    })
}

fn turn(index: usize) -> Value {
    if index.is_multiple_of(2) {
        return json!({
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": format!("toolu_{index:04}"),
                "name": "Read",
                "input": {"path": format!("/workspace/litellm/module_{index}.py")},
            }],
        });
    }
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": format!("toolu_{:04}", index - 1),
            "content": "def handler(request):\n    return request\n".repeat(FILE_BYTES / 41),
        }],
    })
}

fn messages_body(turns: usize) -> Value {
    let question = json!({"role": "user", "content": "Which of these files swallow exceptions?"});
    let history: Vec<Value> = (0..turns)
        .map(turn)
        .chain(std::iter::once(question))
        .collect();
    let tools: Vec<Value> = ["Read", "Edit", "Write", "Bash", "Grep", "Glob"]
        .into_iter()
        .map(tool)
        .collect();
    json!({
        "model": "claude-sonnet-5-5",
        "max_tokens": 2048,
        "stream": true,
        "system": [{"type": "text", "text": "You are a careful coding agent. ".repeat(330), "cache_control": {"type": "ephemeral"}}],
        "tools": tools,
        "messages": history,
    })
}

fn bridge_conversions(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let json = py.import("json").expect("json imports");
        for &(label, turns) in BODIES {
            let value = messages_body(turns);
            let text = serde_json::to_string(&value).expect("body serializes");
            let body = json
                .call_method1("loads", (text.as_str(),))
                .expect("body parses in Python");
            let twin = json
                .call_method1("loads", (text.as_str(),))
                .expect("body parses in Python");
            c.bench_with_input(BenchmarkId::new("json_from_py", label), &body, |b, body| {
                b.iter(|| json_from_py(black_box(body)).expect("converts"))
            });
            c.bench_with_input(BenchmarkId::new("json_to_py", label), &value, |b, value| {
                b.iter(|| json_to_py(py, black_box(value)).expect("converts"))
            });
            c.bench_with_input(BenchmarkId::new("depythonize", label), &body, |b, body| {
                b.iter(|| from_py::<Value>(black_box(body)).expect("converts"))
            });
            c.bench_with_input(BenchmarkId::new("pythonize", label), &value, |b, value| {
                b.iter(|| to_py(py, black_box(value)).expect("converts"))
            });
            c.bench_with_input(
                BenchmarkId::new("python_deep_eq", label),
                &(&body, &twin),
                |b, (body, twin)| b.iter(|| black_box(body).eq(black_box(twin)).expect("compares")),
            );
            c.bench_with_input(
                BenchmarkId::new("serde_clone", label),
                &value,
                |b, value| b.iter(|| black_box(value).clone()),
            );
            c.bench_with_input(
                BenchmarkId::new("serde_to_vec", label),
                &value,
                |b, value| b.iter(|| serde_json::to_vec(black_box(value)).expect("serializes")),
            );
        }
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bridge_conversions
}
criterion_main!(benches);
