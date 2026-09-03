use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use litellm_python_interop::{from_py, to_py, value_to_py};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Value, json};

const PAYLOAD_SIZES: &[(&str, usize)] = &[
    ("1_KiB", 1024),
    ("64_KiB", 64 * 1024),
    ("1_MiB", 1024 * 1024),
    ("4_MiB", 4 * 1024 * 1024),
    ("16_MiB", 16 * 1024 * 1024),
];

const TOOL_CALLS: usize = 18;

fn chat_completion_response() -> Value {
    json!({
        "id": "chatcmpl-9f2c8f0e6b1d4a7fa3c5e2b8d1f0a6c4",
        "object": "chat.completion",
        "created": 1725409200,
        "model": "gpt-5.2",
        "system_fingerprint": "fp_b7c1a9d3e5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here is the summary you asked for. The rollout plan covers three phases. First, the canary fleet moves to the new router. Second, we hold for error-rate parity over two hours. Third, we ramp to full traffic while watching p99 latency. If any guard trips, the router falls back to the previous rule set within one minute, so the blast radius stays bounded to a single shard.",
                    "tool_calls": (0..TOOL_CALLS)
                        .map(|index| {
                            json!({
                                "id": format!("call_0123456789abcdef{index:02}"),
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"city\": \"San Francisco\", \"unit\": \"celsius\"}",
                                },
                            })
                        })
                        .collect::<Vec<_>>(),
                },
                "logprobs": null,
                "finish_reason": "tool_calls",
            },
            {
                "index": 1,
                "message": {
                    "role": "assistant",
                    "content": null,
                },
                "logprobs": null,
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 1024,
            "completion_tokens": 8192,
            "total_tokens": 9216,
            "prompt_tokens_details": {"cached_tokens": 512},
            "completion_tokens_details": {"reasoning_tokens": 2048},
        },
    })
}

fn former_json_roundtrip_from_py(py: Python<'_>, value: &Bound<'_, PyAny>) -> Value {
    let json = py.import("json").expect("Python json module should import");
    let encoded: String = json
        .call_method1("dumps", (value,))
        .expect("payload should serialize")
        .extract()
        .expect("json.dumps should return a string");
    serde_json::from_str(&encoded).expect("serialized JSON should parse")
}

fn pythonize_from_py(value: &Bound<'_, PyAny>) -> Value {
    from_py(value).expect("payload should depythonize")
}

fn former_json_roundtrip_to_py(py: Python<'_>, value: &Value) -> Py<PyAny> {
    let json = py.import("json").expect("Python json module should import");
    let encoded = serde_json::to_string(value).expect("response should serialize");
    json.call_method1("loads", (encoded,))
        .expect("serialized response should parse in Python")
        .unbind()
}

fn pythonize_to_py(py: Python<'_>, value: &Value) -> Py<PyAny> {
    to_py(py, value).expect("response should pythonize")
}

fn interned_to_py(py: Python<'_>, value: &Value) -> Py<PyAny> {
    value_to_py(py, value).expect("response should convert")
}

fn bridge_serialization(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let chat_response = chat_completion_response();
        let chat_size = serde_json::to_string(&chat_response)
            .expect("chat response should serialize")
            .len();
        assert!(
            (3500..=4500).contains(&chat_size),
            "chat benchmark payload should stay near 4 KiB, was {chat_size} bytes"
        );

        c.bench_with_input(
            BenchmarkId::new(
                "rust_to_python_pythonize_chat",
                format!("{chat_size}_bytes"),
            ),
            &chat_response,
            |b, response| b.iter(|| pythonize_to_py(py, black_box(response))),
        );
        c.bench_with_input(
            BenchmarkId::new("rust_to_python_interned_chat", format!("{chat_size}_bytes")),
            &chat_response,
            |b, response| b.iter(|| interned_to_py(py, black_box(response))),
        );

        for &(label, payload_bytes) in PAYLOAD_SIZES {
            let data_uri = format!("data:image/png;base64,{}", "A".repeat(payload_bytes));
            let document = PyDict::new(py);
            document
                .set_item("type", "image_url")
                .expect("document type should be set");
            document
                .set_item("image_url", &data_uri)
                .expect("document URL should be set");
            let response = json!({
                "pages": [{
                    "index": 0,
                    "markdown": "OCR text",
                    "images": [{"image_base64": data_uri}],
                }],
                "model": "mistral-ocr-latest",
                "document_annotation": null,
                "usage_info": {"pages_processed": 1},
                "object": "ocr",
            });

            c.bench_with_input(
                BenchmarkId::new("python_to_rust_json", label),
                &document,
                |b, document| {
                    b.iter(|| former_json_roundtrip_from_py(py, black_box(document.as_any())))
                },
            );
            c.bench_with_input(
                BenchmarkId::new("python_to_rust_pythonize", label),
                &document,
                |b, document| b.iter(|| pythonize_from_py(black_box(document.as_any()))),
            );
            c.bench_with_input(
                BenchmarkId::new("rust_to_python_json", label),
                &response,
                |b, response| b.iter(|| former_json_roundtrip_to_py(py, black_box(response))),
            );
            c.bench_with_input(
                BenchmarkId::new("rust_to_python_pythonize", label),
                &response,
                |b, response| b.iter(|| pythonize_to_py(py, black_box(response))),
            );
            c.bench_with_input(
                BenchmarkId::new("rust_to_python_interned", label),
                &response,
                |b, response| b.iter(|| interned_to_py(py, black_box(response))),
            );
        }
    });
}

mod media;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--media") {
        media::run();
        return;
    }
    let mut criterion = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
        .configure_from_args();
    bridge_serialization(&mut criterion);
    criterion.final_summary();
}
