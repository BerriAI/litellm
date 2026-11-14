use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use litellm_host_python::{from_py, json_to_py, to_py};
use pyo3::{
    prelude::*,
    types::{PyDict, PyString},
};
use serde_json::{Value, json};

const PAYLOAD_SIZES: &[(&str, usize)] = &[
    ("1_KiB", 1024),
    ("64_KiB", 64 * 1024),
    ("1_MiB", 1024 * 1024),
    ("4_MiB", 4 * 1024 * 1024),
    ("16_MiB", 16 * 1024 * 1024),
];

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

fn bridge_serialization(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
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
        }
    });
}

const MESSAGES_INPUT_TEXTS: &[(&str, &str)] = &[
    (
        "ascii",
        "Here is some ordinary English text about the code.\n",
    ),
    (
        "mixed",
        "Here is some ordinary English text about the code. Done \u{2705} caf\u{e9} \u{1f600}\n",
    ),
    (
        "unicode",
        "\u{65e5}\u{672c}\u{8a9e}\u{306e}\u{30c6}\u{30ad}\u{30b9}\u{30c8} \u{1f600}\u{1f680} caf\u{e9}\n",
    ),
];

fn messages_body(text: &str) -> Value {
    let turn = text.repeat(6);
    let messages: Vec<Value> = (0..)
        .map(|index| {
            json!([
                {"role": "user", "content": [{"type": "text", "text": turn}]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": turn},
                    {"type": "tool_use", "id": format!("toolu_{index:06}"), "name": "bash", "input": {"command": "ls"}}
                ]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": format!("toolu_{index:06}"), "content": turn}]}
            ])
        })
        .scan(0, |size, turns| {
            *size += turns.to_string().len();
            (*size < 210 * 1024).then_some(turns)
        })
        .flat_map(|turns| turns.as_array().cloned().unwrap_or_default())
        .collect();
    json!({"model": "claude-sonnet-5", "max_tokens": 4096, "messages": messages, "stream": true})
}

fn messages_input(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let json = py.import("json").expect("Python json module should import");
        for &(label, text) in MESSAGES_INPUT_TEXTS {
            let body = messages_body(text);
            let dict = json_to_py(py, &body).expect("body should convert");
            c.bench_with_input(
                BenchmarkId::new("messages_input_python_json_dumps", label),
                &dict,
                |b, dict| {
                    b.iter(|| {
                        json.call_method1("dumps", (black_box(dict),))
                            .expect("body should serialize")
                    })
                },
            );
            c.bench_with_input(
                BenchmarkId::new("messages_input_rust_raw_json", label),
                &body,
                |b, body| {
                    b.iter(|| {
                        let bytes = litellm_host::raw_json::to_vec(black_box(body))
                            .expect("body should serialize");
                        PyString::new(py, std::str::from_utf8(&bytes).expect("JSON is UTF-8"))
                    })
                },
            );
        }
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4));
    targets = bridge_serialization, messages_input
}
criterion_main!(benches);
