use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use litellm_python_interop::{from_py, to_py};
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

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4));
    targets = bridge_serialization
}
criterion_main!(benches);
