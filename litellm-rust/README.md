# litellm-rust

A Rust port of LiteLLM's Mistral OCR transform logic. One pure, shared core
(`src/llms/...`, `src/pipeline.rs`) is exposed two ways:

1. A Python extension module `litellm_rust` (via PyO3) — the SDK/proxy call this.
2. A standalone axum HTTP server binary `litellm-rust-server` serving `POST /v1/ocr`.

The transform modules are pure (no PyO3, no I/O), mirroring the Python
`litellm/llms/` tree:

- `src/llms/base_llm/ocr/transformation.rs` — `BaseOcrConfig` trait + serde
  response types (`OcrResponse`, `OcrPage`, `OcrPageImage`, `OcrUsageInfo`,
  `OcrPageDimensions`) and `OcrRequest`.
- `src/llms/mistral/ocr/transformation.rs` — `MistralOcrConfig`.

## Build the Python extension

PyO3 is an optional dependency behind the `python` feature, so the server binary
never links libpython.

```bash
# from inside litellm-rust/
maturin develop --features python
```

Then from Python:

```python
import litellm_rust

resp = litellm_rust.ocr(
    "mistral-ocr-latest",
    {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
    None,            # api_key (falls back to MISTRAL_API_KEY)
    None,            # api_base (defaults to https://api.mistral.ai/v1)
    {"include_image_base64": True},
)
# -> {"pages": [...], "model": ..., "document_annotation": ..., "usage_info": {...}, "object": "ocr"}
```

## Run the standalone server

The binary builds without the `python` feature:

```bash
# from inside litellm-rust/
PORT=8088 cargo run --bin litellm-rust-server
```

Example request:

```bash
curl -s http://localhost:8088/v1/ocr \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "https://example.com/doc.pdf"},
        "include_image_base64": true
      }'
```

`api_key` may be supplied in the body; if omitted, the server reads
`MISTRAL_API_KEY` from the environment. `GET /health` returns `{"status":"ok"}`.

## Dev

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo clippy --lib --features python -- -D warnings
cargo check --bin litellm-rust-server
```
