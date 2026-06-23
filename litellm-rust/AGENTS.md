# AGENTS.md — how the Python ↔ Rust OCR bridge works

This explains the wiring between `litellm.ocr()` / `litellm.aocr()` and the Rust
core, so you can extend it without re-deriving the design. For the *rules* of
Rust work here (the pure-transform boundary, the production bar), see
[`CLAUDE.md`](./CLAUDE.md). This file is the *map*.

## The one-paragraph version

`litellm.ocr()` and `litellm.aocr()` are thin Python wrappers. They resolve
credentials and pre-process the document, then hand a typed call to the compiled
`litellm_python_bridge` extension. The bridge parses the Python arguments into a
typed `OcrRequest` (GIL held), runs the whole HTTP call on an async Tokio runtime
with the GIL released, and converts the typed `OcrResponse` back to a dict.
`aocr` returns a Python awaitable; `ocr` blocks on the same future. Everything
provider-specific (URL, headers, request/response transform) is typed Rust — no
`serde_json::Value` blobs except where Python itself is `Any`.

## Crate layout

```
litellm-rust/crates/
  core/          # typed domain model + errors. No I/O, no deps on the others.
    src/ocr/types.rs     OcrRequest, OcrResponse, OcrProvider, OcrDocument, OcrParams, …
    src/error.rs         CoreError (the typed error enum)
  providers/     # the async entry point + pure per-provider transforms.
    src/ocr.rs           `pub async fn ocr(OcrRequest) -> CoreResult<OcrResponse>`  ← re-exported as `litellm_providers::ocr`
    src/mistral/ocr/transformation.rs   complete_url / resolve_api_key / request_body / parse_response (pure)
  python-bridge/ # the PyO3 boundary. The ONLY crate that touches Python objects.
    src/lib.rs           `ocr`, `aocr`, `gil_stats` pyfunctions
    src/gil.rs           GIL-release accounting
```

Boundary (per `CLAUDE.md`): `mistral/ocr/transformation.rs` is **pure** — no
network, no env, no logging. `ocr.rs` is the transport host: it owns the shared
`reqwest` client, the HTTP call, and dispatch. The bridge owns Python interop.

## Naming — it's `ocr` all the way down

| Python | Rust |
|--------|------|
| `litellm.ocr()` | bridge `ocr()` → `litellm_providers::ocr(...)` (`block_on`) |
| `litellm.aocr()` | bridge `aocr()` → `litellm_providers::ocr(...)` (awaited) |
| `MistralOCRConfig.transform_ocr_request/response` | `mistral::request_body` / `mistral::parse_response` |

There is **one** async core function, `litellm_providers::ocr`. The sync and
async bridge entry points are two ways to drive the same future.

## Async model (why there's no `run_in_executor`)

The proxy calls `aocr()`. The old path ran the blocking call in a thread-pool
worker — one OS thread pinned per in-flight OCR for up to 600s, capped at
`min(32, cpu+4)` threads. That bottlenecks under load.

Instead:

- `providers::ocr` is `async` and uses the **async** `reqwest` client.
- `python-bridge`'s `aocr` wraps it with `pyo3_async_runtimes::tokio::future_into_py`,
  returning a Python awaitable. `litellm.aocr()` just `await`s it. The HTTP wait
  happens on a small fixed Tokio worker pool — no thread-per-request.
- `python-bridge`'s `ocr` calls `get_runtime().block_on(...)` inside
  `gil::release_gil` (`py.allow_threads`), so sync SDK callers stay correct and
  other Python threads keep running during the wait.

On the Python side, `litellm.aocr()`'s rust branch returns the *coroutine*
`_arun_rust_ocr(...)`; the existing `aocr` wrapper awaits it on the event loop, so
the executor thread is released before the HTTP wait — not held through it.

## Request lifecycle (one call)

1. **Python shell** (`litellm/ocr/main.py`): detect provider, convert a
   `type:"file"` document to a base64 data-URI, map OCR params, resolve the API
   key via secret managers, run `pre_call` logging. (These stay in Python — see
   "What stays in Python".)
2. **Bridge `ocr`/`aocr`** (`python-bridge/src/lib.rs`): `build_request(...)`
   parses the loose Python args into a typed `OcrRequest` while the GIL is held
   (Python `json` round-trip → serde). Unknown params are dropped.
3. **Core `providers::ocr`** (`providers/src/ocr.rs`): `match request.provider`.
   For Mistral: `resolve_api_key` → `complete_url` → `request_body` (typed,
   serde-serialized) → async POST on the shared client → status check → `parse_response`.
4. **Back across the bridge**: the typed `OcrResponse` is serde-serialized and
   `json.loads`'d into a Python dict; Python wraps it in `OCRResponse.model_validate`.

## The typed model (and the only `Value`s)

`core/src/ocr/types.rs` mirrors the Python `OCRResponse` pydantic models 1:1 as
real Rust types. Serde derives produce the exact wire JSON — no hand-built JSON.
`serde_json::Value` appears in exactly three leaves, only where Python's own type
is `Any`/`Dict[str, Any]`:

- `AnnotationFormat` — a user-supplied JSON Schema we forward verbatim.
- `OcrResponse::document_annotation` — Python `Optional[Any]`.
- `OcrPageImage::bbox` — Python `Optional[Dict[str, Any]]`.

Strict by default: serde ignores unknown upstream fields (vs Python's
`extra="allow"` passthrough).

## Errors

`CoreError` (typed) → `core_error_to_pyerr` in the bridge:

- `Auth` / `UnsupportedProvider` / `InvalidType` / `MissingField` → `ValueError`.
- `Http { status, body }` / `Network` / `InvalidResponse` → `RuntimeError`, with
  the HTTP **status preserved in the message**. The Python `exception_type` layer
  keys off that to raise the right `litellm` exception (RateLimitError on 429,
  AuthenticationError on 401, …). Upstream bodies are truncated to 256 chars so
  document contents/secrets never cross the boundary.

## GIL accounting

`gil_stats()` returns `{"releases": N}` — the count of calls whose HTTP work ran
off the GIL (sync `block_on` releases + async `aocr` offloads). `aocr` work is
off-GIL by construction (Tokio threads); `ocr` uses `py.allow_threads`.

## Adding a provider

1. **core**: the `OcrProvider` enum already lists the variants. Add typed fields
   to `OcrParams` / response types only if the provider needs ones Python models.
2. **providers**: add `providers/src/<provider>/ocr/transformation.rs` with pure
   `complete_url`, key resolution, a typed `request_body` (`#[derive(Serialize)]`),
   and `parse_response` (`#[derive(Deserialize)]` → `OcrResponse`). Unit-test
   param filtering, request shape, response normalization, null/missing, bad input.
3. **providers/src/ocr.rs**: add a `match` arm calling your provider's transforms.
   Non-Bearer auth goes through the `extra_headers` map; pollers (e.g. Azure DocAI)
   use `tokio::time::sleep` in the async path.
4. **python shell** (`litellm/ocr/rust_bridge.py`): add the provider to
   `RUST_SUPPORTED_PROVIDERS`, and its API-key env to `_RUST_PROVIDER_API_KEY_ENV`
   in `main.py`. The shell pre-resolves GCP/Azure tokens and `url→base64` and
   passes them in (Rust does not mint credentials — see `CLAUDE.md`).
5. **tests**: Rust unit tests + Python parity tests (disabled / enabled /
   bridge-unavailable fallback).

## What stays in Python (by design)

`pre_call`/`post_call`, cost & spend tracking, success/failure callbacks, retries,
fallbacks, caching, guardrails, secret-manager and GCP/Azure token minting, and
`exception_type`. These are cross-cutting and FFI/GIL-bound; the Rust core owns
transport + transform + typed errors only.

## Build & test

```bash
cd litellm-rust
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace          # core + providers unit tests
```

Building the Python extension locally (macOS needs the dynamic-lookup flags so
the cdylib can resolve Python symbols at load time):

```bash
RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
  cargo build -p litellm-python-bridge --release
cp target/release/liblitellm_python_bridge.dylib /tmp/litellm_python_bridge.so
# then: PYTHONPATH=/tmp python -c "import litellm, litellm_python_bridge; litellm.use_litellm_rust()"
```

Python-side tests inject a fake bridge (`use_litellm_rust(True, bridge=...)`), so
they run without the compiled wheel: `tests/test_litellm/ocr/test_rust_bridge.py`.
