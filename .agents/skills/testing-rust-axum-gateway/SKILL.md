---
name: testing-rust-axum-gateway
description: How to build, run, and E2E-test the standalone Rust Axum gateway (litellm-ai-gateway), including the OCR routes (/v1/ocr, /ocr) against a live provider, plus the Python baseline proxy for parity.
---

# Testing the LiteLLM Rust Axum gateway (litellm-ai-gateway)

## Build the gateway binary (with config loading)
Loading `model_list` from a proxy YAML requires the `python-config` feature, which links libpython. From `litellm-rust`:

```
PYO3_PYTHON=<repo>/.venv/bin/python \
RUSTFLAGS="-L native=/usr/lib/python3.10/config-3.10-x86_64-linux-gnu" \
cargo build --release -p litellm-ai-gateway --bin litellm-ai-gateway --features "server python-config"
```
- `PYO3_PYTHON` must point at a python with a shared libpython.
- The `RUSTFLAGS` link-search is needed because the dev `libpython3.10.so` symlink lives under that config dir.
- If you hit `undefined symbol: Py...` from a stale extension-module build, run `cargo clean -p pyo3 -p pyo3-ffi` first.

## Python deps the embedded interpreter needs
The gateway's config reader (`litellm.proxy.read_model_list`) and the Python baseline proxy both need `litellm[proxy]`
extras. The bundled `.venv` is often missing them. The `.venv` has no `pip`; use uv WITHOUT touching uv.lock:
```
VIRTUAL_ENV=<repo>/.venv uv pip install orjson apscheduler uvloop python-dotenv \
  gunicorn uvicorn fastapi starlette backoff pyyaml rq fastapi-sso PyJWT python-multipart \
  cryptography pynacl websockets boto3 mcp RestrictedPython rich pydantic-settings expression litellm-proxy-extras
```
Do NOT let `uv run`/`uv sync` rewrite/commit `uv.lock`. Blueprint `uv sync --inexact --frozen` does NOT pull the
proxy extra, so these must be added (ideally `uv sync ... --extra proxy`).

## Minimal config (avoids redis/DB/callbacks)
```yaml
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
model_list:
  - model_name: rust-ocr-mistral
    litellm_params:
      model: mistral/mistral-ocr-latest
      api_key: os.environ/MISTRAL_API_KEY
```

## Run both servers
Gateway (expect boot log `loaded model_list from <path> via python config reader`):
```
LITELLM_MASTER_KEY=sk-e2e-local HOST=127.0.0.1 PORT=4001 LITELLM_CONFIG_PATH=<minimal.yml> \
PYTHONPATH=<repo>/.venv/lib/python3.10/site-packages:<repo> \
./litellm-rust/target/release/litellm-ai-gateway
```
Python baseline:
```
LITELLM_MASTER_KEY=sk-e2e-local PYTHONPATH=<repo>:<repo>/.venv/lib/python3.10/site-packages \
.venv/bin/python litellm/proxy/proxy_cli.py --config <minimal.yml> --port 4000 --host 127.0.0.1
```

## Sanity checks / assertions for OCR
- `GET /health/readiness` -> 200; `POST /v1/ocr` with no bearer -> 401 (fails closed).
- OCR success: HTTP 200, `object=="ocr"`, non-empty `model`, `pages` list len>0 with `pages[0].markdown`, and
  non-empty `usage_info` (Mistral serializes usage as `usage_info`, not `usage`).
- Gateway and Python baseline both echo `model` as the requested alias (e.g. `rust-ocr-mistral`); the gateway
  normalizes the public response `model` back to the alias after the provider call, matching the baseline.
- Never print the provider API key, base64 payloads, raw provider bodies, or full OCR text. Pipe curl through a jq
  projection like `{object, model, pages_len:(.pages|length), md_head:(.pages[0].markdown[0:40]), usage_info}`.

## Devin Secrets Needed
- `MISTRAL_API_KEY` (live Mistral OCR calls)
