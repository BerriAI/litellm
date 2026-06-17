# Plugin architecture POC

A throwaway-quality proof that a thin format-agnostic core plus a plugin
boundary works end-to-end. One real plugin (Mistral OCR) is wired through.
Streaming is intentionally out of scope; this is the easy half so the seam
can be tested before tackling the hard half.

## Layout

```
poc/plugin_arch/
  core/                       Rust thin core (axum + reqwest)
    src/main.rs               peeks model field, forwards bytes
  plugins/mistral_ocr/        Python plugin (separate process)
    base.py                   LLMPlugin + TransformingLLMPlugin
    plugin.py                 MistralOCRPlugin (transform_request, call_upstream, transform_response)
    server.py                 thin HTTP wrapper around any LLMPlugin
  routes.toml                 model -> plugin address
  scripts/run.sh              one-command bring-up
```

## The interface

The core only ever sees this Python-side contract (mirrored over HTTP, so
plugins can be written in any language; the boundary is the wire, not the
import graph). See `plugins/mistral_ocr/base.py`.

```python
class LLMPlugin(ABC):
    def handle(self, request: PluginRequest) -> PluginResponse | PluginError: ...
    def capabilities(self) -> Capabilities: ...

class TransformingLLMPlugin(LLMPlugin):
    def handle(self, request):
        return self.transform_response(self.call_upstream(self.transform_request(request)))
    # subclasses implement transform_request, call_upstream, transform_response
```

`transform_*` are deliberately not on the base interface. Plugins that don't
fit the transform-shape (multi-step auth, entangled signing/call, etc.)
implement `handle` directly and ignore the scaffolding.

### Wire interface (what the core actually speaks)

The plugin process exposes:

- `POST /handle` - raw request bytes in, raw response bytes out (or error
  envelope on failure).
- `GET /capabilities` - `{"models": [...], "endpoints": [...]}`
- `GET /healthz`

Error envelope (returned with the appropriate non-2xx status):

```json
{"error": {"code": "...", "message": "...", "type": "..."}}
```

The core never parses the OCR payload itself. It only extracts the top-level
`model` field to choose a plugin, then carries the bytes through.

## Running it

Prereqs: `cargo` (Rust 1.74+), `python3` (3.11+ for `match` over dataclasses),
and a Mistral API key in your environment.

```bash
cd poc/plugin_arch
export MISTRAL_API_KEY=sk-...
./scripts/run.sh
```

That command builds the Rust core in release mode, starts the Python plugin
on `:8081`, starts the core on `:8080`, and tails both.

### Try it

```bash
curl -sS http://127.0.0.1:8080/v1/capabilities | jq .

curl -sS http://127.0.0.1:8080/v1/handle \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mistral-ocr-latest",
    "document": {
      "type": "document_url",
      "document_url": "https://arxiv.org/pdf/2201.04234"
    },
    "include_image_base64": false
  }' | jq '.pages[0].markdown' | head -c 500
```

## Hard constraints this POC actually tests

- The core has zero knowledge of OCR or Mistral. Grep `core/src/main.rs` for
  "ocr" or "mistral"; there is nothing.
- The plugin does not import any core internals. The plugin runs in its own
  process; it never imports anything from `core/`.
- Mistral errors are mapped into the interface's error envelope rather than
  bubbling raw exceptions across the boundary.
- The API key is read from `MISTRAL_API_KEY`. There is no hardcoded key path.

## Adding a second plugin

1. Implement `LLMPlugin` (or extend `TransformingLLMPlugin`) in a new
   package, e.g. `plugins/openai_chat/`. Write a small `server.py` that
   wraps it the same way the OCR one does (or reuse a future shared
   wrapper). Either Python or any other language that can speak the wire
   interface.
2. Run it on a free port.
3. Add an entry to `routes.toml`:
   ```toml
   [[plugin]]
   name = "openai_chat"
   address = "http://127.0.0.1:8082"
   models = ["gpt-4.1-mini"]
   ```
4. Restart the core. Nothing in `core/` changes.

## What "done" means here

- One command brings up core plus plugin.
- A request through the core round-trips to Mistral and returns a correct
  result.
- Swapping the plugin implementation requires zero changes to the core.
- A second plugin slots in via `LLMPlugin` plus a routing entry.

A green POC means the seam works. It does not mean the architecture is
validated; streaming is the real hard case and the next milestone.
