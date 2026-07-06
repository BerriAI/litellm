# Compresr Guardrail — query-aware context compression with lossless recovery

[Compresr](https://compresr.ai) compresses bulky message content (tool outputs,
RAG chunks, search results) before the request reaches the LLM, cutting prompt
tokens without losing the information the model actually needs for the current
question.

Two things set it apart from whole-conversation compressors:

- **Query-aware:** each message is compressed against the *intent* that
  produced it — for a tool output, the originating tool call's
  `name + arguments` (resolved via `tool_call_id`); otherwise the last user
  message. Content relevant to the question survives; noise goes.
- **Recoverable, not lossy:** every compressed message carries a
  `compresr hash=<...>` marker and the request gains a `compresr_retrieve`
  tool. If the model finds the compressed version insufficient, it calls the
  tool and LiteLLM's agentic loop transparently feeds the original content
  back — no application changes.

## Quickstart

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o

guardrails:
  - guardrail_name: compresr
    litellm_params:
      guardrail: compresr
      mode: pre_call
      default_on: true
      api_key: os.environ/COMPRESR_API_KEY
```

That's it. Tool/function outputs longer than 500 characters are now compressed
~2x, query-aware, on `/chat/completions`, `/v1/messages`, and `/v1/responses`.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `api_key` | `COMPRESR_API_KEY` env | Compresr API key (`cmp_...`) |
| `api_base` | `https://api.compresr.ai` | Point at your on-prem Compresr deployment |
| `model` | `latte_v2` | Compresr compression model (not the LLM) |
| `unreachable_fallback` | `fail_closed` | `fail_open` forwards uncompressed when Compresr is down |

Optional tuning, under `optional_params`:

| Key | Default | Meaning |
| --- | --- | --- |
| `target_compression_ratio` | `0.5` | 0–1 = fraction of tokens to remove; >1 = Nx reduction factor |
| `coarse` | `true` | Paragraph-level (faster) vs token-level compression |
| `min_chars_to_compress` | `500` | Skip messages shorter than this |
| `compress_tool_outputs` | `true` | Compress tool/function results |
| `compress_system` | `false` | Also compress system messages |
| `compress_history` | `false` | Also compress prior user messages |
| `compress_last_user` | `false` | Also compress the last user message |
| `enable_retrieval` | `true` | Inject the `compresr_retrieve` recovery tool |
| `allow_bypass_header` | `false` | Honour `x-compresr-bypass: true` from callers (opt-in; off by default) |
| `max_bytes_per_call` | `10485760` | Max bytes of stored originals per `litellm_call_id` (10 MiB); oldest evicted first |

## Per-request bypass

Send `x-compresr-bypass: true` as a request header to skip compression for a
single call. The header is ignored unless `allow_bypass_header: true` is set in
the guardrail config (off by default to prevent callers from defeating token
accounting or rate-limit strategies).

## Limitations

**Multi-worker deployments:** originals for the recovery tool are stored in
process memory. If your LiteLLM proxy runs with `--workers N > 1` (gunicorn /
uvicorn multi-process), the pre-call hook and post-call agentic hook may land on
different workers — recovery silently fails and the model sees a "hash not found"
message instead of the original content. Set `--workers 1` or disable recovery
with `enable_retrieval: false` in multi-worker setups.

## How recovery works

1. `apply_guardrail` compresses eligible messages and appends a marker:
   `[compresr hash=abc...: parts of this content were compressed away ...]`.
   The original text is kept in-process, scoped to this request's
   `litellm_call_id` (15-minute TTL).
2. A `compresr_retrieve` tool is merged into the request's tools.
3. If the model calls it, `async_build_agentic_loop_plan` answers the tool
   call with the stored original and re-runs the request — one extra LLM
   round-trip, only when the model asks for it.

Hashes are only honored for the request that issued them; a hash pasted in
from another conversation returns a not-found message instead of content.
