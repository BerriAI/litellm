# Headroom compression integration

Wraps [Headroom](https://github.com/chopratejas/headroom)'s `HeadroomCallback`
as a LiteLLM proxy callback (`mt_headroom_aggressive`) that compresses request
messages — chiefly agentic tool-result payloads (Bash/Grep/Glob output) — before
they are sent to the upstream model. On Claude-Code-style traffic this yields
large input-token savings with sub-millisecond classification overhead; the
actual compression runs ModernBERT off the event loop via `asyncio.to_thread`.

## Enabling

Register the callback in the proxy config:

```yaml
litellm_settings:
  callbacks: ["mt_headroom_aggressive"]
```

## Compression toggle

Compression is **opt-out** and controlled at two levels. The callback stays
registered either way — these flags only govern whether compression actually
runs, so it can be flipped per request without a redeploy while the ModernBERT
pipeline stays pre-warmed.

### 1. Global default — `HEADROOM_COMPRESSION_ENABLED`

| Value | Effect |
|-------|--------|
| unset | compression **on** (default) |
| `1` / `true` / `yes` / `on` | compression **on** |
| `0` / `false` / `no` / `off` | compression **off** cluster-wide |

```bash
export HEADROOM_COMPRESSION_ENABLED=0   # disable everywhere
```

### 2. Per-request override — `x-headroom-compress` header

Overrides the global default for a single request. Same truthy/falsy vocabulary.
Absent header → fall back to the env default.

```bash
# Force compression OFF for this call, regardless of the global default:
curl https://<proxy>/v1/messages \
  -H "Authorization: Bearer sk-..." \
  -H "x-headroom-compress: off" \
  -H "Content-Type: application/json" \
  -d '{ "model": "...", "messages": [...] }'
```

```bash
# Force compression ON for this call, even if the global default is off:
  -H "x-headroom-compress: on"
```

Precedence: **header > env default > built-in default (on)**. When compression
is skipped, the `headroom` logger emits an INFO line noting whether the decision
came from the header or the env default, so operators can see per-request why a
call was or wasn't compressed.
