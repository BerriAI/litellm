# Image Guard — vLLM image-cap overflow protection for LiteLLM Proxy

A self-contained LiteLLM custom callback that proactively trims oversized
image payloads **before** they hit vLLM-backed multimodal deployments.

Built from production experience at ShanghaiTech University GenAI platform
(LiteLLM v1.106.1 in front of GLM / DeepSeek vLLM clusters, ~9 model groups,
A/B failover groups per model).

## The problem

vLLM (GLM-4.5V / Qwen-VL backends) enforces:

```
At most 20 image(s) may be provided in one prompt. (parameter=image)
```

Agents that accumulate screenshots across long conversations eventually cross
the cap and every subsequent turn fails with HTTP 400. LiteLLM's router then
tries fallback deployments — which have the same cap — so the request dies
anyway.

## What the guard does

Registered as `litellm_settings.callbacks: [image_guard.image_guard_instance]`:

1. **Pre-call hook** — counts images per conversation *unit* (a user turn plus
   its tool chain), evicts oldest whole units first (tool chains stay intact),
   then trims inside the newest unit keeping the *newest* images and replacing
   the oldest ones with a text placeholder.
2. **Failure hook** — parses `At most N image(s)` from upstream 400s and
   caches the learned cap per model (TTL 24h), so a backend with a different
   limit self-tunes.

Covers Chat Completions (`messages`) and Responses API (`input` item lists).

## Critical findings (verified live on GLM vLLM, 2026-11-14)

These cost us a full day to nail down — learn from them:

1. **The cap counts TOTAL images, including base64 data URIs.**
   21 *distinct* base64 images → 400, same as 21 URLs. Our first implementation
   converted oldest URL images to base64 to "bypass" the cap — it does not
   bypass anything. The conversion tier is now disabled by default
   (`IMAGE_GUARD_CONVERT_URLS=0`).

   *Testing trap:* duplicate-content base64 images get deduplicated upstream,
   so `21 × same b64 → 200` will mislead you. Always test with **content-distinct**
   images (e.g. 1×1 PNGs with different pixel colors).

2. **`verbose_proxy_logger.info` output does not reach `docker logs`**
   under the default LiteLLM logging config. The guard writes its audit line
   (`image_guard: model=... deleted=N`) to a probe file instead — see
   `IMAGE_GUARD_PROBE` below.

3. **Invalid base64 is a *separate* 400** (`Non-base64 digit found`), unrelated
   to the image cap. Don't confuse the two when testing.

## Deployment

### Docker Compose

```yaml
services:
  litellm:
    volumes:
      - /data/litellm/image_guard.py:/app/image_guard.py:ro
    environment:
      IMAGE_GUARD_MAX_IMAGES: "20"        # default cap before anything learned
      IMAGE_GUARD_CONVERT_URLS: "0"       # conversion cannot bypass the cap
      IMAGE_GUARD_PROBE: /tmp/guard_probe.log
```

### config.yaml

```yaml
litellm_settings:
  callbacks:
    - image_guard.image_guard_instance
```

### Rollout checklist (learned the hard way)

- Use `docker-compose up -d` after changing volumes/env — `restart` does not
  apply new mounts.
- LiteLLM takes 10–15 s to start; an immediate curl getting connection-refused
  right after restart is a timing issue, not a crash. A crash-loop shows as
  `Up N seconds` repeatedly resetting to zero — check `docker logs` for
  `ImportError: Could not import image_guard_instance`.
- Verify with three requests through a **virtual key with model alias**
  (admin-key tests don't exercise the same path): plain text → 200,
  21 URL images → 200 with `deleted=1` in the probe file,
  mixed 15 URL + 10 distinct base64 → 200 trimmed to 20 total.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `IMAGE_GUARD_MAX_IMAGES` | `20` | Cap used before any limit is learned |
| `IMAGE_GUARD_LIMIT_TTL` | `86400` | Learned-limit cache TTL (seconds) |
| `IMAGE_GUARD_ENABLED` | `1` | Set `0` to disable trimming |
| `IMAGE_GUARD_CONVERT_URLS` | `0` | URL→base64 conversion (off: does not bypass cap) |
| `IMAGE_GUARD_PROBE` | unset | Path for pre-call/audit probe lines |

## Tests

`tests/image_guard/test_image_guard.py` — 30 deterministic tests covering
unit grouping, tool-chain integrity, newest-turn protection, system-message
preservation, base64 counting, single-turn overflow, Responses input, and
limit learning. Run with any Python 3.10+ (imports a tiny litellm shim, no
package install needed).

## License / provenance

Authored by the ShanghaiTech University IT platform team. Provided as-is;
feel free to adapt. Upstream consideration: an optional "max images per
request" guardrail in LiteLLM core would make callbacks like this unnecessary.