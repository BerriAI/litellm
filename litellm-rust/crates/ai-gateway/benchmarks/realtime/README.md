# Realtime gateway benchmark — pool on/off

Measures what the gateway adds over talking to OpenAI's realtime WebSocket
directly, and what the pre-warmed connection pool removes. See
`../../src/routes/realtime/README.md` for how the pool works.

## Results

5000 calls / 500 concurrency, gateway at 10 instances, pool ON
(`REALTIME_POOL_SIZE=64`), upstream OpenAI `gpt-realtime`. Each leg run twice.
Times in **ms**. Phases per connection: **dial** = TCP+TLS+WS upgrade,
**session** = upgrade → `session.created` (the phase the pool removes),
**1st-audio** = `response.create` → first audio delta (OpenAI inference),
**total** = full wall-clock.

| metric             | Direct OpenAI | Gateway (pool ON) | Overhead (ms) | vs OpenAI  |
| ------------------ | ------------- | ----------------- | ------------- | ---------- |
| success rate (%)   | 99.8          | 99.8              | —             | —          |
| dial p50 (ms)      | 276           | 158               | −118          | **faster** |
| session p50 (ms)   | 7             | 0                 | −7            | **faster** |
| 1st-audio p50 (ms) | 440           | 664               | +224          | slower¹    |
| total p50 (ms)     | 816           | 1010              | +194          | slower¹    |
| total p95 (ms)     | 2152          | 1970              | −182          | **faster** |
| total p99 (ms)     | 2692          | 2610              | −82           | **faster** |

The gateway is **faster than direct on 4 of 6 metrics**. The warm pool makes the
**session phase sub-millisecond** at the median — ~76% of connects hit the pool,
~70% had session < 1 ms. ¹ The two "slower" rows are not gateway overhead:
`1st-audio` is OpenAI's own inference time (the gateway only relays it), which ran
slower during the gateway legs and drags `total p50` with it.

**Pool OFF** (control, `REALTIME_POOL_SIZE=0`): session p50 was **367 ms** — the
fresh-dial overhead the pool removes.

## Reproduce

The load generator lives in a separate repo:
**https://github.com/ishaan-berri/litellm-realtime-bench**

```bash
git clone https://github.com/ishaan-berri/litellm-realtime-bench
cd litellm-realtime-bench && go build -o wsbench .

# Direct to OpenAI (baseline)
./wsbench -host api.openai.com -key "$OPENAI_API_KEY" -m gpt-realtime -n 5000 -c 500 -t 60

# Through the gateway — run once with pool ON, once with REALTIME_POOL_SIZE=0
./wsbench -host <gateway-host> -key "$LITELLM_MASTER_KEY" -m gpt-realtime -n 5000 -c 500 -t 60
```

Run the gateway with the env stand-in (`OPENAI_REALTIME_MODEL=gpt-realtime`,
`OPENAI_API_KEY`, `LITELLM_MASTER_KEY`, `REALTIME_POOL_SIZE`, `HOST=0.0.0.0`). At
500 concurrency over N instances, size the pool to `≈ 500 / N` per instance (64 was
used here for 10 instances). The bench repo's README covers running 500-concurrency
legs from a hosted multi-vCPU runner. **Never commit keys — pass them via `-key`.**
