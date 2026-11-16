"""Veria 4581540795 at PR2 5796f53: evidence._blocks copies the message envelope (every message field except content)
into each block's identity, so the canonical JSON grows with blocks x envelope size. Offline, bounded inputs (largest
request body about 1 MiB), no provider calls, no source edits. The breakpoint is on the last block so the whole body is
in the prefix. extra_field is a made-up non-content message field carrying the bulk."""
import hashlib, json, subprocess, sys, time, tracemalloc
sys.path.insert(0, "/home/ubuntu/cachezero/wt/pr2")
from litellm.integrations.prompt_cache_alerts import evidence
KEY = evidence.fingerprint_key("fixture-salt")
def body(blocks, extra_bytes, text_bytes=20):
    content = [{"type": "text", "text": "t" * text_bytes} for _ in range(blocks)]
    content[-1]["cache_control"] = {"type": "ephemeral"}
    message = {"role": "user", "content": content, **({"extra_field": "e" * extra_bytes} if extra_bytes else {})}
    return {"model": "claude-sonnet-5-5", "max_tokens": 8, "messages": [message]}
def measure(blocks, extra_bytes, text_bytes=20):
    b = body(blocks, extra_bytes, text_bytes); raw = json.dumps(b)
    segments = evidence._segments(evidence._json_body(b))
    canonical = json.dumps([s.identity for s in segments], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    tracemalloc.start(); t0 = time.perf_counter()
    result = evidence.cacheable_prefix(b, KEY)
    ms = (time.perf_counter() - t0) * 1000; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {"blocks": blocks, "envelope_extra_bytes": extra_bytes, "request_json_bytes": len(raw), "canonical_bytes": len(canonical),
            "ratio": round(len(canonical) / len(raw), 1), "peak_traced_bytes": peak, "ms": round(ms, 1), "result": type(result).__name__}
cases = [measure(100, 0), measure(1000, 0), measure(1000, 0, 1000),
         measure(10, 100_000), measure(100, 100_000), measure(1000, 100_000),
         measure(100, 10_000), measure(1000, 10_000), measure(1000, 1_000_000)]
sha = subprocess.check_output(["git", "-C", "/home/ubuntu/cachezero/wt/pr2", "rev-parse", "HEAD"], text=True).strip()
json.dump({"sha": sha, "cases": cases}, open("measure.json", "w"), indent=1)
for c in cases: print(c)
