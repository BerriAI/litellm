#!/usr/bin/env python3
"""Quick functionality + latency check for all proxy models."""
import json
import os
import time
import urllib.error
import urllib.request

BASE = "http://localhost:4000"
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-sophnet-azure-local")
MODELS = [
    "sophnet-glm-5.1",
    "sophnet-gpt-5.5",
    "sophnet-deepseekv4-pro",
    "sophnet-deepseekv4-flash",
    "sophnet-claude-opus-4-7",
    "azure-gpt-5.5",
    "azure-gpt-5.4",
]
PROMPT = "Reply with exactly one word: OK"


def test_model(model: str) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {MASTER_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        elapsed = time.perf_counter() - t0
        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = (msg.get("content") or "")[:80]
        usage = body.get("usage") or {}
        return {
            "model": model,
            "status": "ok",
            "latency_s": round(elapsed, 2),
            "reply": content.replace("\n", " "),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "upstream_model": body.get("model"),
        }
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        err_body = e.read().decode(errors="replace")[:300]
        try:
            err_json = json.loads(err_body)
            err_msg = err_json.get("error", {}).get("message", err_body)
        except json.JSONDecodeError:
            err_msg = err_body
        return {
            "model": model,
            "status": f"http_{e.code}",
            "latency_s": round(elapsed, 2),
            "error": err_msg[:200],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "model": model,
            "status": "error",
            "latency_s": round(elapsed, 2),
            "error": str(e)[:200],
        }


def main() -> None:
    print(f"Testing {len(MODELS)} models @ {BASE}\n")
    results = []
    for model in MODELS:
        print(f"  testing {model}...", flush=True)
        results.append(test_model(model))

    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]

    print("\n=== Results ===")
    print(f"{'Model':<20} {'Status':<10} {'Latency':>8}  Reply / Error")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x.get("latency_s", 999)):
        if r["status"] == "ok":
            print(
                f"{r['model']:<20} {'OK':<10} {r['latency_s']:>6.2f}s  "
                f"{r.get('reply','')!r}  (tokens: {r.get('prompt_tokens')}+{r.get('completion_tokens')})"
            )
        else:
            print(
                f"{r['model']:<20} {r['status']:<10} {r['latency_s']:>6.2f}s  "
                f"{r.get('error','')[:60]}"
            )

    if ok:
        lats = [r["latency_s"] for r in ok]
        print(
            f"\nSuccess: {len(ok)}/{len(MODELS)}  |  "
            f"Fastest: {min(lats):.2f}s  |  Slowest: {max(lats):.2f}s  |  "
            f"Avg: {sum(lats)/len(lats):.2f}s"
        )
    if fail:
        print(f"Failed:  {len(fail)}/{len(MODELS)}")


if __name__ == "__main__":
    main()
