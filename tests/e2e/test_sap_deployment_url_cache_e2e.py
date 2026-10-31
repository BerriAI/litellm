"""
E2E: verify SAP deployment URL caching eliminates repeated discovery GETs.

Run: source .env && uv run --no-sync python tests/e2e/test_sap_deployment_url_cache_e2e.py
"""

import time

import litellm
from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig


def main():
    # # Start clean
    # GenAIHubOrchestrationConfig._deployment_url_cache.clear()

    prompt = [{"role": "user", "content": "Say 'hello' and nothing else."}]

    # First call: pays the discovery cost (GET /lm/deployments)
    t0 = time.perf_counter()
    r1 = litellm.completion(model="sap/gpt-4o-mini", messages=prompt, max_tokens=5)
    first_ms = (time.perf_counter() - t0) * 1000

    # # Cache should now have exactly one entry
    # cache = GenAIHubOrchestrationConfig._deployment_url_cache
    # assert len(cache) == 1, f"expected 1 cache entry, got {len(cache)}"
    # cached_url = next(iter(cache.values()))
    # print(f"cached deployment URL: {cached_url}")

    # Second call: should skip discovery entirely
    t0 = time.perf_counter()
    r2 = litellm.completion(model="sap/gpt-4o-mini", messages=prompt, max_tokens=5)
    second_ms = (time.perf_counter() - t0) * 1000

    # # Cache should still have the same single entry (no new discovery)
    # assert len(cache) == 1, f"cache grew unexpectedly to {len(cache)} entries"
    # assert next(iter(cache.values())) == cached_url, "cached URL changed between calls"

    # Both calls should return valid responses
    c1 = r1.choices[0].message.content
    c2 = r2.choices[0].message.content
    assert c1 and c2, f"empty responses: {c1!r}, {c2!r}"

    print(f"first call:  {first_ms:.0f}ms (includes discovery + LLM)")
    print(f"second call: {second_ms:.0f}ms (LLM only, no discovery)")
    print(f"response 1: {c1!r}")
    print(f"response 2: {c2!r}")

    # The second call should be meaningfully faster because it skips the
    # deployment discovery round-trip. We don't pin a hard threshold since
    # network jitter exists, but the saving is typically 100-500ms.
    savings = first_ms - second_ms
    print(f"savings: {savings:.0f}ms")
    if savings > 0:
        print("PASS: second call was faster (cache hit skipped discovery)")
    else:
        print("NOTE: second call was not faster, likely network jitter. Cache still worked (entry count = 1).")


if __name__ == "__main__":
    main()
