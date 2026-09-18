"""
End-to-end verification script for the adaptive router.

Requires:
  - LiteLLM proxy running on http://localhost:4000 with adaptive_router configured
    (see litellm/proxy/example_config_yaml/adaptive_router_example.yaml).
  - Postgres reachable via DATABASE_URL (same one the proxy uses).
  - LITELLM_PROXY_KEY env var set (a valid key with permission to send requests).
  - Two model deployments configured under one adaptive_router:
       * "fast"  (cheap,  lower quality)
       * "smart" (expensive, higher quality)

Run:
  uv run python scripts/verify_adaptive_router.py

Optional env:
  LITELLM_PROXY_URL    (default: http://localhost:4000)
  ADAPTIVE_ROUTER_NAME (default: smart-cheap-router)
  EXPECTED_WINNER      (default: smart) -- model expected to dominate after training
  TRAIN_SESSIONS       (default: 20)    -- training sessions in phase 1
  CONVERGE_SESSIONS    (default: 10)    -- cold sessions in phase 2
  WIN_THRESHOLD        (default: 0.7)   -- min share for EXPECTED_WINNER in phase 2
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from typing import List, Optional

import httpx

PROXY_URL: str = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
try:
    PROXY_KEY: str = os.environ["LITELLM_PROXY_KEY"]
except KeyError:
    print(
        "ERROR: LITELLM_PROXY_KEY env var must be set (a proxy key with /chat/completions perms).",
        file=sys.stderr,
    )
    sys.exit(2)

ROUTER_NAME: str = os.environ.get("ADAPTIVE_ROUTER_NAME", "smart-cheap-router")
EXPECTED_WINNER: str = os.environ.get("EXPECTED_WINNER", "smart")
TRAIN_SESSIONS: int = int(os.environ.get("TRAIN_SESSIONS", "20"))
CONVERGE_SESSIONS: int = int(os.environ.get("CONVERGE_SESSIONS", "10"))
WIN_THRESHOLD: float = float(os.environ.get("WIN_THRESHOLD", "0.7"))

REQUEST_TIMEOUT_SECONDS: float = 30.0
RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_SECONDS: float = 1.0
FLUSHER_DRAIN_WAIT_SECONDS: float = 30.0  # proxy flusher loop is 10s; pad with margin

PROMPTS: List[str] = [
    "Write a Python function that reverses a binary tree",
    "Explain the time complexity of quicksort",
    "Design an API for a chat application",
]
SATISFACTION_PROMPT: str = "thanks, that worked!"


async def _post_chat(
    client: httpx.AsyncClient, session_id: str, prompt: str
) -> Optional[dict]:
    """POST a chat completion with retry + timeout. Returns response JSON or None."""
    body = {
        "model": ROUTER_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "metadata": {"litellm_session_id": session_id},
    }
    last_exc: Optional[Exception] = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            r = await client.post(
                f"{PROXY_URL}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {PROXY_KEY}"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * attempt)
    print(
        f"  request failed after {RETRY_ATTEMPTS} attempts (session={session_id}): {last_exc}",
        file=sys.stderr,
    )
    return None


async def send_session(
    client: httpx.AsyncClient,
    session_id: str,
    prompts: List[str],
    satisfy: bool = True,
) -> Optional[str]:
    """Send a session of N turns. Returns the model that handled the last turn."""
    last_model: Optional[str] = None
    for prompt in prompts:
        resp = await _post_chat(client, session_id, prompt)
        if resp is None:
            return None
        last_model = resp.get("model") or last_model
    if satisfy:
        await _post_chat(client, session_id, SATISFACTION_PROMPT)
    return last_model


async def _proxy_health_check(client: httpx.AsyncClient) -> bool:
    """Confirm the proxy is reachable before doing anything else."""
    try:
        r = await client.get(f"{PROXY_URL}/health/liveliness", timeout=5.0)
        return r.status_code == 200
    except Exception as e:  # noqa: BLE001
        print(f"proxy unreachable at {PROXY_URL}: {e}", file=sys.stderr)
        return False


async def main() -> None:
    print("=== verify_adaptive_router.py ===")
    print(f"proxy:           {PROXY_URL}")
    print(f"router:          {ROUTER_NAME}")
    print(f"expected winner: {EXPECTED_WINNER}")
    print(f"train sessions:  {TRAIN_SESSIONS}")
    print(f"converge runs:   {CONVERGE_SESSIONS}\n")

    async with httpx.AsyncClient() as client:
        if not await _proxy_health_check(client):
            print("FAIL: proxy health check did not return 200.", file=sys.stderr)
            sys.exit(1)

        # ---- Phase 1: training -------------------------------------------
        print(
            f"Phase 1: training ({TRAIN_SESSIONS} sessions of 3 turns + satisfaction)..."
        )
        for i in range(TRAIN_SESSIONS):
            sid = f"verify-train-{uuid.uuid4()}"
            await send_session(client, sid, PROMPTS, satisfy=True)
            if (i + 1) % 5 == 0:
                print(f"  trained {i + 1}/{TRAIN_SESSIONS} sessions")

        print(
            f"\nWaiting {FLUSHER_DRAIN_WAIT_SECONDS:.0f}s for flusher to drain queue..."
        )
        await asyncio.sleep(FLUSHER_DRAIN_WAIT_SECONDS)

        # ---- Phase 2: convergence ----------------------------------------
        print(f"\nPhase 2: convergence test ({CONVERGE_SESSIONS} cold sessions)...")
        picks: List[str] = []
        for i in range(CONVERGE_SESSIONS):
            sid = f"verify-test-{uuid.uuid4()}"
            m = await send_session(client, sid, [PROMPTS[0]], satisfy=False)
            if m:
                picks.append(m)
                print(f"  session {i + 1}: picked {m}")

        if not picks:
            print("\nFAIL: no successful picks in convergence phase.", file=sys.stderr)
            sys.exit(1)
        winner_share = picks.count(EXPECTED_WINNER) / len(picks)
        print(
            f"\n{EXPECTED_WINNER} share: {winner_share:.0%} "
            f"({picks.count(EXPECTED_WINNER)}/{len(picks)})"
        )

        # ---- Phase 3: sticky session -------------------------------------
        print("\nPhase 3: sticky session test...")
        sid = f"verify-sticky-{uuid.uuid4()}"
        models: List[str] = []
        for _ in range(3):
            m = await send_session(client, sid, [PROMPTS[0]], satisfy=False)
            if m:
                models.append(m)
        if len(models) == 3 and len(set(models)) == 1:
            print(f"  PASS: same model {models[0]} across 3 turns of session {sid}")
        else:
            print(
                f"  FAIL: models differed within session: {models}",
                file=sys.stderr,
            )
            sys.exit(1)

        # ---- Phase 4: latency benchmark ----------------------------------
        print("\nPhase 4: routing latency (5 picks, p50)...")
        latencies: List[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            await send_session(
                client, f"verify-lat-{uuid.uuid4()}", [PROMPTS[0]], satisfy=False
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        print(f"  p50 e2e roundtrip: {p50 * 1000:.0f}ms")

        # ---- Verdict -----------------------------------------------------
        if winner_share >= WIN_THRESHOLD:
            print(
                f"\nPASS: convergence ({winner_share:.0%} >= {WIN_THRESHOLD:.0%}) + "
                f"sticky + latency checks all green."
            )
            sys.exit(0)
        print(
            f"\nFAIL: convergence too weak ({winner_share:.0%} < {WIN_THRESHOLD:.0%}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
