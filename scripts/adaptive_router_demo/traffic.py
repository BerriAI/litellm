"""
Synthetic traffic generator for the adaptive_router demo dashboard.

What it does:
  - Sends labeled multi-turn chat requests to the proxy's adaptive router.
  - For each turn, peeks at the `x-litellm-adaptive-router-model` response
    header to learn which underlying model was picked.
  - Draws a Bernoulli outcome from a hard-coded ORACLE table that says
    "model M succeeds at request type T with probability p".
  - Sends a final follow-up turn whose user message is engineered to
    BOTH classify into the same RequestType AND match the
    satisfaction regex on success (so the bandit's `(type, model)` cell
    gets +alpha). On failure we send a neutral follow-up so no signal
    fires — over time, models the oracle favors accumulate alpha faster.

Why this shape:
  - The post-call hook gates signal recording on len(messages) >= 4.
    A single 5-message request passes the gate in one round-trip, which
    keeps the demo cheap.
  - Mock responses (`mock_response=...`) skip the real LLM call but still
    flow through routing + post-call hooks, so no API keys / no spend.

Run:
  uv run python scripts/adaptive_router_demo/traffic.py \\
      --proxy-url http://localhost:4000 \\
      --api-key   sk-1234 \\
      --router    smart-cheap-router \\
      --rounds    100 \\
      --rate      0.5

Open `dashboard.html` in a browser alongside this and watch the bars move.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import uuid
from typing import Dict, List, Tuple

import httpx

# ---- prompts (paired with the RequestType the classifier will assign) ----
# Each prompt is engineered to (a) classify into the listed type and (b) make
# sense as a user request. Keep prompts short to limit token cost.
PROMPTS: Dict[str, List[str]] = {
    "code_generation": [
        "Write a Python function that flattens a nested list",
        "Create a TypeScript function that debounces another function",
        "Build a Rust function that parses a CSV string",
        "Generate a SQL function that returns running totals",
    ],
    "factual_lookup": [
        "What is the capital of New Zealand?",
        "When was the Treaty of Westphalia signed?",
        "Who is the current Secretary General of the UN?",
        "Where is Mount Kilimanjaro located?",
    ],
    "writing": [
        "Write an email declining a meeting politely",
        "Draft a paragraph introducing a product launch",
        "Compose a short blog post about morning routines",
        "Rewrite this sentence to be more concise: ...",
    ],
}

# Engineered satisfaction follow-ups — each one is designed to:
#   (1) match the satisfaction regex (thanks/great/works/perfect/etc.), AND
#   (2) re-classify into the SAME RequestType as the first prompt
# so that signals attribute to the right (type, model) bandit cell.
SATISFY: Dict[str, str] = {
    "code_generation": "thanks, that works! now write me a python function that does the inverse",
    "factual_lookup":  "perfect, thanks! who is the current prime minister?",
    "writing":         "great, thanks! now write a follow-up email confirming attendance",
}

# Neutral follow-up — does not match any signal regex, does not move the bandit.
NEUTRAL_FOLLOWUP = "ok, noted"

# Oracle: P(success | request_type, model). Tunable.
# Defaults: smart dominates code/writing; both are fine for factual_lookup.
ORACLE: Dict[str, Dict[str, float]] = {
    "code_generation": {"smart": 0.92, "fast": 0.35},
    "factual_lookup":  {"smart": 0.90, "fast": 0.85},
    "writing":         {"smart": 0.85, "fast": 0.55},
}

# Fabricated assistant turn — content doesn't matter for the hook, only the role.
FAB_ASSISTANT = "Got it. Working on that now."


def _build_messages(prompt: str, last_user: str) -> List[Dict[str, str]]:
    """5-message conversation that passes the SIGNAL_GATE_MIN_MESSAGES=4 gate."""
    return [
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": FAB_ASSISTANT},
        {"role": "user",      "content": "ok continue"},
        {"role": "assistant", "content": FAB_ASSISTANT},
        {"role": "user",      "content": last_user},
    ]


async def _send(
    client: httpx.AsyncClient,
    proxy_url: str,
    api_key: str,
    router: str,
    session_id: str,
    messages: List[Dict[str, str]],
    mock_response: str,
) -> Tuple[bool, str]:
    """Returns (ok, chosen_model)."""
    body = {
        "model": router,
        "messages": messages,
        "metadata": {"litellm_session_id": session_id},
        "mock_response": mock_response,
    }
    try:
        r = await client.post(
            f"{proxy_url}/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  request failed: {e}", file=sys.stderr)
        return False, ""
    chosen = r.headers.get("x-litellm-adaptive-router-model", "")
    return True, chosen


async def _drive_one_session(
    client: httpx.AsyncClient,
    proxy_url: str,
    api_key: str,
    router: str,
    request_type: str,
    prompt: str,
) -> str:
    """Run one labeled session. Returns the chosen model (for logging)."""
    session_id = f"demo-{uuid.uuid4()}"

    # Send the engineered 5-message conversation. The follow-up is chosen
    # AFTER we observe what model the router would pick — but since the
    # router is sticky-per-session, the model on this single round-trip
    # IS the model we're crediting.
    #
    # Pre-decide success based on the oracle for whichever model gets picked.
    # We can't know the pick before sending, so: send a neutral follow-up
    # first to learn the pick, then send a second round with credit attached.
    #
    # Round 1: neutral follow-up → no signal fires, but we learn the pick.
    ok, chosen = await _send(
        client, proxy_url, api_key, router, session_id,
        _build_messages(prompt, NEUTRAL_FOLLOWUP),
        mock_response=FAB_ASSISTANT,
    )
    if not ok or not chosen:
        return ""

    # Decide outcome from oracle.
    p = ORACLE.get(request_type, {}).get(chosen, 0.5)
    success = random.random() < p
    follow_up = SATISFY[request_type] if success else NEUTRAL_FOLLOWUP

    # Round 2: include the round-1 turns + a new follow-up. On success the
    # follow-up matches satisfaction → +alpha for (request_type, chosen).
    history = _build_messages(prompt, NEUTRAL_FOLLOWUP) + [
        {"role": "assistant", "content": FAB_ASSISTANT},
        {"role": "user",      "content": follow_up},
    ]
    await _send(
        client, proxy_url, api_key, router, session_id, history,
        mock_response=FAB_ASSISTANT,
    )
    return chosen


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-url", default="http://localhost:4000")
    ap.add_argument("--api-key",   required=True, help="proxy key with /v1/chat/completions perms")
    ap.add_argument("--router",    default="smart-cheap-router")
    ap.add_argument("--rounds",    type=int, default=100)
    ap.add_argument("--rate",      type=float, default=0.5,
                    help="seconds between sessions; lower = faster")
    ap.add_argument("--types",     default="code_generation,factual_lookup,writing",
                    help="comma-separated subset of request types to drive")
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(",") if t.strip() in PROMPTS]
    if not types:
        print(f"ERROR: no valid types. Choose from: {list(PROMPTS)}", file=sys.stderr)
        sys.exit(2)

    print(f"driving {args.rounds} sessions across types: {types}")
    print(f"oracle: {ORACLE}")
    print(f"proxy: {args.proxy_url}  router: {args.router}\n")

    counts: Dict[Tuple[str, str], int] = {}
    async with httpx.AsyncClient() as client:
        for i in range(args.rounds):
            rt = random.choice(types)
            prompt = random.choice(PROMPTS[rt])
            chosen = await _drive_one_session(
                client, args.proxy_url, args.api_key, args.router, rt, prompt,
            )
            if chosen:
                counts[(rt, chosen)] = counts.get((rt, chosen), 0) + 1
            if (i + 1) % 10 == 0:
                summary = ", ".join(
                    f"{rt}/{m}={n}" for (rt, m), n in sorted(counts.items())
                )
                print(f"  round {i + 1}/{args.rounds}  picks: {summary}")
            await asyncio.sleep(args.rate)

    print("\nfinal pick distribution:")
    for (rt, m), n in sorted(counts.items()):
        print(f"  {rt:22s} → {m:8s}  {n}")


if __name__ == "__main__":
    asyncio.run(main())
