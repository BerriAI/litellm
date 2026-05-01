# ruff: noqa: T201
"""
Adaptive router evaluator — LLM-as-judge harness.

For each test case:
  1. Sends the prompt to the adaptive router.
  2. Reads which model was picked (x-litellm-adaptive-router-model header).
  3. Asks the judge model whether the response meets the ideal criteria.
  4. Prints PASS or FAIL with one line of reasoning.

Run:
  uv run python scripts/adaptive_router_demo/eval.py \
      --proxy-url   http://localhost:4000 \
      --api-key     sk-1234 \
      --router      smart-cheap-router \
      --judge-model smart
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    category: str
    prompt: str
    ideal: str          # criteria the judge checks the response against


EVAL_CASES: List[EvalCase] = [
    # code_generation
    EvalCase(
        category="code_generation",
        prompt="Write a Python function that flattens a nested list of arbitrary depth.",
        ideal=(
            "A Python function (def flatten(...)) that accepts a list which may "
            "contain nested lists to arbitrary depth and returns a single flat list "
            "with all elements in order. Must handle at least two levels of nesting."
        ),
    ),
    EvalCase(
        category="code_generation",
        prompt="Write a Python decorator that retries a function up to 3 times on exception.",
        ideal=(
            "A Python decorator that wraps a callable, catches exceptions, and "
            "retries the call up to 3 times before re-raising. Should use functools.wraps "
            "or equivalent to preserve the wrapped function's metadata."
        ),
    ),
    EvalCase(
        category="code_generation",
        prompt="Write a SQL query that returns the top 5 customers by total order value.",
        ideal=(
            "A valid SQL SELECT query that JOINs an orders or order_items table with a "
            "customers table, groups by customer, sums order value, orders descending, "
            "and limits to 5 rows."
        ),
    ),
    # factual_lookup
    EvalCase(
        category="factual_lookup",
        prompt="What is the capital of New Zealand?",
        ideal="The answer must state Wellington as the capital of New Zealand.",
    ),
    EvalCase(
        category="factual_lookup",
        prompt="In what year did World War II end?",
        ideal="The answer must state 1945 as the year World War II ended.",
    ),
    EvalCase(
        category="factual_lookup",
        prompt="What is the chemical symbol for gold?",
        ideal="The answer must include 'Au' as the chemical symbol for gold.",
    ),
    # writing
    EvalCase(
        category="writing",
        prompt=(
            "Write a short, polite email declining a meeting request because of "
            "a scheduling conflict."
        ),
        ideal=(
            "A professional email that: (1) thanks the sender for the invitation, "
            "(2) clearly declines, (3) mentions a scheduling conflict as the reason, "
            "and (4) offers to reschedule or an alternative. Tone must be polite."
        ),
    ),
    EvalCase(
        category="writing",
        prompt="Write a one-paragraph product description for noise-cancelling headphones.",
        ideal=(
            "A marketing paragraph for noise-cancelling headphones that mentions "
            "noise cancellation as a feature, highlights at least one other benefit "
            "(comfort, audio quality, battery life, or similar), and ends with a "
            "persuasive call to action or closing statement."
        ),
    ),
]

# Matches the satisfaction regex in signals.py (_SATISFACTION_PATTERNS).
SATISFY_FOLLOWUP = "great, thanks!"
NEUTRAL_FOLLOWUP = "ok, noted"
FAB_ASSISTANT = "Got it. Working on that now."

JUDGE_SYSTEM = (
    "You are a strict but fair evaluator. Your job is to decide whether a model "
    "response meets the stated requirements. Reply with exactly two lines:\n"
    "Line 1: PASS or FAIL\n"
    "Line 2: One sentence of reasoning (≤ 25 words)."
)


def _judge_user(prompt: str, ideal: str, actual: str) -> str:
    return (
        f"Question sent to model:\n{prompt}\n\n"
        f"Requirements the response must meet:\n{ideal}\n\n"
        f"Actual model response:\n{actual}\n\n"
        "Does the response meet the requirements? Reply PASS or FAIL."
    )


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def _chat(
    client: httpx.AsyncClient,
    proxy_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    session_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Returns (response_text, chosen_model_header).
    chosen_model_header is empty for non-router calls.
    """
    body: Dict = {"model": model, "messages": messages}
    if session_id:
        body["metadata"] = {"litellm_session_id": session_id}

    resp = await client.post(
        f"{proxy_url}/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    chosen = resp.headers.get("x-litellm-adaptive-router-model", "")
    return text, chosen


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
async def evaluate(
    proxy_url: str,
    api_key: str,
    router: str,
    judge_model: str,
) -> None:
    passed = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for i, case in enumerate(EVAL_CASES, 1):
            print(f"\n[{i}/{len(EVAL_CASES)}] category={case.category}")
            print(f"  prompt   : {case.prompt[:80]}{'…' if len(case.prompt) > 80 else ''}")

            session_id = f"eval-{uuid.uuid4()}"

            # Round 1: single-turn real request — get the actual LLM response to judge.
            try:
                response, chosen = await _chat(
                    client, proxy_url, api_key, router,
                    [{"role": "user", "content": case.prompt}],
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR calling router: {exc}", file=sys.stderr)
                failed += 1
                continue

            print(f"  model    : {chosen or router}")
            print(f"  response : {response[:120].replace(chr(10), ' ')}{'…' if len(response) > 120 else ''}")

            # Judge the real response.
            judge_msgs = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": _judge_user(case.prompt, case.ideal, response)},
            ]
            try:
                verdict, _ = await _chat(
                    client, proxy_url, api_key, judge_model, judge_msgs,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR calling judge: {exc}", file=sys.stderr)
                failed += 1
                continue

            # Parse verdict — first non-empty line should be PASS or FAIL.
            lines = [ln.strip() for ln in verdict.splitlines() if ln.strip()]
            first = lines[0].upper() if lines else ""
            reason = lines[1] if len(lines) > 1 else ""
            is_pass = "PASS" in first

            if is_pass:
                passed += 1
                print(f"  verdict  : \033[32mPASS\033[0m  {reason}")
            else:
                failed += 1
                print(f"  verdict  : \033[31mFAIL\033[0m  {reason}")

            # Round 2: 5-message conversation on the same session_id so the bandit fires.
            # On PASS → satisfaction follow-up (+alpha). On FAIL → neutral (no signal).
            follow_up = SATISFY_FOLLOWUP if is_pass else NEUTRAL_FOLLOWUP
            bandit_msgs = [
                {"role": "user",      "content": case.prompt},
                {"role": "assistant", "content": response},
                {"role": "user",      "content": "ok continue"},
                {"role": "assistant", "content": FAB_ASSISTANT},
                {"role": "user",      "content": follow_up},
            ]
            try:
                await _chat(
                    client, proxy_url, api_key, router, bandit_msgs,
                    session_id=session_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: bandit update failed: {exc}", file=sys.stderr)

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed  ({failed} failed)")
    if passed == total:
        print("All test cases passed — the adaptive router is working well!")
    elif passed >= total * 0.8:
        print("Most test cases passed — minor issues to investigate.")
    else:
        print("Significant failures — check router config and model availability.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the adaptive router with LLM-as-judge.")
    ap.add_argument("--proxy-url",    default="http://localhost:4000")
    ap.add_argument("--api-key",      required=True, help="proxy API key")
    ap.add_argument("--router",       default="smart-cheap-router", help="adaptive router model name")
    ap.add_argument("--judge-model",  default="smart", help="model name for the judge (via proxy)")
    args = ap.parse_args()

    asyncio.run(evaluate(args.proxy_url, args.api_key, args.router, args.judge_model))


if __name__ == "__main__":
    main()
