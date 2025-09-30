#!/usr/bin/env python3
"""Run multiple mini-agent jobs concurrently to validate reentrancy."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


def _env_int(default: int, *names: str) -> int:
    for name in names:
        value = os.getenv(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            pass
    return default

try:
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"Skipping mini-agent concurrency scenario (components unavailable): {exc}")
    raise SystemExit(0)

PROMPTS: List[List[Dict[str, Any]]] = [
    [{"role": "user", "content": "Say hi"}],
    [{"role": "user", "content": "Name two fruits."}],
    [{"role": "user", "content": "Provide a short motivational quote."}],
    [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "List two benefits of code review."},
    ],
]


async def _one(i: int) -> Tuple[bool, Dict[str, Any]]:
    req = AgentRunReq(
        messages=PROMPTS[i % len(PROMPTS)],
        model=f"mini-agent-concurrency-{i}",
        tool_backend="local",
        max_iterations=3,
        max_total_seconds=30,
    )
    try:
        resp = await run(req)
        payload: Any
        try:
            payload = resp.model_dump()
        except Exception:
            try:
                payload = resp.dict()
            except Exception:
                payload = resp
        if not isinstance(payload, dict):
            payload = {"result": payload}
        return True, payload
    except Exception as exc:  # pragma: no cover - live path
        return False, {"error": str(exc)}


async def main() -> None:
    total = max(1, _env_int(8, "STRESS_TOTAL", "SCENARIO_MINI_TOTAL"))
    concurrency = max(1, _env_int(4, "STRESS_CONCURRENCY", "SCENARIO_MINI_CONCURRENCY"))
    if (total > 16 or concurrency > 8) and os.getenv("STRESS_HEAVY") != "1":
        print(
            f"Requested total={total} concurrency={concurrency} exceeds safe defaults. "
            "Set STRESS_HEAVY=1 to opt into heavier mini-agent concurrency runs or lower STRESS_TOTAL/STRESS_CONCURRENCY."
        )
        return
    sem = asyncio.Semaphore(concurrency)

    async def runner(i: int) -> Tuple[bool, Dict[str, Any]]:
        async with sem:
            return await _one(i)

    results = await asyncio.gather(*(runner(i) for i in range(total)))
    successes = sum(1 for ok, _ in results if ok)
    errors = total - successes
    print(f"=== mini-agent concurrency === total={total} concurrency={concurrency} successes={successes} errors={errors}")
    for idx, (ok, payload) in enumerate(results[:3]):
        print(f"sample[{idx}] ok={ok}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
