#!/usr/bin/env python3
"""
Run multiple mini-agent tasks concurrently to validate reentrancy and stability.
Env:
  SCENARIO_MINI_CONCURRENCY=10
  SCENARIO_MINI_TOTAL=30
"""
import asyncio
import json
import os
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

try:
    from litellm.experimental_mcp_client.mini_agent.agent_proxy import AgentRunReq, run
except Exception as import_err:  # noqa: BLE001
    print(
        "Skipping mini-agent concurrency scenario (components unavailable): "
        f"{import_err}"
    )
    raise SystemExit(0)

PROMPTS = [
    [{"role": "user", "content": "Say hi"}],
    [{"role": "user", "content": "Name two fruits."}],
    [{"role": "user", "content": "Provide a short motivational quote."}],
    [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "List two benefits of code review."},
    ],
]


def _normalize_payload(resp: Any) -> Any:
    try:
        return resp.model_dump()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        try:
            return resp.dict()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            if isinstance(resp, (dict, list, str, int, float, bool, type(None))):
                return resp
            return str(resp)


async def _run_one(i: int) -> tuple[bool, Any]:
    req = AgentRunReq(
        messages=PROMPTS[i % len(PROMPTS)],
        model=f"mini-agent-concurrency-{i}",
        tool_backend="local",
        max_iterations=3,
        max_total_seconds=30,
    )
    try:
        resp = await run(req)
        return True, _normalize_payload(resp)
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


async def main() -> None:
    total = int(os.getenv("SCENARIO_MINI_TOTAL", "30"))
    concurrency = int(os.getenv("SCENARIO_MINI_CONCURRENCY", "10"))
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(i: int) -> tuple[bool, Any]:
        async with sem:
            return await _run_one(i)

    results = await asyncio.gather(*(wrapped(i) for i in range(total)))
    successes = sum(1 for ok, _ in results if ok)
    errors = total - successes
    print(
        f"=== mini-agent concurrency === total={total} concurrency={concurrency} "
        f"successes={successes} errors={errors}"
    )
    for idx, (ok, payload) in enumerate(results[:3]):
        print(f"sample[{idx}] ok={ok}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
