import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class RouterParallelRequest:
    model: str
    messages: List[Dict[str, Any]]
    kwargs: Dict[str, Any]

async def _run_one(router, req: RouterParallelRequest, idx: int):
    try:
        resp = await router.acompletion(model=req.model, messages=req.messages, **req.kwargs)
        return idx, resp, None
    except Exception as e:
        return idx, None, e

async def run_parallel_requests(router, requests, preserve_order=True, return_exceptions=True):
    """Internal test helper: run multiple acompletions concurrently.

    Not a public API. Used by tests to verify ordering and fail-fast behavior without
    adding new surface to Router. """
    if not return_exceptions:
        # Let exceptions propagate (fail-fast)
        tasks = [asyncio.create_task(router.acompletion(model=req.model, messages=req.messages, **req.kwargs)) for req in requests]
        results_raw = await asyncio.gather(*tasks, return_exceptions=False)
        results = list(enumerate(results_raw))
        if preserve_order:
            results.sort(key=lambda x: x[0])
        return [(i, r, None) for i, r in results]
    # return_exceptions=True path: capture per-request exceptions
    tasks = [asyncio.create_task(_run_one(router, req, i)) for i, req in enumerate(requests)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    if preserve_order:
        results.sort(key=lambda x: x[0])
    return results
