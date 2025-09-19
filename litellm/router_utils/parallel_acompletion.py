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
