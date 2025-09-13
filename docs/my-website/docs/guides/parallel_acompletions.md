# Experimental: Parallel ACompletions

`parallel_acompletions` lets you fire multiple `Router.acompletion` requests concurrently with a simple list, handling retries/fallbacks per underlying call exactly as the normal router would. A sister iterator interface streams results as soon as each finishes.

## Status

- Flag gated (OFF by default).
- API surface, naming, and result shape may change.
- Enable explicitly before relying in production.

## Enabling

```bash
export LITELLM_ENABLE_PARALLEL_ACOMPLETIONS=1
```

## Basic Usage

```python
import os, asyncio
os.environ["LITELLM_ENABLE_PARALLEL_ACOMPLETIONS"] = "1"

from litellm import Router
from litellm.router_utils.parallel_acompletion import RouterParallelRequest

router = Router(model_list=[
    {
        "model_name": "gpt35",
        "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "..."}
    }
])

requests = [
    RouterParallelRequest(model="gpt35", messages=[{"role":"user","content":"Hello"}]),
    RouterParallelRequest(model="gpt35", messages=[{"role":"user","content":"Tell me a joke"}]),
]

async def main():
    results = await router.parallel_acompletions(requests, concurrency=4, preserve_order=True)
    for r in results:
        if r.exception:
            print("ERR:", r.exception)
        else:
            print(r.response)

asyncio.run(main())
```

## Iterator Form (completion order)

```python
async for result in router.iter_parallel_acompletions(requests, concurrency=4):
    ...
```

## Behavior & Error Semantics

- `return_exceptions=True` (default)
  - Each item in the results has either `response` or `exception` set.
  - Iterator form yields all results; you handle `result.exception` per item.

- `return_exceptions=False` (fail-fast)
  - `parallel_acompletions(...)` raises on the first error and cancels remaining tasks.
  - `iter_parallel_acompletions(...)` raises on the first error and stops iteration; any outstanding tasks are cancelled.

- Concurrency
  - A bounded semaphore limits in-flight calls at the orchestration layer.
  - For very large request lists, tasks are scheduled but concurrency limits actual in-flight calls.

## Flag Gating Notes

- The feature is controlled by `LITELLM_ENABLE_PARALLEL_ACOMPLETIONS` and evaluated on import.
- If toggling the env var at runtime, restart the process (or reload modules) to apply.

## Arguments

| Param | Description |
|-------|-------------|
| `requests` | List of `RouterParallelRequest` (model, messages, optional kwargs) |
| `concurrency` | Max in-flight tasks at orchestration layer (default 8) |
| `return_exceptions` | If True (default) errors captured per-result; if False first error cancels all |
| `preserve_order` | For `parallel_acompletions`: if True, final list matches input order |

## Result Object

Each result is a `RouterParallelResult`:
```python
{
  "index": 0,
  "request": RouterParallelRequest(...),
  "response": <ModelResponse or custom>,
  "exception": Optional[Exception]
}
```

## Error Handling

- With `return_exceptions=True` you get a result entry containing `exception`.
- With `False`, the first exception aborts remaining tasks (standard `asyncio.gather` propagation).

## Why Flag Gated?

To allow fast iteration on:
- Naming (`parallel_acompletions`, `iter_parallel_acompletions`)
- Result shape (object vs tuple)
- Performance tuning (fair queueing, cancellation semantics)

Provide feedback in the GitHub issue / PR.

## Roadmap Ideas

- Per-request timeouts.
- Integrated progress callbacks.
- Batch adaptive concurrency.
- Streaming passthrough merge (aggregate token usage).

---
