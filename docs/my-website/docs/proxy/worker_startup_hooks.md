# Worker Startup Hooks

Use `LITELLM_WORKER_STARTUP_HOOKS` to run custom initialization functions in **each worker process** during proxy startup. This is essential when using multi-worker deployments (`--num_workers > 1`) with libraries that require per-process initialization, such as [gflags](https://github.com/google/python-gflags).

## The Problem

When running the LiteLLM proxy with multiple workers:

```bash
litellm --config config.yaml --num_workers 4
```

Each worker is a **separate process** spawned by uvicorn or gunicorn. Any in-process state initialized in the master process (before `run_server()`) is **not available** in worker processes. This includes:

- [python-gflags](https://github.com/google/python-gflags) (`gflags.FLAGS`)
- [absl-py flags](https://abseil.io/docs/python/guides/flags) (`absl.flags.FLAGS`)
- Custom singleton registries or connection pools
- Any module-level state that requires explicit initialization

## Usage

Set the `LITELLM_WORKER_STARTUP_HOOKS` environment variable to a comma-separated list of `module.path:function_name` callables:

```bash
export LITELLM_WORKER_STARTUP_HOOKS="my_module:my_init_function"
```

Each hook is called **early** in the worker startup lifecycle — before config loading, database setup, or any request handling. Both sync and async functions are supported.

## Example: gflags Initialization

### 1. Define your wrapper module

```python title="my_litellm_wrapper.py"
import gflags
import json
import os
import sys
from typing import Optional, List, Any


def init_gflags(
    usage: Optional[Any] = None,
    raw_args: Optional[List[str]] = None,
    known_only: bool = False,
) -> List[str]:
    """Initialize gflags from command-line arguments."""
    try:
        gflags.FLAGS.set_gnu_getopt(True)
        if raw_args is None:
            raw_args = sys.argv
        argv = gflags.FLAGS(raw_args, known_only=known_only)
    except gflags.Error as e:
        if usage is None:
            print("%s\nUsage: %s ARGS\n%s" % (e, sys.argv[0], gflags.FLAGS))
        else:
            print(usage % dict(cmd=sys.argv[0], flags=gflags.FLAGS))
        sys.exit(1)
    return argv


def init_gflags_for_worker():
    """Re-initialize gflags in each worker process.

    Reads the original sys.argv from the GFLAGS_ARGV env var
    (set by the master process before starting the proxy).
    """
    raw_args = json.loads(os.environ.get("GFLAGS_ARGV", "[]")) or sys.argv
    init_gflags(raw_args=raw_args, known_only=True)
```

### 2. Start the proxy

```python title="start_proxy.py"
import json
import os
import sys

from my_litellm_wrapper import init_gflags

# Store sys.argv so workers can re-parse the same flags
os.environ["GFLAGS_ARGV"] = json.dumps(sys.argv)

# Tell LiteLLM to call our hook in each worker
os.environ["LITELLM_WORKER_STARTUP_HOOKS"] = "my_litellm_wrapper:init_gflags_for_worker"

# Initialize gflags in the master process
init_gflags()

# Start the proxy (programmatic invocation)
from litellm.proxy.proxy_cli import run_server

run_server(
    ["--config", "config.yaml", "--num_workers", "4"],
    standalone_mode=False,
)
```

Or via shell:

```bash
export GFLAGS_ARGV='["my_app", "--my_flag=value", "--batch_size=32"]'
export LITELLM_WORKER_STARTUP_HOOKS="my_litellm_wrapper:init_gflags_for_worker"

litellm --config config.yaml --num_workers 4
```

## How It Works

```
Master Process                          Worker Process (×N)
─────────────────                       ──────────────────────
1. init_gflags()                        3. proxy_startup_event():
2. run_server()                            → Read LITELLM_WORKER_STARTUP_HOOKS
   → sets env vars                         → Import & call each hook
   → uvicorn.run(workers=N)                  (gflags.FLAGS re-initialized ✓)
   → spawns workers ──────────────────►    → Continue with config/DB setup
                                           → Ready to serve requests
```

- Hooks run at the **very beginning** of `proxy_startup_event` (the FastAPI lifespan), before config loading, database connections, or any other initialization.
- Environment variables set in the master process are **inherited** by worker processes (standard Unix fork/spawn behavior).
- If a hook **raises an exception**, the worker fails to start — this is intentional, since missing initialization (e.g., uninitialized gflags) would cause downstream errors.

## Multiple Hooks

Separate multiple hooks with commas:

```bash
export LITELLM_WORKER_STARTUP_HOOKS="my_module:init_gflags,my_module:init_metrics,my_module:init_connections"
```

Hooks are executed **in order**, left to right.

## Async Hooks

Async functions are also supported — they are automatically awaited:

```python
async def init_async_connections():
    """Example async hook for initializing async resources."""
    await setup_async_connection_pool()
```

```bash
export LITELLM_WORKER_STARTUP_HOOKS="my_module:init_async_connections"
```

## Reference

| Environment Variable | Description |
|---|---|
| `LITELLM_WORKER_STARTUP_HOOKS` | Comma-separated `module.path:function_name` callables to run in each worker on startup |

The hook format follows the standard Python entry point syntax: `module.path:function_name`, where `module.path` is a dotted Python import path and `function_name` is the name of the callable within that module.
