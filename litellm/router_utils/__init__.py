# Expose testing helpers for parallel acompletions
from .parallel_acompletion import (
    RouterParallelRequest,
    run_parallel_requests,
)

# Optional helper if present (added for smokes)
try:
    from .parallel_acompletion import gather_parallel_acompletions  # type: ignore
except Exception:
    def gather_parallel_acompletions(*_a, **_k):  # type: ignore
        raise ImportError("gather_parallel_acompletions not available")

__all__ = [
    "RouterParallelRequest",
    "run_parallel_requests",
    "gather_parallel_acompletions",
]