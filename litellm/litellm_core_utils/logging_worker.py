import asyncio
import contextlib
from typing import Optional

_LOG_Q: Optional[asyncio.Queue] = None
_LOG_TASK: Optional[asyncio.Task] = None
_LOG_TIMEOUT_S = 10.0  # keep short; logs are best-effort
_MAX_LOG_QUEUE_SIZE = 10_000

def _ensure_queue():
    global _LOG_Q
    if _LOG_Q is None:
        _LOG_Q = asyncio.Queue(maxsize=_MAX_LOG_QUEUE_SIZE)  # bounded => no backpressure

def ensure_log_worker(logging_obj):
    """Idempotent: start one worker on the current loop."""
    _ensure_queue()
    global _LOG_TASK
    if _LOG_TASK is None or _LOG_TASK.done():
        _LOG_TASK = asyncio.create_task(_log_worker(logging_obj))

async def _log_worker(logging_obj):
    try:
        if _LOG_Q is None:
            return
        while True:
            # process sequentially, one at a time (keeps loop load predictable)
            result, t0, t1 = await _LOG_Q.get()
            try:
                await asyncio.wait_for(
                    logging_obj.async_success_handler(result, t0, t1),
                    timeout=_LOG_TIMEOUT_S,
                )
            except Exception:
                # best-effort; swallow errors/timeouts
                pass
            finally:
                _LOG_Q.task_done()
    except asyncio.CancelledError:
        pass

def try_enqueue_log(result, t0, t1):
    """Hot path: never await; drop if full."""
    if _LOG_Q is None:
        return
    try:
        _LOG_Q.put_nowait((result, t0, t1))
    except asyncio.QueueFull:
        # drop on overload to protect RPS
        pass

async def stop_log_worker():
    global _LOG_TASK
    if _LOG_TASK:
        _LOG_TASK.cancel()
        with contextlib.suppress(Exception):
            await _LOG_TASK
        _LOG_TASK = None