# Start tracing memory allocations
import os
import tracemalloc

from fastapi import APIRouter

from litellm._logging import verbose_proxy_logger

router = APIRouter()

if os.environ.get("LITELLM_PROFILE", "false").lower() == "true":
    tracemalloc.start()

    @router.get("/memory-usage", include_in_schema=False)
    async def memory_usage():
        # Take a snapshot of the current memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        verbose_proxy_logger.debug("TOP STATS: %s", top_stats)

        # Get the top 50 memory usage lines
        top_50 = top_stats[:50]
        result = []
        for stat in top_50:
            result.append(f"{stat.traceback.format()}: {stat.size / 1024} KiB")

        return {"top_50_memory_usage": result}
