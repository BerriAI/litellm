import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
from engine.models import (
    Claim,
    EngineSettings,
    Execution,
    ExecutionContent,
    Job,
    ModelResult,
    Result,
    Sample,
    TracePart,
)
from engine.worker import EngineWorker


async def main() -> None:
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    claims: Final = iter(("full", "healthy"))
    saved: Final = SimpleQueue[Result]()
    pages: Final = SimpleQueue[str]()
    settings: Final = EngineSettings(name="Storage recovery", model="unused", context="Finish the task", concurrency=1)
    execution: Final = Execution(
        id="run", source="traces", trace_id="trace", team_id="", name="Task", start_time="", span_count=10000
    )

    def handle(request: httpx.Request) -> httpx.Response:
        path: Final = request.url.path
        if path.endswith("/claim"):
            claim: Final = Claim(
                engine_id="lens",
                job=Job(id=next(claims), created_at=now, start=now, end=now, settings=settings, revision=1),
                findings=(),
            )
            return httpx.Response(200, json=claim.model_dump(mode="json"))
        if path.endswith("/sample"):
            return httpx.Response(200, json=Sample(executions=(execution,), eligible=1).model_dump())
        if path.endswith("/content"):
            healthy: Final = "/healthy/" in path
            cursor: Final = request.url.params.get("cursor", "")
            pages.put(cursor)
            assert pages.qsize() < 100, "The deliberately small temporary mount must fill"
            content: Final = ExecutionContent(
                execution=execution,
                parts=tuple(
                    TracePart(
                        execution_id="run",
                        span_id=f"{cursor}-{i}",
                        name="tool",
                        kind="tool",
                        content="Finished" if healthy else "x" * 8000,
                    )
                    for i in range(1 if healthy else 40)
                ),
                next_cursor=None if healthy else str(pages.qsize()),
            )
            return httpx.Response(200, json=content.model_dump())
        if path.endswith("/model"):
            assert "/healthy/" in path, "Storage failure must occur before spending on analysis"
            return httpx.Response(200, json=ModelResult(content='{"observations":[]}', cost=0).model_dump())
        if path.endswith("/result"):
            saved.put(Result.model_validate_json(request.content))
            return httpx.Response(200, json=True)
        assert path.endswith(("/progress", "/heartbeat")), path
        return httpx.Response(200, json=True)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        worker: Final = EngineWorker(client)
        assert await worker.run_once()
        failed: Final = saved.get_nowait()
        assert failed.error.startswith("Worker temporary storage failed.")
        assert not failed.findings
        assert not tuple(Path("/tmp").glob("lens-trace-*")), "Failed scan left temporary files behind"
        assert await worker.run_once()
        recovered: Final = saved.get_nowait()
        assert recovered.error == "" and recovered.coverage.screened == 1
        assert not tuple(Path("/tmp").glob("lens-trace-*"))
    logging.info("Storage-full scan failed clearly; temporary files cleaned; next scan completed")


if __name__ == "__main__":
    asyncio.run(main())
