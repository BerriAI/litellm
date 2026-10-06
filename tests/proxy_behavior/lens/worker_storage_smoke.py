import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
from lens.agent_runtime import PythonAgentTurn
from lens.agent_workspace import PythonRequest
from lens.analysis import Extraction
from lens.models import (
    Claim,
    Execution,
    ExecutionContent,
    Job,
    LensSettings,
    ModelRequest,
    ModelResult,
    Result,
    Sample,
    TracePart,
)
from lens.worker import LensWorker
from pydantic import BaseModel, ConfigDict


class ToolReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tool_results: tuple[str, ...]


class PythonOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stdout: str
    stderr: str
    error: str
    output_complete: bool


class PythonReply(BaseModel):
    model_config = ConfigDict(extra="ignore")
    output: PythonOutput


async def main() -> None:
    now: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    claims: Final = iter(("full", "healthy"))
    saved: Final = SimpleQueue[Result]()
    failures: Final = SimpleQueue[str]()
    settings: Final = LensSettings(name="Storage recovery", model="unused", context="Finish the task", concurrency=1)
    execution: Final = Execution(
        id="run", source="traces", trace_id="trace", team_id="", name="Task", start_time="", span_count=1
    )

    def handle(request: httpx.Request) -> httpx.Response:
        path: Final = request.url.path
        if path.endswith("/claim"):
            claim: Final = Claim(
                lens_id="lens",
                job=Job(id=next(claims), created_at=now, start=now, end=now, settings=settings, revision=1),
                findings=(),
            )
            return httpx.Response(200, json=claim.model_dump(mode="json"))
        if path.endswith("/sample"):
            return httpx.Response(200, json=Sample(executions=(execution,), eligible=1).model_dump())
        if path.endswith("/reviews"):
            return httpx.Response(200, json=[])
        if path.endswith("/content"):
            content: Final = ExecutionContent(
                execution=execution,
                parts=(TracePart(execution_id="run", span_id="span", name="tool", kind="tool", content="Finished"),),
            )
            return httpx.Response(200, json=content.model_dump())
        if path.endswith("/model"):
            body: Final = ModelRequest.model_validate_json(request.content)
            full: Final = "/full/" in path
            if len(body.messages) == 2:
                code: Final = 'open("large", "wb").write(b"x" * 1048576)' if full else 'print("recovered")'
                return httpx.Response(
                    200,
                    json=ModelResult(
                        content=PythonAgentTurn[Extraction](
                            tools=(PythonRequest(action="python", code=code),)
                        ).model_dump_json(),
                        cost=0,
                    ).model_dump(),
                )
            output: Final = PythonReply.model_validate_json(
                ToolReply.model_validate_json(body.messages[-1].content).tool_results[0]
            ).output
            if full:
                assert output.error and not output.output_complete and "No space left on device" in output.stderr, (
                    output
                )
                failures.put(output.stderr)
            else:
                assert not output.error and output.output_complete and output.stdout == "recovered\n", output
            return httpx.Response(
                200,
                json=ModelResult(
                    content=PythonAgentTurn[Extraction](
                        result=Extraction(
                            cannot_assess=full,
                            reasoning="Python temporary storage was full" if full else "Analysis recovered",
                        )
                    ).model_dump_json(),
                    cost=0,
                ).model_dump(),
            )
        if path.endswith("/result"):
            saved.put(Result.model_validate_json(request.content))
            return httpx.Response(200, json=True)
        assert path.endswith(("/progress", "/heartbeat")), path
        return httpx.Response(200, json=True)

    async with httpx.AsyncClient(base_url="https://proxy.test", transport=httpx.MockTransport(handle)) as client:
        worker: Final = LensWorker(client)
        assert await worker.run_once()
        failed: Final = saved.get_nowait()
        assert failures.qsize() == 1 and failed.coverage.unassessable == 1 and not failed.findings
        assert not tuple(Path("/tmp").glob("lens-python-*")), "Failed computation left temporary files behind"
        assert await worker.run_once()
        recovered: Final = saved.get_nowait()
        assert recovered.error == "" and recovered.coverage.screened == 1 and recovered.coverage.unassessable == 0
        assert not tuple(Path("/tmp").glob("lens-python-*"))
    logging.warning(
        "Default worker reported Python storage exhaustion, cleaned scratch, and completed its next investigation"
    )


if __name__ == "__main__":
    asyncio.run(main())
