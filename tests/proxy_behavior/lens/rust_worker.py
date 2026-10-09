import asyncio
import os
import secrets
import socket
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Final

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException

from litellm.proxy.lens.models import Claim, ExecutionContent, ModelRequest, ModelResult, Progress, Result, Sample
from litellm.proxy.lens.release import PROTOCOL_VERSION


async def run_worker(
    binary: Path,
    claim: Claim,
    sample: Sample,
    read: Callable[[str, str, int], Awaitable[ExecutionContent]],
    model: Callable[[ModelRequest], Awaitable[ModelResult]],
    progress: Callable[[Progress], Awaitable[None]],
) -> Result:
    token: Final = secrets.token_urlsafe(32)
    release: Final = "lens-evaluation"

    def auth(authorization: str = Header()) -> None:
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid worker credential")

    app: Final = FastAPI(dependencies=[Depends(auth)])
    completed: Final = asyncio.Future[Result]()

    @app.post("/lens/worker/claim")
    async def take(protocol_version: int, worker_release: str) -> Claim:
        if protocol_version != PROTOCOL_VERSION or worker_release != release:
            raise HTTPException(409, "Incompatible worker")
        return claim

    @app.get("/lens/worker/{lens_id}/{job_id}/sample")
    async def sampled(lens_id: str, job_id: str) -> Sample:
        return sample

    @app.get("/lens/worker/{lens_id}/{job_id}/reviews")
    async def reviews(lens_id: str, job_id: str) -> tuple[()]:
        return ()

    @app.get("/lens/worker/{lens_id}/{job_id}/content")
    async def content(
        lens_id: str, job_id: str, execution_id: str, cursor: str = "", offset: int = 1
    ) -> ExecutionContent:
        return await read(execution_id, cursor, max(0, offset - 1))

    @app.post("/lens/worker/{lens_id}/{job_id}/model")
    async def infer(lens_id: str, job_id: str, body: ModelRequest) -> ModelResult:
        return await model(body)

    @app.post("/lens/worker/{lens_id}/{job_id}/progress")
    async def update(lens_id: str, job_id: str, body: Progress) -> bool:
        await progress(body)
        return True

    @app.post("/lens/worker/{lens_id}/{job_id}/heartbeat")
    async def heartbeat(lens_id: str, job_id: str) -> bool:
        return True

    @app.post("/lens/worker/{lens_id}/{job_id}/result")
    async def result(lens_id: str, job_id: str, body: Result) -> bool:
        if not completed.done():
            completed.set_result(body)
        return True

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server: Final = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        serving: Final = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            while not server.started:
                if serving.done():
                    await serving
                    raise RuntimeError("Evaluation gateway failed to start")
                await asyncio.sleep(0.01)
            process: Final = await asyncio.create_subprocess_exec(
                str(binary.resolve()),
                env={
                    **os.environ,
                    "LITELLM_URL": f"http://127.0.0.1:{listener.getsockname()[1]}",
                    "LENS_WORKER_TOKEN": token,
                    "LITELLM_RELEASE_TAG": release,
                },
            )
            try:
                exit_code: Final = await process.wait()
                if exit_code != 0 or not completed.done():
                    raise RuntimeError(f"Rust worker exited without a result (exit {exit_code})")
                return completed.result()
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        finally:
            server.should_exit = True
            with suppress(asyncio.CancelledError):
                await serving
