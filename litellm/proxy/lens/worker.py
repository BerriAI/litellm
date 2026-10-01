import asyncio
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import MappingProxyType
from typing import Final

import httpx

from .analysis import analyze_sample
from .models import Claim, Coverage, ExecutionContent, ModelRequest, ModelResult, Progress, Result, Sample

logger: Final = logging.getLogger("litellm.lens.worker")


class LensWorker:
    def __init__(self, client: httpx.AsyncClient, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.client: Final = client
        self.sleep: Final = sleep

    async def model_request(self, path: str, body: ModelRequest, attempt: int = 0) -> ModelResult:
        try:
            result: Final = await self.client.post(path, json=body.model_dump())
            result.raise_for_status()
            return ModelResult.model_validate(result.json())
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable: Final = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code in (
                429,
                502,
                503,
                504,
            )
            if not retryable or attempt >= 2:
                raise
            await self.sleep(2**attempt)
            return await self.model_request(path, body, attempt + 1)

    async def run_once(self) -> bool:
        response: Final = await self.client.post("/lens/worker/claim", params=MappingProxyType({"protocol_version": 2}))
        response.raise_for_status()
        if response.json() is None:
            return False
        claim: Final = Claim.model_validate(response.json())
        prefix: Final = f"/lens/worker/{claim.lens_id}/{claim.job.id}"

        async def model(body: ModelRequest) -> ModelResult:
            return await self.model_request(prefix + "/model", body)

        async def read(execution_id: str, cursor: str, offset: int) -> ExecutionContent:
            result: Final = await self.client.get(
                prefix + "/content",
                params=MappingProxyType(
                    {
                        "execution_id": execution_id,
                        "cursor": cursor,
                        "offset": offset,
                    }
                ),
            )
            result.raise_for_status()
            return ExecutionContent.model_validate(result.json())

        async def progress(stage: str, coverage: Coverage) -> None:
            result: Final = await self.client.post(
                prefix + "/progress", json=Progress(stage=stage, coverage=coverage).model_dump()
            )
            result.raise_for_status()

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(30)
                (await self.client.post(prefix + "/heartbeat")).raise_for_status()

        pulse_task: Final = asyncio.create_task(heartbeat())
        try:
            data: Final = await self.client.get(prefix + "/sample")
            data.raise_for_status()
            sample: Final = Sample.model_validate(data.json())
            result: Final = await analyze_sample(claim, sample, read, model, progress)
            saved: Final = await self.client.post(prefix + "/result", json=result.model_dump(mode="json"))
            saved.raise_for_status()
        except (httpx.HTTPError, ValueError, OSError, sqlite3.Error) as exc:
            status: Final = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            message: Final = (
                "Worker temporary storage failed. Increase its capacity or reduce analysis parallelism."
                if isinstance(exc, (OSError, sqlite3.Error))
                else "Monthly budget reached"
                if status == 402
                else "Analysis interrupted. Check worker connectivity, model configuration, and trace storage."
            )
            logger.warning("Analysis %s interrupted (%s)", claim.job.id, type(exc).__name__)
            failed: Final = await self.client.post(
                prefix + "/result", json=Result(coverage=Coverage(), error=message).model_dump()
            )
            if failed.status_code != 409:
                failed.raise_for_status()
        finally:
            pulse_task.cancel()
            with suppress(asyncio.CancelledError, httpx.HTTPError):
                await pulse_task
        return True


async def main() -> None:
    url: Final = os.environ["LITELLM_URL"].rstrip("/")
    token: Final = os.environ["LENS_WORKER_TOKEN"]
    async with httpx.AsyncClient(
        base_url=url, headers=MappingProxyType({"Authorization": f"Bearer {token}"}), timeout=180
    ) as client:
        worker: Final = LensWorker(client)
        while True:
            try:
                await worker.run_once()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Worker could not reach Lens (%s)", type(exc).__name__)
            await asyncio.sleep(10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
