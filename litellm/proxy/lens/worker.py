import asyncio
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import suppress
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .analysis import analyze_sample
from .models import Claim, Coverage, ExecutionContent, ModelRequest, ModelResult, Progress, Result, Sample

logger: Final = logging.getLogger("litellm.lens.worker")


class ClaimedJobIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str


class ClaimIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    lens_id: str
    job: ClaimedJobIdentity


def failure_message(error: Exception) -> str:
    if isinstance(error, (OSError, sqlite3.Error)):
        return "Worker temporary storage failed. Increase its capacity or reduce analysis parallelism."
    if isinstance(error, httpx.TimeoutException):
        return "The worker timed out waiting for the proxy. Check proxy availability and model response times."
    if isinstance(error, httpx.TransportError):
        return "The worker could not connect to the proxy. Check the proxy URL, network access, and TLS configuration."
    if isinstance(error, httpx.HTTPStatusError):
        path: Final = error.request.url.path
        action: Final = (
            "Model request"
            if path.endswith("/model")
            else "Reading trace data"
            if path.endswith(("/sample", "/content"))
            else "Saving results"
            if path.endswith("/result")
            else "Worker request"
        )
        status: Final = error.response.status_code
        guidance: Final = MappingProxyType(
            {
                400: "Check the configured model and whether the worker's billing key is enabled.",
                401: "Check the worker credential and its assigned billing key.",
                402: "Check the investigation's monthly limit and the worker key's remaining budget.",
                403: "Check the worker key's model permissions and access restrictions.",
                404: "Check that the proxy and worker versions match and the requested model is configured.",
                409: "This worker no longer owns the run. Check whether it was cancelled or claimed again.",
                429: "The request was rate limited. Retry later or check the worker key's rate limits.",
            }
        ).get(status, "Check proxy and model availability, then retry the investigation.")
        return f"{action} failed (HTTP {status}). {guidance}"
    return "The worker could not read an analysis response. Check structured JSON support and matching proxy/worker versions."


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
        payload: Final = response.json()
        if payload is None:
            return False
        try:
            claim: Final = Claim.model_validate(payload)
        except ValidationError:
            identity: Final = ClaimIdentity.model_validate(payload)
            failure: Final = await self.client.post(
                f"/lens/worker/{identity.lens_id}/{identity.job.id}/result",
                json=Result(
                    coverage=Coverage(),
                    error="The worker could not read this investigation. Update the worker to match the gateway, then retry.",
                ).model_dump(),
            )
            if failure.status_code != 409:
                failure.raise_for_status()
            logger.warning("Worker could not read a claimed investigation; reported a version compatibility failure")
            return True
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
            message: Final = failure_message(exc)
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
