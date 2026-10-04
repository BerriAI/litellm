import asyncio
import logging
import os
import sqlite3
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from .analysis import AnalysisResponseError, analyze_sample, validation_details
from .models import Claim, Coverage, ExecutionContent, ModelRequest, ModelResult, Progress, Result, Sample
from .release import PROTOCOL_VERSION, release_tag

logger: Final = logging.getLogger("litellm.lens.worker")


class ClaimedJobIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str


class ClaimIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    lens_id: str
    job: ClaimedJobIdentity


class PublicModelError(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lens_error: str


class ModelErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    detail: PublicModelError


def failure_message(error: Exception) -> str:
    if isinstance(error, AnalysisResponseError):
        return str(error)
    if isinstance(error, ValidationError):
        return f"Invalid {error.title} response (ValidationError):\n{validation_details(error)}"
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
        if path.endswith("/model"):
            try:
                diagnostic: Final = ModelErrorEnvelope.model_validate_json(error.response.content)
                return f"Model request failed (HTTP {status}):\n{diagnostic.detail.lens_error}"
            except ValueError:
                pass
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
    def __init__(
        self,
        client: httpx.AsyncClient,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        heartbeat_wait: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.client: Final = client
        self.sleep: Final = sleep
        self.heartbeat_wait: Final = heartbeat_wait

    async def model_request(self, path: str, body: ModelRequest, attempt: int = 0) -> ModelResult:
        try:
            timeout: Final = httpx.Timeout(
                None,
                connect=self.client.timeout.connect,
                write=self.client.timeout.write,
                pool=self.client.timeout.pool,
            )
            result: Final = await self.client.post(path, json=body.model_dump(), timeout=timeout)
            result.raise_for_status()
            parsed: Final = ModelResult.model_validate(result.json())
            reason: Final = result.headers.get("x-litellm-lens-finish-reason")
            return (
                parsed.model_copy(update=MappingProxyType({"finish_reason": reason}))
                if reason in ("length", "content_filter")
                else parsed
            )
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

    async def report_unreadable_claim(self, identity: ClaimIdentity) -> None:
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

    async def run_once(self) -> bool:
        response: Final = await self.client.post(
            "/lens/worker/claim",
            params=MappingProxyType({"protocol_version": str(PROTOCOL_VERSION), "worker_release": release_tag()}),
        )
        if response.status_code == 409:
            logger.warning("Lens worker cannot claim work: %s", response.text)
            return False
        response.raise_for_status()
        payload: Final = response.json()
        if payload is None:
            return False
        try:
            claim: Final = Claim.model_validate(payload)
        except ValidationError:
            await self.report_unreadable_claim(ClaimIdentity.model_validate(payload))
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
                await self.heartbeat_wait(30)
                try:
                    (await self.client.post(prefix + "/heartbeat")).raise_for_status()
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and (
                        exc.response.status_code < 500 and exc.response.status_code != 429
                    ):
                        raise
                    logger.warning("Analysis %s heartbeat will retry (%s)", claim.job.id, type(exc).__name__)

        async def investigate() -> None:
            data: Final = await self.client.get(prefix + "/sample")
            data.raise_for_status()
            sample: Final = Sample.model_validate(data.json())
            result: Final = await analyze_sample(claim, sample, read, model, progress)
            saved: Final = await self.client.post(prefix + "/result", json=result.model_dump(mode="json"))
            saved.raise_for_status()

        pulse_task: Final = asyncio.create_task(heartbeat())
        work_task: Final = asyncio.create_task(investigate())
        try:
            finished, _ = await asyncio.wait((pulse_task, work_task), return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                await task
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
            work_task.cancel()
            await asyncio.gather(pulse_task, work_task, return_exceptions=True)
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
