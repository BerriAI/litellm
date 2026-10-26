import asyncio
import re
from collections.abc import Mapping
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class FinalResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    class_name: Literal["benign", "attack"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class InjectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    level: Literal["L3"]
    final_result: FinalResult


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)
    state: Literal["complete", "degraded", "failed"]
    failures: tuple[object, ...] = ()


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)
    job_id: str
    status: Literal["running", "completed", "failed"]
    completion: Completion | None = None
    categories: Mapping[str, InjectionResult] = Field(default_factory=dict)


class Submission(BaseModel):
    model_config = ConfigDict(frozen=True)
    jobs: tuple[Job, ...] = Field(min_length=1, max_length=1)


class WolfAPI:
    def __init__(self, client: httpx.AsyncClient, api_base: str, api_key: str) -> None:
        self.client = client
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key

    async def scan(self, text: str, timeout: float) -> bool:
        deadline: Final = asyncio.get_running_loop().time() + timeout
        response: Final = await self.client.post(
            f"{self.api_base}/api/v1/scan",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "text": text,
                "config": {
                    "categories": ["injection"],
                    "max_level": "L3",
                    "gates": {"l1": False, "l2": False, "l3": True},
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        if response.status_code not in (200, 202):
            raise ValueError("Unexpected Wolf submission status")
        submission: Final = Submission.model_validate_json(response.content)
        return await self._finish(submission.jobs[0], deadline)

    async def _finish(self, job: Job, deadline: float) -> bool:
        if job.status == "failed":
            raise ValueError("Wolf scan failed")
        if job.status == "completed":
            if job.completion is None or job.completion.state != "complete" or job.completion.failures:
                raise ValueError("Wolf scan is incomplete")
            result: Final = job.categories.get("injection")
            if result is None:
                raise ValueError("Wolf injection result is missing")
            return result.final_result.class_name == "attack"
        if re.fullmatch(r"job_[0-9a-f]{32}", job.job_id) is None:
            raise ValueError("Invalid Wolf job ID")
        remaining: Final = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Wolf scan deadline exceeded")
        await asyncio.sleep(min(0.1, remaining))
        poll_budget: Final = deadline - asyncio.get_running_loop().time()
        if poll_budget <= 0:
            raise TimeoutError("Wolf scan deadline exceeded")
        response: Final = await self.client.get(
            f"{self.api_base}/api/v1/scan/{job.job_id}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=poll_budget,
        )
        response.raise_for_status()
        next_job: Final = Job.model_validate_json(response.content)
        if next_job.job_id != job.job_id:
            raise ValueError("Wolf job ID changed")
        return await self._finish(next_job, deadline)
