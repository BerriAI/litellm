# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import TYPE_CHECKING

from .jobs import Jobs, AsyncJobs, JobsWithRawResponse, AsyncJobsWithRawResponse
from ..._resource import SyncAPIResource, AsyncAPIResource

if TYPE_CHECKING:
    from ..._client import OpenAI, AsyncOpenAI

__all__ = ["FineTuning", "AsyncFineTuning"]


class FineTuning(SyncAPIResource):
    jobs: Jobs
    with_raw_response: FineTuningWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.jobs = Jobs(client)
        self.with_raw_response = FineTuningWithRawResponse(self)


class AsyncFineTuning(AsyncAPIResource):
    jobs: AsyncJobs
    with_raw_response: AsyncFineTuningWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(client)
        self.jobs = AsyncJobs(client)
        self.with_raw_response = AsyncFineTuningWithRawResponse(self)


class FineTuningWithRawResponse:
    def __init__(self, fine_tuning: FineTuning) -> None:
        self.jobs = JobsWithRawResponse(fine_tuning.jobs)


class AsyncFineTuningWithRawResponse:
    def __init__(self, fine_tuning: AsyncFineTuning) -> None:
        self.jobs = AsyncJobsWithRawResponse(fine_tuning.jobs)
