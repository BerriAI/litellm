# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from ..._compat import cached_property
from .jobs.jobs import (
    Jobs,
    AsyncJobs,
    JobsWithRawResponse,
    AsyncJobsWithRawResponse,
    JobsWithStreamingResponse,
    AsyncJobsWithStreamingResponse,
)
from ..._resource import SyncAPIResource, AsyncAPIResource
from .alpha.alpha import (
    Alpha,
    AsyncAlpha,
    AlphaWithRawResponse,
    AsyncAlphaWithRawResponse,
    AlphaWithStreamingResponse,
    AsyncAlphaWithStreamingResponse,
)
from .checkpoints.checkpoints import (
    Checkpoints,
    AsyncCheckpoints,
    CheckpointsWithRawResponse,
    AsyncCheckpointsWithRawResponse,
    CheckpointsWithStreamingResponse,
    AsyncCheckpointsWithStreamingResponse,
)

__all__ = ["FineTuning", "AsyncFineTuning"]


class FineTuning(SyncAPIResource):
    @cached_property
    def jobs(self) -> Jobs:
        return Jobs(self._client)

    @cached_property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(self._client)

    @cached_property
    def alpha(self) -> Alpha:
        return Alpha(self._client)

    @cached_property
    def with_raw_response(self) -> FineTuningWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return FineTuningWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> FineTuningWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return FineTuningWithStreamingResponse(self)


class AsyncFineTuning(AsyncAPIResource):
    @cached_property
    def jobs(self) -> AsyncJobs:
        return AsyncJobs(self._client)

    @cached_property
    def checkpoints(self) -> AsyncCheckpoints:
        return AsyncCheckpoints(self._client)

    @cached_property
    def alpha(self) -> AsyncAlpha:
        return AsyncAlpha(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncFineTuningWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return AsyncFineTuningWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncFineTuningWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return AsyncFineTuningWithStreamingResponse(self)


class FineTuningWithRawResponse:
    def __init__(self, fine_tuning: FineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> JobsWithRawResponse:
        return JobsWithRawResponse(self._fine_tuning.jobs)

    @cached_property
    def checkpoints(self) -> CheckpointsWithRawResponse:
        return CheckpointsWithRawResponse(self._fine_tuning.checkpoints)

    @cached_property
    def alpha(self) -> AlphaWithRawResponse:
        return AlphaWithRawResponse(self._fine_tuning.alpha)


class AsyncFineTuningWithRawResponse:
    def __init__(self, fine_tuning: AsyncFineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> AsyncJobsWithRawResponse:
        return AsyncJobsWithRawResponse(self._fine_tuning.jobs)

    @cached_property
    def checkpoints(self) -> AsyncCheckpointsWithRawResponse:
        return AsyncCheckpointsWithRawResponse(self._fine_tuning.checkpoints)

    @cached_property
    def alpha(self) -> AsyncAlphaWithRawResponse:
        return AsyncAlphaWithRawResponse(self._fine_tuning.alpha)


class FineTuningWithStreamingResponse:
    def __init__(self, fine_tuning: FineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> JobsWithStreamingResponse:
        return JobsWithStreamingResponse(self._fine_tuning.jobs)

    @cached_property
    def checkpoints(self) -> CheckpointsWithStreamingResponse:
        return CheckpointsWithStreamingResponse(self._fine_tuning.checkpoints)

    @cached_property
    def alpha(self) -> AlphaWithStreamingResponse:
        return AlphaWithStreamingResponse(self._fine_tuning.alpha)


class AsyncFineTuningWithStreamingResponse:
    def __init__(self, fine_tuning: AsyncFineTuning) -> None:
        self._fine_tuning = fine_tuning

    @cached_property
    def jobs(self) -> AsyncJobsWithStreamingResponse:
        return AsyncJobsWithStreamingResponse(self._fine_tuning.jobs)

    @cached_property
    def checkpoints(self) -> AsyncCheckpointsWithStreamingResponse:
        return AsyncCheckpointsWithStreamingResponse(self._fine_tuning.checkpoints)

    @cached_property
    def alpha(self) -> AsyncAlphaWithStreamingResponse:
        return AsyncAlphaWithStreamingResponse(self._fine_tuning.alpha)
