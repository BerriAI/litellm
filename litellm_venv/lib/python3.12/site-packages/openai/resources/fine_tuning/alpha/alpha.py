# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from .graders import (
    Graders,
    AsyncGraders,
    GradersWithRawResponse,
    AsyncGradersWithRawResponse,
    GradersWithStreamingResponse,
    AsyncGradersWithStreamingResponse,
)
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource

__all__ = ["Alpha", "AsyncAlpha"]


class Alpha(SyncAPIResource):
    @cached_property
    def graders(self) -> Graders:
        return Graders(self._client)

    @cached_property
    def with_raw_response(self) -> AlphaWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return AlphaWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AlphaWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return AlphaWithStreamingResponse(self)


class AsyncAlpha(AsyncAPIResource):
    @cached_property
    def graders(self) -> AsyncGraders:
        return AsyncGraders(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncAlphaWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return AsyncAlphaWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncAlphaWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return AsyncAlphaWithStreamingResponse(self)


class AlphaWithRawResponse:
    def __init__(self, alpha: Alpha) -> None:
        self._alpha = alpha

    @cached_property
    def graders(self) -> GradersWithRawResponse:
        return GradersWithRawResponse(self._alpha.graders)


class AsyncAlphaWithRawResponse:
    def __init__(self, alpha: AsyncAlpha) -> None:
        self._alpha = alpha

    @cached_property
    def graders(self) -> AsyncGradersWithRawResponse:
        return AsyncGradersWithRawResponse(self._alpha.graders)


class AlphaWithStreamingResponse:
    def __init__(self, alpha: Alpha) -> None:
        self._alpha = alpha

    @cached_property
    def graders(self) -> GradersWithStreamingResponse:
        return GradersWithStreamingResponse(self._alpha.graders)


class AsyncAlphaWithStreamingResponse:
    def __init__(self, alpha: AsyncAlpha) -> None:
        self._alpha = alpha

    @cached_property
    def graders(self) -> AsyncGradersWithStreamingResponse:
        return AsyncGradersWithStreamingResponse(self._alpha.graders)
