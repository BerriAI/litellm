# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from ...._compat import cached_property
from .permissions import (
    Permissions,
    AsyncPermissions,
    PermissionsWithRawResponse,
    AsyncPermissionsWithRawResponse,
    PermissionsWithStreamingResponse,
    AsyncPermissionsWithStreamingResponse,
)
from ...._resource import SyncAPIResource, AsyncAPIResource

__all__ = ["Checkpoints", "AsyncCheckpoints"]


class Checkpoints(SyncAPIResource):
    @cached_property
    def permissions(self) -> Permissions:
        return Permissions(self._client)

    @cached_property
    def with_raw_response(self) -> CheckpointsWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return CheckpointsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> CheckpointsWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return CheckpointsWithStreamingResponse(self)


class AsyncCheckpoints(AsyncAPIResource):
    @cached_property
    def permissions(self) -> AsyncPermissions:
        return AsyncPermissions(self._client)

    @cached_property
    def with_raw_response(self) -> AsyncCheckpointsWithRawResponse:
        """
        This property can be used as a prefix for any HTTP method call to return
        the raw response object instead of the parsed content.

        For more information, see https://www.github.com/openai/openai-python#accessing-raw-response-data-eg-headers
        """
        return AsyncCheckpointsWithRawResponse(self)

    @cached_property
    def with_streaming_response(self) -> AsyncCheckpointsWithStreamingResponse:
        """
        An alternative to `.with_raw_response` that doesn't eagerly read the response body.

        For more information, see https://www.github.com/openai/openai-python#with_streaming_response
        """
        return AsyncCheckpointsWithStreamingResponse(self)


class CheckpointsWithRawResponse:
    def __init__(self, checkpoints: Checkpoints) -> None:
        self._checkpoints = checkpoints

    @cached_property
    def permissions(self) -> PermissionsWithRawResponse:
        return PermissionsWithRawResponse(self._checkpoints.permissions)


class AsyncCheckpointsWithRawResponse:
    def __init__(self, checkpoints: AsyncCheckpoints) -> None:
        self._checkpoints = checkpoints

    @cached_property
    def permissions(self) -> AsyncPermissionsWithRawResponse:
        return AsyncPermissionsWithRawResponse(self._checkpoints.permissions)


class CheckpointsWithStreamingResponse:
    def __init__(self, checkpoints: Checkpoints) -> None:
        self._checkpoints = checkpoints

    @cached_property
    def permissions(self) -> PermissionsWithStreamingResponse:
        return PermissionsWithStreamingResponse(self._checkpoints.permissions)


class AsyncCheckpointsWithStreamingResponse:
    def __init__(self, checkpoints: AsyncCheckpoints) -> None:
        self._checkpoints = checkpoints

    @cached_property
    def permissions(self) -> AsyncPermissionsWithStreamingResponse:
        return AsyncPermissionsWithStreamingResponse(self._checkpoints.permissions)
