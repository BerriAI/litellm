# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Sequence,
    Tuple,
    Optional,
    Iterator,
)

from google.cloud.aiplatform_v1.types import schedule
from google.cloud.aiplatform_v1.types import schedule_service


class ListSchedulesPager:
    """A pager for iterating through ``list_schedules`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListSchedulesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``schedules`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListSchedules`` requests and continue to iterate
    through the ``schedules`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListSchedulesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., schedule_service.ListSchedulesResponse],
        request: schedule_service.ListSchedulesRequest,
        response: schedule_service.ListSchedulesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListSchedulesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListSchedulesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = schedule_service.ListSchedulesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[schedule_service.ListSchedulesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[schedule.Schedule]:
        for page in self.pages:
            yield from page.schedules

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListSchedulesAsyncPager:
    """A pager for iterating through ``list_schedules`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListSchedulesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``schedules`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListSchedules`` requests and continue to iterate
    through the ``schedules`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListSchedulesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[schedule_service.ListSchedulesResponse]],
        request: schedule_service.ListSchedulesRequest,
        response: schedule_service.ListSchedulesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListSchedulesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListSchedulesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = schedule_service.ListSchedulesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[schedule_service.ListSchedulesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[schedule.Schedule]:
        async def async_generator():
            async for page in self.pages:
                for response in page.schedules:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
