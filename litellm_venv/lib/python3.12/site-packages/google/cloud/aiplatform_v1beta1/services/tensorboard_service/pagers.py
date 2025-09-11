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

from google.cloud.aiplatform_v1beta1.types import tensorboard
from google.cloud.aiplatform_v1beta1.types import tensorboard_data
from google.cloud.aiplatform_v1beta1.types import tensorboard_experiment
from google.cloud.aiplatform_v1beta1.types import tensorboard_run
from google.cloud.aiplatform_v1beta1.types import tensorboard_service
from google.cloud.aiplatform_v1beta1.types import tensorboard_time_series


class ListTensorboardsPager:
    """A pager for iterating through ``list_tensorboards`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``tensorboards`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTensorboards`` requests and continue to iterate
    through the ``tensorboards`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., tensorboard_service.ListTensorboardsResponse],
        request: tensorboard_service.ListTensorboardsRequest,
        response: tensorboard_service.ListTensorboardsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[tensorboard_service.ListTensorboardsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[tensorboard.Tensorboard]:
        for page in self.pages:
            yield from page.tensorboards

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardsAsyncPager:
    """A pager for iterating through ``list_tensorboards`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``tensorboards`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListTensorboards`` requests and continue to iterate
    through the ``tensorboards`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[tensorboard_service.ListTensorboardsResponse]],
        request: tensorboard_service.ListTensorboardsRequest,
        response: tensorboard_service.ListTensorboardsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[tensorboard_service.ListTensorboardsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[tensorboard.Tensorboard]:
        async def async_generator():
            async for page in self.pages:
                for response in page.tensorboards:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardExperimentsPager:
    """A pager for iterating through ``list_tensorboard_experiments`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``tensorboard_experiments`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTensorboardExperiments`` requests and continue to iterate
    through the ``tensorboard_experiments`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., tensorboard_service.ListTensorboardExperimentsResponse],
        request: tensorboard_service.ListTensorboardExperimentsRequest,
        response: tensorboard_service.ListTensorboardExperimentsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardExperimentsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[tensorboard_service.ListTensorboardExperimentsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[tensorboard_experiment.TensorboardExperiment]:
        for page in self.pages:
            yield from page.tensorboard_experiments

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardExperimentsAsyncPager:
    """A pager for iterating through ``list_tensorboard_experiments`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``tensorboard_experiments`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListTensorboardExperiments`` requests and continue to iterate
    through the ``tensorboard_experiments`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[tensorboard_service.ListTensorboardExperimentsResponse]
        ],
        request: tensorboard_service.ListTensorboardExperimentsRequest,
        response: tensorboard_service.ListTensorboardExperimentsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardExperimentsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardExperimentsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[tensorboard_service.ListTensorboardExperimentsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[tensorboard_experiment.TensorboardExperiment]:
        async def async_generator():
            async for page in self.pages:
                for response in page.tensorboard_experiments:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardRunsPager:
    """A pager for iterating through ``list_tensorboard_runs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``tensorboard_runs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTensorboardRuns`` requests and continue to iterate
    through the ``tensorboard_runs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., tensorboard_service.ListTensorboardRunsResponse],
        request: tensorboard_service.ListTensorboardRunsRequest,
        response: tensorboard_service.ListTensorboardRunsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardRunsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[tensorboard_service.ListTensorboardRunsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[tensorboard_run.TensorboardRun]:
        for page in self.pages:
            yield from page.tensorboard_runs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardRunsAsyncPager:
    """A pager for iterating through ``list_tensorboard_runs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``tensorboard_runs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListTensorboardRuns`` requests and continue to iterate
    through the ``tensorboard_runs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[tensorboard_service.ListTensorboardRunsResponse]
        ],
        request: tensorboard_service.ListTensorboardRunsRequest,
        response: tensorboard_service.ListTensorboardRunsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardRunsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardRunsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[tensorboard_service.ListTensorboardRunsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[tensorboard_run.TensorboardRun]:
        async def async_generator():
            async for page in self.pages:
                for response in page.tensorboard_runs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardTimeSeriesPager:
    """A pager for iterating through ``list_tensorboard_time_series`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``tensorboard_time_series`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTensorboardTimeSeries`` requests and continue to iterate
    through the ``tensorboard_time_series`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., tensorboard_service.ListTensorboardTimeSeriesResponse],
        request: tensorboard_service.ListTensorboardTimeSeriesRequest,
        response: tensorboard_service.ListTensorboardTimeSeriesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardTimeSeriesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[tensorboard_service.ListTensorboardTimeSeriesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[tensorboard_time_series.TensorboardTimeSeries]:
        for page in self.pages:
            yield from page.tensorboard_time_series

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTensorboardTimeSeriesAsyncPager:
    """A pager for iterating through ``list_tensorboard_time_series`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``tensorboard_time_series`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListTensorboardTimeSeries`` requests and continue to iterate
    through the ``tensorboard_time_series`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[tensorboard_service.ListTensorboardTimeSeriesResponse]
        ],
        request: tensorboard_service.ListTensorboardTimeSeriesRequest,
        response: tensorboard_service.ListTensorboardTimeSeriesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListTensorboardTimeSeriesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ListTensorboardTimeSeriesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[tensorboard_service.ListTensorboardTimeSeriesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[tensorboard_time_series.TensorboardTimeSeries]:
        async def async_generator():
            async for page in self.pages:
                for response in page.tensorboard_time_series:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ExportTensorboardTimeSeriesDataPager:
    """A pager for iterating through ``export_tensorboard_time_series_data`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``time_series_data_points`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ExportTensorboardTimeSeriesData`` requests and continue to iterate
    through the ``time_series_data_points`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., tensorboard_service.ExportTensorboardTimeSeriesDataResponse
        ],
        request: tensorboard_service.ExportTensorboardTimeSeriesDataRequest,
        response: tensorboard_service.ExportTensorboardTimeSeriesDataResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ExportTensorboardTimeSeriesDataRequest(
            request
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[tensorboard_service.ExportTensorboardTimeSeriesDataResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[tensorboard_data.TimeSeriesDataPoint]:
        for page in self.pages:
            yield from page.time_series_data_points

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ExportTensorboardTimeSeriesDataAsyncPager:
    """A pager for iterating through ``export_tensorboard_time_series_data`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``time_series_data_points`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ExportTensorboardTimeSeriesData`` requests and continue to iterate
    through the ``time_series_data_points`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[tensorboard_service.ExportTensorboardTimeSeriesDataResponse]
        ],
        request: tensorboard_service.ExportTensorboardTimeSeriesDataRequest,
        response: tensorboard_service.ExportTensorboardTimeSeriesDataResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ExportTensorboardTimeSeriesDataResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = tensorboard_service.ExportTensorboardTimeSeriesDataRequest(
            request
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[tensorboard_service.ExportTensorboardTimeSeriesDataResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[tensorboard_data.TimeSeriesDataPoint]:
        async def async_generator():
            async for page in self.pages:
                for response in page.time_series_data_points:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
