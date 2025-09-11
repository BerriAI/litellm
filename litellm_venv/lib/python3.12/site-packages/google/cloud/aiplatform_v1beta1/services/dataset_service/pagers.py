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

from google.cloud.aiplatform_v1beta1.types import annotation
from google.cloud.aiplatform_v1beta1.types import data_item
from google.cloud.aiplatform_v1beta1.types import dataset
from google.cloud.aiplatform_v1beta1.types import dataset_service
from google.cloud.aiplatform_v1beta1.types import dataset_version
from google.cloud.aiplatform_v1beta1.types import saved_query


class ListDatasetsPager:
    """A pager for iterating through ``list_datasets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``datasets`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDatasets`` requests and continue to iterate
    through the ``datasets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.ListDatasetsResponse],
        request: dataset_service.ListDatasetsRequest,
        response: dataset_service.ListDatasetsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDatasetsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDatasetsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.ListDatasetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[dataset.Dataset]:
        for page in self.pages:
            yield from page.datasets

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatasetsAsyncPager:
    """A pager for iterating through ``list_datasets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``datasets`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDatasets`` requests and continue to iterate
    through the ``datasets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.ListDatasetsResponse]],
        request: dataset_service.ListDatasetsRequest,
        response: dataset_service.ListDatasetsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDatasetsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDatasetsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDatasetsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.ListDatasetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[dataset.Dataset]:
        async def async_generator():
            async for page in self.pages:
                for response in page.datasets:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatasetVersionsPager:
    """A pager for iterating through ``list_dataset_versions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``dataset_versions`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDatasetVersions`` requests and continue to iterate
    through the ``dataset_versions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.ListDatasetVersionsResponse],
        request: dataset_service.ListDatasetVersionsRequest,
        response: dataset_service.ListDatasetVersionsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDatasetVersionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.ListDatasetVersionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[dataset_version.DatasetVersion]:
        for page in self.pages:
            yield from page.dataset_versions

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDatasetVersionsAsyncPager:
    """A pager for iterating through ``list_dataset_versions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``dataset_versions`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDatasetVersions`` requests and continue to iterate
    through the ``dataset_versions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.ListDatasetVersionsResponse]],
        request: dataset_service.ListDatasetVersionsRequest,
        response: dataset_service.ListDatasetVersionsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDatasetVersionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDatasetVersionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.ListDatasetVersionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[dataset_version.DatasetVersion]:
        async def async_generator():
            async for page in self.pages:
                for response in page.dataset_versions:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDataItemsPager:
    """A pager for iterating through ``list_data_items`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``data_items`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDataItems`` requests and continue to iterate
    through the ``data_items`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.ListDataItemsResponse],
        request: dataset_service.ListDataItemsRequest,
        response: dataset_service.ListDataItemsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDataItemsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDataItemsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.ListDataItemsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[data_item.DataItem]:
        for page in self.pages:
            yield from page.data_items

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDataItemsAsyncPager:
    """A pager for iterating through ``list_data_items`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``data_items`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDataItems`` requests and continue to iterate
    through the ``data_items`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.ListDataItemsResponse]],
        request: dataset_service.ListDataItemsRequest,
        response: dataset_service.ListDataItemsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDataItemsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDataItemsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListDataItemsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.ListDataItemsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[data_item.DataItem]:
        async def async_generator():
            async for page in self.pages:
                for response in page.data_items:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchDataItemsPager:
    """A pager for iterating through ``search_data_items`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``data_item_views`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``SearchDataItems`` requests and continue to iterate
    through the ``data_item_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.SearchDataItemsResponse],
        request: dataset_service.SearchDataItemsRequest,
        response: dataset_service.SearchDataItemsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchDataItemsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.SearchDataItemsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.SearchDataItemsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[dataset_service.DataItemView]:
        for page in self.pages:
            yield from page.data_item_views

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchDataItemsAsyncPager:
    """A pager for iterating through ``search_data_items`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``data_item_views`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``SearchDataItems`` requests and continue to iterate
    through the ``data_item_views`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.SearchDataItemsResponse]],
        request: dataset_service.SearchDataItemsRequest,
        response: dataset_service.SearchDataItemsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.SearchDataItemsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.SearchDataItemsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.SearchDataItemsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.SearchDataItemsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[dataset_service.DataItemView]:
        async def async_generator():
            async for page in self.pages:
                for response in page.data_item_views:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListSavedQueriesPager:
    """A pager for iterating through ``list_saved_queries`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``saved_queries`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListSavedQueries`` requests and continue to iterate
    through the ``saved_queries`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.ListSavedQueriesResponse],
        request: dataset_service.ListSavedQueriesRequest,
        response: dataset_service.ListSavedQueriesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListSavedQueriesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListSavedQueriesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.ListSavedQueriesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[saved_query.SavedQuery]:
        for page in self.pages:
            yield from page.saved_queries

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListSavedQueriesAsyncPager:
    """A pager for iterating through ``list_saved_queries`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``saved_queries`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListSavedQueries`` requests and continue to iterate
    through the ``saved_queries`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.ListSavedQueriesResponse]],
        request: dataset_service.ListSavedQueriesRequest,
        response: dataset_service.ListSavedQueriesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListSavedQueriesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListSavedQueriesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListSavedQueriesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.ListSavedQueriesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[saved_query.SavedQuery]:
        async def async_generator():
            async for page in self.pages:
                for response in page.saved_queries:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListAnnotationsPager:
    """A pager for iterating through ``list_annotations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``annotations`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListAnnotations`` requests and continue to iterate
    through the ``annotations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., dataset_service.ListAnnotationsResponse],
        request: dataset_service.ListAnnotationsRequest,
        response: dataset_service.ListAnnotationsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListAnnotationsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListAnnotationsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[dataset_service.ListAnnotationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[annotation.Annotation]:
        for page in self.pages:
            yield from page.annotations

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListAnnotationsAsyncPager:
    """A pager for iterating through ``list_annotations`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``annotations`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListAnnotations`` requests and continue to iterate
    through the ``annotations`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[dataset_service.ListAnnotationsResponse]],
        request: dataset_service.ListAnnotationsRequest,
        response: dataset_service.ListAnnotationsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListAnnotationsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListAnnotationsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = dataset_service.ListAnnotationsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[dataset_service.ListAnnotationsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[annotation.Annotation]:
        async def async_generator():
            async for page in self.pages:
                for response in page.annotations:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
