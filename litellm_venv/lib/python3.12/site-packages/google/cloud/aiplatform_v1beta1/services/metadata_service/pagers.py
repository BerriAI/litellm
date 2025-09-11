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

from google.cloud.aiplatform_v1beta1.types import artifact
from google.cloud.aiplatform_v1beta1.types import context
from google.cloud.aiplatform_v1beta1.types import execution
from google.cloud.aiplatform_v1beta1.types import metadata_schema
from google.cloud.aiplatform_v1beta1.types import metadata_service
from google.cloud.aiplatform_v1beta1.types import metadata_store


class ListMetadataStoresPager:
    """A pager for iterating through ``list_metadata_stores`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``metadata_stores`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListMetadataStores`` requests and continue to iterate
    through the ``metadata_stores`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., metadata_service.ListMetadataStoresResponse],
        request: metadata_service.ListMetadataStoresRequest,
        response: metadata_service.ListMetadataStoresResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListMetadataStoresRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListMetadataStoresRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[metadata_service.ListMetadataStoresResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[metadata_store.MetadataStore]:
        for page in self.pages:
            yield from page.metadata_stores

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListMetadataStoresAsyncPager:
    """A pager for iterating through ``list_metadata_stores`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``metadata_stores`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListMetadataStores`` requests and continue to iterate
    through the ``metadata_stores`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[metadata_service.ListMetadataStoresResponse]],
        request: metadata_service.ListMetadataStoresRequest,
        response: metadata_service.ListMetadataStoresResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListMetadataStoresRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListMetadataStoresResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListMetadataStoresRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[metadata_service.ListMetadataStoresResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[metadata_store.MetadataStore]:
        async def async_generator():
            async for page in self.pages:
                for response in page.metadata_stores:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListArtifactsPager:
    """A pager for iterating through ``list_artifacts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``artifacts`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListArtifacts`` requests and continue to iterate
    through the ``artifacts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., metadata_service.ListArtifactsResponse],
        request: metadata_service.ListArtifactsRequest,
        response: metadata_service.ListArtifactsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListArtifactsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListArtifactsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[metadata_service.ListArtifactsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[artifact.Artifact]:
        for page in self.pages:
            yield from page.artifacts

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListArtifactsAsyncPager:
    """A pager for iterating through ``list_artifacts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``artifacts`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListArtifacts`` requests and continue to iterate
    through the ``artifacts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[metadata_service.ListArtifactsResponse]],
        request: metadata_service.ListArtifactsRequest,
        response: metadata_service.ListArtifactsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListArtifactsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListArtifactsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListArtifactsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[metadata_service.ListArtifactsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[artifact.Artifact]:
        async def async_generator():
            async for page in self.pages:
                for response in page.artifacts:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListContextsPager:
    """A pager for iterating through ``list_contexts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListContextsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``contexts`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListContexts`` requests and continue to iterate
    through the ``contexts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListContextsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., metadata_service.ListContextsResponse],
        request: metadata_service.ListContextsRequest,
        response: metadata_service.ListContextsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListContextsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListContextsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListContextsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[metadata_service.ListContextsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[context.Context]:
        for page in self.pages:
            yield from page.contexts

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListContextsAsyncPager:
    """A pager for iterating through ``list_contexts`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListContextsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``contexts`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListContexts`` requests and continue to iterate
    through the ``contexts`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListContextsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[metadata_service.ListContextsResponse]],
        request: metadata_service.ListContextsRequest,
        response: metadata_service.ListContextsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListContextsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListContextsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListContextsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[metadata_service.ListContextsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[context.Context]:
        async def async_generator():
            async for page in self.pages:
                for response in page.contexts:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListExecutionsPager:
    """A pager for iterating through ``list_executions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``executions`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListExecutions`` requests and continue to iterate
    through the ``executions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., metadata_service.ListExecutionsResponse],
        request: metadata_service.ListExecutionsRequest,
        response: metadata_service.ListExecutionsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListExecutionsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListExecutionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[metadata_service.ListExecutionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[execution.Execution]:
        for page in self.pages:
            yield from page.executions

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListExecutionsAsyncPager:
    """A pager for iterating through ``list_executions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``executions`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListExecutions`` requests and continue to iterate
    through the ``executions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[metadata_service.ListExecutionsResponse]],
        request: metadata_service.ListExecutionsRequest,
        response: metadata_service.ListExecutionsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListExecutionsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListExecutionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListExecutionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[metadata_service.ListExecutionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[execution.Execution]:
        async def async_generator():
            async for page in self.pages:
                for response in page.executions:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListMetadataSchemasPager:
    """A pager for iterating through ``list_metadata_schemas`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``metadata_schemas`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListMetadataSchemas`` requests and continue to iterate
    through the ``metadata_schemas`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., metadata_service.ListMetadataSchemasResponse],
        request: metadata_service.ListMetadataSchemasRequest,
        response: metadata_service.ListMetadataSchemasResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListMetadataSchemasRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[metadata_service.ListMetadataSchemasResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[metadata_schema.MetadataSchema]:
        for page in self.pages:
            yield from page.metadata_schemas

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListMetadataSchemasAsyncPager:
    """A pager for iterating through ``list_metadata_schemas`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``metadata_schemas`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListMetadataSchemas`` requests and continue to iterate
    through the ``metadata_schemas`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[metadata_service.ListMetadataSchemasResponse]],
        request: metadata_service.ListMetadataSchemasRequest,
        response: metadata_service.ListMetadataSchemasResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListMetadataSchemasResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = metadata_service.ListMetadataSchemasRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[metadata_service.ListMetadataSchemasResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[metadata_schema.MetadataSchema]:
        async def async_generator():
            async for page in self.pages:
                for response in page.metadata_schemas:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
