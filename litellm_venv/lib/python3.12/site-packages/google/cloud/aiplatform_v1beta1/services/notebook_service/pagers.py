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

from google.cloud.aiplatform_v1beta1.types import notebook_runtime
from google.cloud.aiplatform_v1beta1.types import notebook_service


class ListNotebookRuntimeTemplatesPager:
    """A pager for iterating through ``list_notebook_runtime_templates`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``notebook_runtime_templates`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListNotebookRuntimeTemplates`` requests and continue to iterate
    through the ``notebook_runtime_templates`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., notebook_service.ListNotebookRuntimeTemplatesResponse],
        request: notebook_service.ListNotebookRuntimeTemplatesRequest,
        response: notebook_service.ListNotebookRuntimeTemplatesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = notebook_service.ListNotebookRuntimeTemplatesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[notebook_service.ListNotebookRuntimeTemplatesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[notebook_runtime.NotebookRuntimeTemplate]:
        for page in self.pages:
            yield from page.notebook_runtime_templates

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNotebookRuntimeTemplatesAsyncPager:
    """A pager for iterating through ``list_notebook_runtime_templates`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``notebook_runtime_templates`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListNotebookRuntimeTemplates`` requests and continue to iterate
    through the ``notebook_runtime_templates`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[notebook_service.ListNotebookRuntimeTemplatesResponse]
        ],
        request: notebook_service.ListNotebookRuntimeTemplatesRequest,
        response: notebook_service.ListNotebookRuntimeTemplatesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimeTemplatesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = notebook_service.ListNotebookRuntimeTemplatesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[notebook_service.ListNotebookRuntimeTemplatesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[notebook_runtime.NotebookRuntimeTemplate]:
        async def async_generator():
            async for page in self.pages:
                for response in page.notebook_runtime_templates:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNotebookRuntimesPager:
    """A pager for iterating through ``list_notebook_runtimes`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``notebook_runtimes`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListNotebookRuntimes`` requests and continue to iterate
    through the ``notebook_runtimes`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., notebook_service.ListNotebookRuntimesResponse],
        request: notebook_service.ListNotebookRuntimesRequest,
        response: notebook_service.ListNotebookRuntimesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = notebook_service.ListNotebookRuntimesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[notebook_service.ListNotebookRuntimesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[notebook_runtime.NotebookRuntime]:
        for page in self.pages:
            yield from page.notebook_runtimes

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNotebookRuntimesAsyncPager:
    """A pager for iterating through ``list_notebook_runtimes`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``notebook_runtimes`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListNotebookRuntimes`` requests and continue to iterate
    through the ``notebook_runtimes`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[notebook_service.ListNotebookRuntimesResponse]],
        request: notebook_service.ListNotebookRuntimesRequest,
        response: notebook_service.ListNotebookRuntimesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListNotebookRuntimesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = notebook_service.ListNotebookRuntimesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[notebook_service.ListNotebookRuntimesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[notebook_runtime.NotebookRuntime]:
        async def async_generator():
            async for page in self.pages:
                for response in page.notebook_runtimes:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
