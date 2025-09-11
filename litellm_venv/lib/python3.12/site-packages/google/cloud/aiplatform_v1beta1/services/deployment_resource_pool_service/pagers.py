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

from google.cloud.aiplatform_v1beta1.types import deployment_resource_pool
from google.cloud.aiplatform_v1beta1.types import deployment_resource_pool_service
from google.cloud.aiplatform_v1beta1.types import endpoint


class ListDeploymentResourcePoolsPager:
    """A pager for iterating through ``list_deployment_resource_pools`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``deployment_resource_pools`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDeploymentResourcePools`` requests and continue to iterate
    through the ``deployment_resource_pools`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., deployment_resource_pool_service.ListDeploymentResourcePoolsResponse
        ],
        request: deployment_resource_pool_service.ListDeploymentResourcePoolsRequest,
        response: deployment_resource_pool_service.ListDeploymentResourcePoolsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = (
            deployment_resource_pool_service.ListDeploymentResourcePoolsRequest(request)
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[deployment_resource_pool_service.ListDeploymentResourcePoolsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[deployment_resource_pool.DeploymentResourcePool]:
        for page in self.pages:
            yield from page.deployment_resource_pools

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDeploymentResourcePoolsAsyncPager:
    """A pager for iterating through ``list_deployment_resource_pools`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``deployment_resource_pools`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDeploymentResourcePools`` requests and continue to iterate
    through the ``deployment_resource_pools`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ...,
            Awaitable[
                deployment_resource_pool_service.ListDeploymentResourcePoolsResponse
            ],
        ],
        request: deployment_resource_pool_service.ListDeploymentResourcePoolsRequest,
        response: deployment_resource_pool_service.ListDeploymentResourcePoolsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.ListDeploymentResourcePoolsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = (
            deployment_resource_pool_service.ListDeploymentResourcePoolsRequest(request)
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[
        deployment_resource_pool_service.ListDeploymentResourcePoolsResponse
    ]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(
        self,
    ) -> AsyncIterator[deployment_resource_pool.DeploymentResourcePool]:
        async def async_generator():
            async for page in self.pages:
                for response in page.deployment_resource_pools:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class QueryDeployedModelsPager:
    """A pager for iterating through ``query_deployed_models`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``deployed_models`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``QueryDeployedModels`` requests and continue to iterate
    through the ``deployed_models`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., deployment_resource_pool_service.QueryDeployedModelsResponse
        ],
        request: deployment_resource_pool_service.QueryDeployedModelsRequest,
        response: deployment_resource_pool_service.QueryDeployedModelsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = deployment_resource_pool_service.QueryDeployedModelsRequest(
            request
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[deployment_resource_pool_service.QueryDeployedModelsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[endpoint.DeployedModel]:
        for page in self.pages:
            yield from page.deployed_models

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class QueryDeployedModelsAsyncPager:
    """A pager for iterating through ``query_deployed_models`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``deployed_models`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``QueryDeployedModels`` requests and continue to iterate
    through the ``deployed_models`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[deployment_resource_pool_service.QueryDeployedModelsResponse]
        ],
        request: deployment_resource_pool_service.QueryDeployedModelsRequest,
        response: deployment_resource_pool_service.QueryDeployedModelsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1beta1.types.QueryDeployedModelsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = deployment_resource_pool_service.QueryDeployedModelsRequest(
            request
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[deployment_resource_pool_service.QueryDeployedModelsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[endpoint.DeployedModel]:
        async def async_generator():
            async for page in self.pages:
                for response in page.deployed_models:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
