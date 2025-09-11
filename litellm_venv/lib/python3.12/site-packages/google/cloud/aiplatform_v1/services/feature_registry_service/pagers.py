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

from google.cloud.aiplatform_v1.types import feature
from google.cloud.aiplatform_v1.types import feature_group
from google.cloud.aiplatform_v1.types import feature_registry_service
from google.cloud.aiplatform_v1.types import featurestore_service


class ListFeatureGroupsPager:
    """A pager for iterating through ``list_feature_groups`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``feature_groups`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListFeatureGroups`` requests and continue to iterate
    through the ``feature_groups`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., feature_registry_service.ListFeatureGroupsResponse],
        request: feature_registry_service.ListFeatureGroupsRequest,
        response: feature_registry_service.ListFeatureGroupsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListFeatureGroupsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = feature_registry_service.ListFeatureGroupsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[feature_registry_service.ListFeatureGroupsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[feature_group.FeatureGroup]:
        for page in self.pages:
            yield from page.feature_groups

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListFeatureGroupsAsyncPager:
    """A pager for iterating through ``list_feature_groups`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``feature_groups`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListFeatureGroups`` requests and continue to iterate
    through the ``feature_groups`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[feature_registry_service.ListFeatureGroupsResponse]
        ],
        request: feature_registry_service.ListFeatureGroupsRequest,
        response: feature_registry_service.ListFeatureGroupsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListFeatureGroupsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListFeatureGroupsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = feature_registry_service.ListFeatureGroupsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[feature_registry_service.ListFeatureGroupsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[feature_group.FeatureGroup]:
        async def async_generator():
            async for page in self.pages:
                for response in page.feature_groups:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListFeaturesPager:
    """A pager for iterating through ``list_features`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListFeaturesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``features`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListFeatures`` requests and continue to iterate
    through the ``features`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListFeaturesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., featurestore_service.ListFeaturesResponse],
        request: featurestore_service.ListFeaturesRequest,
        response: featurestore_service.ListFeaturesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListFeaturesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListFeaturesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = featurestore_service.ListFeaturesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[featurestore_service.ListFeaturesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[feature.Feature]:
        for page in self.pages:
            yield from page.features

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListFeaturesAsyncPager:
    """A pager for iterating through ``list_features`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListFeaturesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``features`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListFeatures`` requests and continue to iterate
    through the ``features`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListFeaturesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[featurestore_service.ListFeaturesResponse]],
        request: featurestore_service.ListFeaturesRequest,
        response: featurestore_service.ListFeaturesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListFeaturesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListFeaturesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = featurestore_service.ListFeaturesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[featurestore_service.ListFeaturesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[feature.Feature]:
        async def async_generator():
            async for page in self.pages:
                for response in page.features:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
