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

from google.cloud.aiplatform_v1.types import pipeline_job
from google.cloud.aiplatform_v1.types import pipeline_service
from google.cloud.aiplatform_v1.types import training_pipeline


class ListTrainingPipelinesPager:
    """A pager for iterating through ``list_training_pipelines`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``training_pipelines`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTrainingPipelines`` requests and continue to iterate
    through the ``training_pipelines`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., pipeline_service.ListTrainingPipelinesResponse],
        request: pipeline_service.ListTrainingPipelinesRequest,
        response: pipeline_service.ListTrainingPipelinesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListTrainingPipelinesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = pipeline_service.ListTrainingPipelinesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[pipeline_service.ListTrainingPipelinesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[training_pipeline.TrainingPipeline]:
        for page in self.pages:
            yield from page.training_pipelines

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListTrainingPipelinesAsyncPager:
    """A pager for iterating through ``list_training_pipelines`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``training_pipelines`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListTrainingPipelines`` requests and continue to iterate
    through the ``training_pipelines`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[pipeline_service.ListTrainingPipelinesResponse]
        ],
        request: pipeline_service.ListTrainingPipelinesRequest,
        response: pipeline_service.ListTrainingPipelinesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListTrainingPipelinesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListTrainingPipelinesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = pipeline_service.ListTrainingPipelinesRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[pipeline_service.ListTrainingPipelinesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[training_pipeline.TrainingPipeline]:
        async def async_generator():
            async for page in self.pages:
                for response in page.training_pipelines:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListPipelineJobsPager:
    """A pager for iterating through ``list_pipeline_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListPipelineJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``pipeline_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListPipelineJobs`` requests and continue to iterate
    through the ``pipeline_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListPipelineJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., pipeline_service.ListPipelineJobsResponse],
        request: pipeline_service.ListPipelineJobsRequest,
        response: pipeline_service.ListPipelineJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListPipelineJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListPipelineJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = pipeline_service.ListPipelineJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[pipeline_service.ListPipelineJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[pipeline_job.PipelineJob]:
        for page in self.pages:
            yield from page.pipeline_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListPipelineJobsAsyncPager:
    """A pager for iterating through ``list_pipeline_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListPipelineJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``pipeline_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListPipelineJobs`` requests and continue to iterate
    through the ``pipeline_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListPipelineJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[pipeline_service.ListPipelineJobsResponse]],
        request: pipeline_service.ListPipelineJobsRequest,
        response: pipeline_service.ListPipelineJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListPipelineJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListPipelineJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = pipeline_service.ListPipelineJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[pipeline_service.ListPipelineJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[pipeline_job.PipelineJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.pipeline_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
