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

from google.cloud.aiplatform_v1.types import batch_prediction_job
from google.cloud.aiplatform_v1.types import custom_job
from google.cloud.aiplatform_v1.types import data_labeling_job
from google.cloud.aiplatform_v1.types import hyperparameter_tuning_job
from google.cloud.aiplatform_v1.types import job_service
from google.cloud.aiplatform_v1.types import model_deployment_monitoring_job
from google.cloud.aiplatform_v1.types import (
    model_deployment_monitoring_job as gca_model_deployment_monitoring_job,
)
from google.cloud.aiplatform_v1.types import nas_job


class ListCustomJobsPager:
    """A pager for iterating through ``list_custom_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListCustomJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``custom_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListCustomJobs`` requests and continue to iterate
    through the ``custom_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListCustomJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListCustomJobsResponse],
        request: job_service.ListCustomJobsRequest,
        response: job_service.ListCustomJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListCustomJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListCustomJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListCustomJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListCustomJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[custom_job.CustomJob]:
        for page in self.pages:
            yield from page.custom_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListCustomJobsAsyncPager:
    """A pager for iterating through ``list_custom_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListCustomJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``custom_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListCustomJobs`` requests and continue to iterate
    through the ``custom_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListCustomJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[job_service.ListCustomJobsResponse]],
        request: job_service.ListCustomJobsRequest,
        response: job_service.ListCustomJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListCustomJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListCustomJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListCustomJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[job_service.ListCustomJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[custom_job.CustomJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.custom_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDataLabelingJobsPager:
    """A pager for iterating through ``list_data_labeling_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``data_labeling_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListDataLabelingJobs`` requests and continue to iterate
    through the ``data_labeling_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListDataLabelingJobsResponse],
        request: job_service.ListDataLabelingJobsRequest,
        response: job_service.ListDataLabelingJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListDataLabelingJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListDataLabelingJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListDataLabelingJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[data_labeling_job.DataLabelingJob]:
        for page in self.pages:
            yield from page.data_labeling_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListDataLabelingJobsAsyncPager:
    """A pager for iterating through ``list_data_labeling_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``data_labeling_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListDataLabelingJobs`` requests and continue to iterate
    through the ``data_labeling_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[job_service.ListDataLabelingJobsResponse]],
        request: job_service.ListDataLabelingJobsRequest,
        response: job_service.ListDataLabelingJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListDataLabelingJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListDataLabelingJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListDataLabelingJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[job_service.ListDataLabelingJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[data_labeling_job.DataLabelingJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.data_labeling_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListHyperparameterTuningJobsPager:
    """A pager for iterating through ``list_hyperparameter_tuning_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``hyperparameter_tuning_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListHyperparameterTuningJobs`` requests and continue to iterate
    through the ``hyperparameter_tuning_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListHyperparameterTuningJobsResponse],
        request: job_service.ListHyperparameterTuningJobsRequest,
        response: job_service.ListHyperparameterTuningJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListHyperparameterTuningJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListHyperparameterTuningJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[hyperparameter_tuning_job.HyperparameterTuningJob]:
        for page in self.pages:
            yield from page.hyperparameter_tuning_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListHyperparameterTuningJobsAsyncPager:
    """A pager for iterating through ``list_hyperparameter_tuning_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``hyperparameter_tuning_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListHyperparameterTuningJobs`` requests and continue to iterate
    through the ``hyperparameter_tuning_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[job_service.ListHyperparameterTuningJobsResponse]
        ],
        request: job_service.ListHyperparameterTuningJobsRequest,
        response: job_service.ListHyperparameterTuningJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListHyperparameterTuningJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListHyperparameterTuningJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[job_service.ListHyperparameterTuningJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(
        self,
    ) -> AsyncIterator[hyperparameter_tuning_job.HyperparameterTuningJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.hyperparameter_tuning_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNasJobsPager:
    """A pager for iterating through ``list_nas_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListNasJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``nas_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListNasJobs`` requests and continue to iterate
    through the ``nas_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListNasJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListNasJobsResponse],
        request: job_service.ListNasJobsRequest,
        response: job_service.ListNasJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListNasJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListNasJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListNasJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListNasJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[nas_job.NasJob]:
        for page in self.pages:
            yield from page.nas_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNasJobsAsyncPager:
    """A pager for iterating through ``list_nas_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListNasJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``nas_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListNasJobs`` requests and continue to iterate
    through the ``nas_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListNasJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[job_service.ListNasJobsResponse]],
        request: job_service.ListNasJobsRequest,
        response: job_service.ListNasJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListNasJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListNasJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListNasJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[job_service.ListNasJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[nas_job.NasJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.nas_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNasTrialDetailsPager:
    """A pager for iterating through ``list_nas_trial_details`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``nas_trial_details`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListNasTrialDetails`` requests and continue to iterate
    through the ``nas_trial_details`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListNasTrialDetailsResponse],
        request: job_service.ListNasTrialDetailsRequest,
        response: job_service.ListNasTrialDetailsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListNasTrialDetailsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListNasTrialDetailsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListNasTrialDetailsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[nas_job.NasTrialDetail]:
        for page in self.pages:
            yield from page.nas_trial_details

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListNasTrialDetailsAsyncPager:
    """A pager for iterating through ``list_nas_trial_details`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``nas_trial_details`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListNasTrialDetails`` requests and continue to iterate
    through the ``nas_trial_details`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[job_service.ListNasTrialDetailsResponse]],
        request: job_service.ListNasTrialDetailsRequest,
        response: job_service.ListNasTrialDetailsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListNasTrialDetailsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListNasTrialDetailsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListNasTrialDetailsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[job_service.ListNasTrialDetailsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[nas_job.NasTrialDetail]:
        async def async_generator():
            async for page in self.pages:
                for response in page.nas_trial_details:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBatchPredictionJobsPager:
    """A pager for iterating through ``list_batch_prediction_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``batch_prediction_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListBatchPredictionJobs`` requests and continue to iterate
    through the ``batch_prediction_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListBatchPredictionJobsResponse],
        request: job_service.ListBatchPredictionJobsRequest,
        response: job_service.ListBatchPredictionJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListBatchPredictionJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListBatchPredictionJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListBatchPredictionJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[batch_prediction_job.BatchPredictionJob]:
        for page in self.pages:
            yield from page.batch_prediction_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListBatchPredictionJobsAsyncPager:
    """A pager for iterating through ``list_batch_prediction_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``batch_prediction_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListBatchPredictionJobs`` requests and continue to iterate
    through the ``batch_prediction_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., Awaitable[job_service.ListBatchPredictionJobsResponse]],
        request: job_service.ListBatchPredictionJobsRequest,
        response: job_service.ListBatchPredictionJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListBatchPredictionJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListBatchPredictionJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListBatchPredictionJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[job_service.ListBatchPredictionJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[batch_prediction_job.BatchPredictionJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.batch_prediction_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelDeploymentMonitoringStatsAnomaliesPager:
    """A pager for iterating through ``search_model_deployment_monitoring_stats_anomalies`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``monitoring_stats`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``SearchModelDeploymentMonitoringStatsAnomalies`` requests and continue to iterate
    through the ``monitoring_stats`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse
        ],
        request: job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest,
        response: job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = (
            job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest(request)
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(
        self,
    ) -> Iterator[job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(
        self,
    ) -> Iterator[gca_model_deployment_monitoring_job.ModelMonitoringStatsAnomalies]:
        for page in self.pages:
            yield from page.monitoring_stats

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class SearchModelDeploymentMonitoringStatsAnomaliesAsyncPager:
    """A pager for iterating through ``search_model_deployment_monitoring_stats_anomalies`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``monitoring_stats`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``SearchModelDeploymentMonitoringStatsAnomalies`` requests and continue to iterate
    through the ``monitoring_stats`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ...,
            Awaitable[
                job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse
            ],
        ],
        request: job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest,
        response: job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.SearchModelDeploymentMonitoringStatsAnomaliesResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = (
            job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest(request)
        )
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[
        job_service.SearchModelDeploymentMonitoringStatsAnomaliesResponse
    ]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(
        self,
    ) -> AsyncIterator[
        gca_model_deployment_monitoring_job.ModelMonitoringStatsAnomalies
    ]:
        async def async_generator():
            async for page in self.pages:
                for response in page.monitoring_stats:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListModelDeploymentMonitoringJobsPager:
    """A pager for iterating through ``list_model_deployment_monitoring_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``model_deployment_monitoring_jobs`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListModelDeploymentMonitoringJobs`` requests and continue to iterate
    through the ``model_deployment_monitoring_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[..., job_service.ListModelDeploymentMonitoringJobsResponse],
        request: job_service.ListModelDeploymentMonitoringJobsRequest,
        response: job_service.ListModelDeploymentMonitoringJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListModelDeploymentMonitoringJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[job_service.ListModelDeploymentMonitoringJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(
        self,
    ) -> Iterator[model_deployment_monitoring_job.ModelDeploymentMonitoringJob]:
        for page in self.pages:
            yield from page.model_deployment_monitoring_jobs

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)


class ListModelDeploymentMonitoringJobsAsyncPager:
    """A pager for iterating through ``list_model_deployment_monitoring_jobs`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``model_deployment_monitoring_jobs`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListModelDeploymentMonitoringJobs`` requests and continue to iterate
    through the ``model_deployment_monitoring_jobs`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(
        self,
        method: Callable[
            ..., Awaitable[job_service.ListModelDeploymentMonitoringJobsResponse]
        ],
        request: job_service.ListModelDeploymentMonitoringJobsRequest,
        response: job_service.ListModelDeploymentMonitoringJobsResponse,
        *,
        metadata: Sequence[Tuple[str, str]] = ()
    ):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsRequest):
                The initial request object.
            response (google.cloud.aiplatform_v1.types.ListModelDeploymentMonitoringJobsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = job_service.ListModelDeploymentMonitoringJobsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(
        self,
    ) -> AsyncIterator[job_service.ListModelDeploymentMonitoringJobsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(
        self,
    ) -> AsyncIterator[model_deployment_monitoring_job.ModelDeploymentMonitoringJob]:
        async def async_generator():
            async for page in self.pages:
                for response in page.model_deployment_monitoring_jobs:
                    yield response

        return async_generator()

    def __repr__(self) -> str:
        return "{0}<{1!r}>".format(self.__class__.__name__, self._response)
