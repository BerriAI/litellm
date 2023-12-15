# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, Optional, overload
from typing_extensions import Literal

import httpx

from ..types import (
    FineTune,
    FineTuneEvent,
    FineTuneEventsListResponse,
    fine_tune_create_params,
    fine_tune_list_events_params,
)
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import maybe_transform
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .._streaming import Stream, AsyncStream
from ..pagination import SyncPage, AsyncPage
from .._base_client import AsyncPaginator, make_request_options

if TYPE_CHECKING:
    from .._client import OpenAI, AsyncOpenAI

__all__ = ["FineTunes", "AsyncFineTunes"]


class FineTunes(SyncAPIResource):
    with_raw_response: FineTunesWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.with_raw_response = FineTunesWithRawResponse(self)

    def create(
        self,
        *,
        training_file: str,
        batch_size: Optional[int] | NotGiven = NOT_GIVEN,
        classification_betas: Optional[List[float]] | NotGiven = NOT_GIVEN,
        classification_n_classes: Optional[int] | NotGiven = NOT_GIVEN,
        classification_positive_class: Optional[str] | NotGiven = NOT_GIVEN,
        compute_classification_metrics: Optional[bool] | NotGiven = NOT_GIVEN,
        hyperparameters: fine_tune_create_params.Hyperparameters | NotGiven = NOT_GIVEN,
        learning_rate_multiplier: Optional[float] | NotGiven = NOT_GIVEN,
        model: Union[str, Literal["ada", "babbage", "curie", "davinci"], None] | NotGiven = NOT_GIVEN,
        prompt_loss_weight: Optional[float] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        validation_file: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Creates a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name
        of the fine-tuned models once complete.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/legacy-fine-tuning)

        Args:
          training_file: The ID of an uploaded file that contains training data.

              See [upload file](https://platform.openai.com/docs/api-reference/files/upload)
              for how to upload a file.

              Your dataset must be formatted as a JSONL file, where each training example is a
              JSON object with the keys "prompt" and "completion". Additionally, you must
              upload your file with the purpose `fine-tune`.

              See the
              [fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning/creating-training-data)
              for more details.

          batch_size: The batch size to use for training. The batch size is the number of training
              examples used to train a single forward and backward pass.

              By default, the batch size will be dynamically configured to be ~0.2% of the
              number of examples in the training set, capped at 256 - in general, we've found
              that larger batch sizes tend to work better for larger datasets.

          classification_betas: If this is provided, we calculate F-beta scores at the specified beta values.
              The F-beta score is a generalization of F-1 score. This is only used for binary
              classification.

              With a beta of 1 (i.e. the F-1 score), precision and recall are given the same
              weight. A larger beta score puts more weight on recall and less on precision. A
              smaller beta score puts more weight on precision and less on recall.

          classification_n_classes: The number of classes in a classification task.

              This parameter is required for multiclass classification.

          classification_positive_class: The positive class in binary classification.

              This parameter is needed to generate precision, recall, and F1 metrics when
              doing binary classification.

          compute_classification_metrics: If set, we calculate classification-specific metrics such as accuracy and F-1
              score using the validation set at the end of every epoch. These metrics can be
              viewed in the
              [results file](https://platform.openai.com/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).

              In order to compute classification metrics, you must provide a
              `validation_file`. Additionally, you must specify `classification_n_classes` for
              multiclass classification or `classification_positive_class` for binary
              classification.

          hyperparameters: The hyperparameters used for the fine-tuning job.

          learning_rate_multiplier: The learning rate multiplier to use for training. The fine-tuning learning rate
              is the original learning rate used for pretraining multiplied by this value.

              By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on
              final `batch_size` (larger learning rates tend to perform better with larger
              batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to
              see what produces the best results.

          model: The name of the base model to fine-tune. You can select one of "ada", "babbage",
              "curie", "davinci", or a fine-tuned model created after 2022-04-21 and before
              2023-08-22. To learn more about these models, see the
              [Models](https://platform.openai.com/docs/models) documentation.

          prompt_loss_weight: The weight to use for loss on the prompt tokens. This controls how much the
              model tries to learn to generate the prompt (as compared to the completion which
              always has a weight of 1.0), and can add a stabilizing effect to training when
              completions are short.

              If prompts are extremely long (relative to completions), it may make sense to
              reduce this weight so as to avoid over-prioritizing learning the prompt.

          suffix: A string of up to 40 characters that will be added to your fine-tuned model
              name.

              For example, a `suffix` of "custom-model-name" would produce a model name like
              `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.

          validation_file: The ID of an uploaded file that contains validation data.

              If you provide this file, the data is used to generate validation metrics
              periodically during fine-tuning. These metrics can be viewed in the
              [fine-tuning results file](https://platform.openai.com/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).
              Your train and validation data should be mutually exclusive.

              Your dataset must be formatted as a JSONL file, where each validation example is
              a JSON object with the keys "prompt" and "completion". Additionally, you must
              upload your file with the purpose `fine-tune`.

              See the
              [fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning/creating-training-data)
              for more details.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/fine-tunes",
            body=maybe_transform(
                {
                    "training_file": training_file,
                    "batch_size": batch_size,
                    "classification_betas": classification_betas,
                    "classification_n_classes": classification_n_classes,
                    "classification_positive_class": classification_positive_class,
                    "compute_classification_metrics": compute_classification_metrics,
                    "hyperparameters": hyperparameters,
                    "learning_rate_multiplier": learning_rate_multiplier,
                    "model": model,
                    "prompt_loss_weight": prompt_loss_weight,
                    "suffix": suffix,
                    "validation_file": validation_file,
                },
                fine_tune_create_params.FineTuneCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    def retrieve(
        self,
        fine_tune_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Gets info about the fine-tune job.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/legacy-fine-tuning)

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._get(
            f"/fine-tunes/{fine_tune_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    def list(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> SyncPage[FineTune]:
        """List your organization's fine-tuning jobs"""
        return self._get_api_list(
            "/fine-tunes",
            page=SyncPage[FineTune],
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            model=FineTune,
        )

    def cancel(
        self,
        fine_tune_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Immediately cancel a fine-tune job.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            f"/fine-tunes/{fine_tune_id}/cancel",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    @overload
    def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[False] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[True],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> Stream[FineTuneEvent]:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: bool,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse | Stream[FineTuneEvent]:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[False] | Literal[True] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse | Stream[FineTuneEvent]:
        return self._get(
            f"/fine-tunes/{fine_tune_id}/events",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"stream": stream}, fine_tune_list_events_params.FineTuneListEventsParams),
            ),
            cast_to=FineTuneEventsListResponse,
            stream=stream or False,
            stream_cls=Stream[FineTuneEvent],
        )


class AsyncFineTunes(AsyncAPIResource):
    with_raw_response: AsyncFineTunesWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(client)
        self.with_raw_response = AsyncFineTunesWithRawResponse(self)

    async def create(
        self,
        *,
        training_file: str,
        batch_size: Optional[int] | NotGiven = NOT_GIVEN,
        classification_betas: Optional[List[float]] | NotGiven = NOT_GIVEN,
        classification_n_classes: Optional[int] | NotGiven = NOT_GIVEN,
        classification_positive_class: Optional[str] | NotGiven = NOT_GIVEN,
        compute_classification_metrics: Optional[bool] | NotGiven = NOT_GIVEN,
        hyperparameters: fine_tune_create_params.Hyperparameters | NotGiven = NOT_GIVEN,
        learning_rate_multiplier: Optional[float] | NotGiven = NOT_GIVEN,
        model: Union[str, Literal["ada", "babbage", "curie", "davinci"], None] | NotGiven = NOT_GIVEN,
        prompt_loss_weight: Optional[float] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        validation_file: Optional[str] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Creates a job that fine-tunes a specified model from a given dataset.

        Response includes details of the enqueued job including job status and the name
        of the fine-tuned models once complete.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/legacy-fine-tuning)

        Args:
          training_file: The ID of an uploaded file that contains training data.

              See [upload file](https://platform.openai.com/docs/api-reference/files/upload)
              for how to upload a file.

              Your dataset must be formatted as a JSONL file, where each training example is a
              JSON object with the keys "prompt" and "completion". Additionally, you must
              upload your file with the purpose `fine-tune`.

              See the
              [fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning/creating-training-data)
              for more details.

          batch_size: The batch size to use for training. The batch size is the number of training
              examples used to train a single forward and backward pass.

              By default, the batch size will be dynamically configured to be ~0.2% of the
              number of examples in the training set, capped at 256 - in general, we've found
              that larger batch sizes tend to work better for larger datasets.

          classification_betas: If this is provided, we calculate F-beta scores at the specified beta values.
              The F-beta score is a generalization of F-1 score. This is only used for binary
              classification.

              With a beta of 1 (i.e. the F-1 score), precision and recall are given the same
              weight. A larger beta score puts more weight on recall and less on precision. A
              smaller beta score puts more weight on precision and less on recall.

          classification_n_classes: The number of classes in a classification task.

              This parameter is required for multiclass classification.

          classification_positive_class: The positive class in binary classification.

              This parameter is needed to generate precision, recall, and F1 metrics when
              doing binary classification.

          compute_classification_metrics: If set, we calculate classification-specific metrics such as accuracy and F-1
              score using the validation set at the end of every epoch. These metrics can be
              viewed in the
              [results file](https://platform.openai.com/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).

              In order to compute classification metrics, you must provide a
              `validation_file`. Additionally, you must specify `classification_n_classes` for
              multiclass classification or `classification_positive_class` for binary
              classification.

          hyperparameters: The hyperparameters used for the fine-tuning job.

          learning_rate_multiplier: The learning rate multiplier to use for training. The fine-tuning learning rate
              is the original learning rate used for pretraining multiplied by this value.

              By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on
              final `batch_size` (larger learning rates tend to perform better with larger
              batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to
              see what produces the best results.

          model: The name of the base model to fine-tune. You can select one of "ada", "babbage",
              "curie", "davinci", or a fine-tuned model created after 2022-04-21 and before
              2023-08-22. To learn more about these models, see the
              [Models](https://platform.openai.com/docs/models) documentation.

          prompt_loss_weight: The weight to use for loss on the prompt tokens. This controls how much the
              model tries to learn to generate the prompt (as compared to the completion which
              always has a weight of 1.0), and can add a stabilizing effect to training when
              completions are short.

              If prompts are extremely long (relative to completions), it may make sense to
              reduce this weight so as to avoid over-prioritizing learning the prompt.

          suffix: A string of up to 40 characters that will be added to your fine-tuned model
              name.

              For example, a `suffix` of "custom-model-name" would produce a model name like
              `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.

          validation_file: The ID of an uploaded file that contains validation data.

              If you provide this file, the data is used to generate validation metrics
              periodically during fine-tuning. These metrics can be viewed in the
              [fine-tuning results file](https://platform.openai.com/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).
              Your train and validation data should be mutually exclusive.

              Your dataset must be formatted as a JSONL file, where each validation example is
              a JSON object with the keys "prompt" and "completion". Additionally, you must
              upload your file with the purpose `fine-tune`.

              See the
              [fine-tuning guide](https://platform.openai.com/docs/guides/legacy-fine-tuning/creating-training-data)
              for more details.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/fine-tunes",
            body=maybe_transform(
                {
                    "training_file": training_file,
                    "batch_size": batch_size,
                    "classification_betas": classification_betas,
                    "classification_n_classes": classification_n_classes,
                    "classification_positive_class": classification_positive_class,
                    "compute_classification_metrics": compute_classification_metrics,
                    "hyperparameters": hyperparameters,
                    "learning_rate_multiplier": learning_rate_multiplier,
                    "model": model,
                    "prompt_loss_weight": prompt_loss_weight,
                    "suffix": suffix,
                    "validation_file": validation_file,
                },
                fine_tune_create_params.FineTuneCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    async def retrieve(
        self,
        fine_tune_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Gets info about the fine-tune job.

        [Learn more about fine-tuning](https://platform.openai.com/docs/guides/legacy-fine-tuning)

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._get(
            f"/fine-tunes/{fine_tune_id}",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    def list(
        self,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> AsyncPaginator[FineTune, AsyncPage[FineTune]]:
        """List your organization's fine-tuning jobs"""
        return self._get_api_list(
            "/fine-tunes",
            page=AsyncPage[FineTune],
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            model=FineTune,
        )

    async def cancel(
        self,
        fine_tune_id: str,
        *,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> FineTune:
        """
        Immediately cancel a fine-tune job.

        Args:
          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            f"/fine-tunes/{fine_tune_id}/cancel",
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=FineTune,
        )

    @overload
    async def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[False] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    async def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[True],
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> AsyncStream[FineTuneEvent]:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    @overload
    async def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: bool,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse | AsyncStream[FineTuneEvent]:
        """
        Get fine-grained status updates for a fine-tune job.

        Args:
          stream: Whether to stream events for the fine-tune job. If set to true, events will be
              sent as data-only
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
              as they become available. The stream will terminate with a `data: [DONE]`
              message when the job is finished (succeeded, cancelled, or failed).

              If set to false, only events generated so far will be returned.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        ...

    async def list_events(
        self,
        fine_tune_id: str,
        *,
        stream: Literal[False] | Literal[True] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = 86400,
    ) -> FineTuneEventsListResponse | AsyncStream[FineTuneEvent]:
        return await self._get(
            f"/fine-tunes/{fine_tune_id}/events",
            options=make_request_options(
                extra_headers=extra_headers,
                extra_query=extra_query,
                extra_body=extra_body,
                timeout=timeout,
                query=maybe_transform({"stream": stream}, fine_tune_list_events_params.FineTuneListEventsParams),
            ),
            cast_to=FineTuneEventsListResponse,
            stream=stream or False,
            stream_cls=AsyncStream[FineTuneEvent],
        )


class FineTunesWithRawResponse:
    def __init__(self, fine_tunes: FineTunes) -> None:
        self.create = to_raw_response_wrapper(
            fine_tunes.create,
        )
        self.retrieve = to_raw_response_wrapper(
            fine_tunes.retrieve,
        )
        self.list = to_raw_response_wrapper(
            fine_tunes.list,
        )
        self.cancel = to_raw_response_wrapper(
            fine_tunes.cancel,
        )
        self.list_events = to_raw_response_wrapper(
            fine_tunes.list_events,
        )


class AsyncFineTunesWithRawResponse:
    def __init__(self, fine_tunes: AsyncFineTunes) -> None:
        self.create = async_to_raw_response_wrapper(
            fine_tunes.create,
        )
        self.retrieve = async_to_raw_response_wrapper(
            fine_tunes.retrieve,
        )
        self.list = async_to_raw_response_wrapper(
            fine_tunes.list,
        )
        self.cancel = async_to_raw_response_wrapper(
            fine_tunes.cancel,
        )
        self.list_events = async_to_raw_response_wrapper(
            fine_tunes.list_events,
        )
