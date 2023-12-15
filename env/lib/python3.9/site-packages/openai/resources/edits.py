# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

import typing_extensions
from typing import TYPE_CHECKING, Union, Optional
from typing_extensions import Literal

import httpx

from ..types import Edit, edit_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import maybe_transform
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_raw_response_wrapper, async_to_raw_response_wrapper
from .._base_client import make_request_options

if TYPE_CHECKING:
    from .._client import OpenAI, AsyncOpenAI

__all__ = ["Edits", "AsyncEdits"]


class Edits(SyncAPIResource):
    with_raw_response: EditsWithRawResponse

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self.with_raw_response = EditsWithRawResponse(self)

    @typing_extensions.deprecated(
        "The Edits API is deprecated; please use Chat Completions instead.\n\nhttps://openai.com/blog/gpt-4-api-general-availability#deprecation-of-the-edits-api\n"
    )
    def create(
        self,
        *,
        instruction: str,
        model: Union[str, Literal["text-davinci-edit-001", "code-davinci-edit-001"]],
        input: Optional[str] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Edit:
        """
        Creates a new edit for the provided input, instruction, and parameters.

        Args:
          instruction: The instruction that tells the model how to edit the prompt.

          model: ID of the model to use. You can use the `text-davinci-edit-001` or
              `code-davinci-edit-001` model with this endpoint.

          input: The input text to use as a starting point for the edit.

          n: How many edits to generate for the input and instruction.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return self._post(
            "/edits",
            body=maybe_transform(
                {
                    "instruction": instruction,
                    "model": model,
                    "input": input,
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                edit_create_params.EditCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Edit,
        )


class AsyncEdits(AsyncAPIResource):
    with_raw_response: AsyncEditsWithRawResponse

    def __init__(self, client: AsyncOpenAI) -> None:
        super().__init__(client)
        self.with_raw_response = AsyncEditsWithRawResponse(self)

    @typing_extensions.deprecated(
        "The Edits API is deprecated; please use Chat Completions instead.\n\nhttps://openai.com/blog/gpt-4-api-general-availability#deprecation-of-the-edits-api\n"
    )
    async def create(
        self,
        *,
        instruction: str,
        model: Union[str, Literal["text-davinci-edit-001", "code-davinci-edit-001"]],
        input: Optional[str] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Edit:
        """
        Creates a new edit for the provided input, instruction, and parameters.

        Args:
          instruction: The instruction that tells the model how to edit the prompt.

          model: ID of the model to use. You can use the `text-davinci-edit-001` or
              `code-davinci-edit-001` model with this endpoint.

          input: The input text to use as a starting point for the edit.

          n: How many edits to generate for the input and instruction.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic.

              We generally recommend altering this or `top_p` but not both.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """
        return await self._post(
            "/edits",
            body=maybe_transform(
                {
                    "instruction": instruction,
                    "model": model,
                    "input": input,
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                edit_create_params.EditCreateParams,
            ),
            options=make_request_options(
                extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
            ),
            cast_to=Edit,
        )


class EditsWithRawResponse:
    def __init__(self, edits: Edits) -> None:
        self.create = to_raw_response_wrapper(  # pyright: ignore[reportDeprecated]
            edits.create  # pyright: ignore[reportDeprecated],
        )


class AsyncEditsWithRawResponse:
    def __init__(self, edits: AsyncEdits) -> None:
        self.create = async_to_raw_response_wrapper(  # pyright: ignore[reportDeprecated]
            edits.create  # pyright: ignore[reportDeprecated],
        )
