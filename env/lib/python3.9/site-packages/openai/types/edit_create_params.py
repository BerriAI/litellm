# File generated from our OpenAPI spec by Stainless.

from __future__ import annotations

from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict

__all__ = ["EditCreateParams"]


class EditCreateParams(TypedDict, total=False):
    instruction: Required[str]
    """The instruction that tells the model how to edit the prompt."""

    model: Required[Union[str, Literal["text-davinci-edit-001", "code-davinci-edit-001"]]]
    """ID of the model to use.

    You can use the `text-davinci-edit-001` or `code-davinci-edit-001` model with
    this endpoint.
    """

    input: Optional[str]
    """The input text to use as a starting point for the edit."""

    n: Optional[int]
    """How many edits to generate for the input and instruction."""

    temperature: Optional[float]
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    top_p: Optional[float]
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """
