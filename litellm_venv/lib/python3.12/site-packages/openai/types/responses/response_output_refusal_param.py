# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ResponseOutputRefusalParam"]


class ResponseOutputRefusalParam(TypedDict, total=False):
    refusal: Required[str]
    """The refusal explanation from the model."""

    type: Required[Literal["refusal"]]
    """The type of the refusal. Always `refusal`."""
