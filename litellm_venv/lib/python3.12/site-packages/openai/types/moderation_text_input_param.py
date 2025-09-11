# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ModerationTextInputParam"]


class ModerationTextInputParam(TypedDict, total=False):
    text: Required[str]
    """A string of text to classify."""

    type: Required[Literal["text"]]
    """Always `text`."""
