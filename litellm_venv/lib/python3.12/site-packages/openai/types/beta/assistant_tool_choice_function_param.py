# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, TypedDict

__all__ = ["AssistantToolChoiceFunctionParam"]


class AssistantToolChoiceFunctionParam(TypedDict, total=False):
    name: Required[str]
    """The name of the function to call."""
