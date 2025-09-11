# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal, TypeAlias

__all__ = ["ChatCompletionRole"]

ChatCompletionRole: TypeAlias = Literal[
    "developer", "system", "user", "assistant", "tool", "function"
]
