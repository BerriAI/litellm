# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal, TypeAlias

__all__ = ["RunStepInclude"]

RunStepInclude: TypeAlias = Literal[
    "step_details.tool_calls[*].file_search.results[*].content"
]
