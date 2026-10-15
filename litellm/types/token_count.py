from typing import Literal, TypeAlias

from typing_extensions import ReadOnly, TypedDict

TokenCountSource: TypeAlias = Literal["provider", "fallback_model", "local_estimate"]


class TokenCountSourceFields(TypedDict, total=False):
    litellm_count_source: ReadOnly[TokenCountSource]
    litellm_count_fallback_model: ReadOnly[str]
