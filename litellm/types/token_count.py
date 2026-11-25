from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

TokenCountSource: TypeAlias = Literal["provider", "token_count_model", "local_estimate"]
TokenCountTokenizer: TypeAlias = Literal["kimi_k3"]
TokenCountProvider: TypeAlias = Literal["bedrock", "bedrock_mantle"]


class TokenCountModelInfo(TypedDict):
    model: ReadOnly[str]
    region: ReadOnly[NotRequired[str]]
    litellm_provider: ReadOnly[NotRequired[TokenCountProvider]]


class TokenCountSourceFields(TypedDict, total=False):
    litellm_count_source: ReadOnly[TokenCountSource]
    litellm_count_model: ReadOnly[str]
