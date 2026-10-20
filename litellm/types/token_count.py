from typing import Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

TokenCountSource: TypeAlias = Literal["provider", "token_count_model", "local_estimate"]
TokenCountTokenizer: TypeAlias = Literal["kimi_k3"]
TokenCountEndpoint: TypeAlias = Literal["bedrock", "bedrock_mantle"]
TokenCountProvider: TypeAlias = TokenCountEndpoint | Literal["local"]


class TokenCountModelInfo(TypedDict):
    model: ReadOnly[NotRequired[str]]
    region: ReadOnly[NotRequired[str]]
    litellm_provider: ReadOnly[NotRequired[TokenCountProvider]]


class TokenCountSourceFields(TypedDict, total=False):
    litellm_count_source: ReadOnly[TokenCountSource]
    litellm_count_model: ReadOnly[str]
