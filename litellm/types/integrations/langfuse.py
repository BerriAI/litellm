from typing import Optional

from typing_extensions import TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]


class LangfuseUsageDetails(TypedDict):
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
