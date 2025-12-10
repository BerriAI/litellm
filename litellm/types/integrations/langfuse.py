from typing import Any, Callable, Dict, Optional

from typing_extensions import TypedDict


class LangfuseLoggingConfig(TypedDict):
    langfuse_secret: Optional[str]
    langfuse_public_key: Optional[str]
    langfuse_host: Optional[str]
    langfuse_masking_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]


class LangfuseUsageDetails(TypedDict):
    input: Optional[int]
    output: Optional[int]
    total: Optional[int]
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]
