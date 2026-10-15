from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Literal, NamedTuple, TypedDict

from typing_extensions import NotRequired, ReadOnly

FileContentProvider = Literal[
    "openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "litellm_proxy", "anthropic", "manus", "mistral"
]
FileRetrieveProvider = Literal[
    "openai", "azure", "gemini", "vertex_ai", "hosted_vllm", "litellm_proxy", "manus", "anthropic", "mistral", "xai"
]


class FileContentCallOptions(TypedDict, total=False):
    custom_llm_provider: ReadOnly[FileContentProvider]
    extra_body: ReadOnly[NotRequired[dict[str, str] | None]]
    extra_headers: ReadOnly[NotRequired[dict[str, str] | None]]
    chunk_size: ReadOnly[NotRequired[int]]
    stream: ReadOnly[NotRequired[bool]]


class FileContentRequestKwargs(TypedDict):
    file_id: ReadOnly[str]
    custom_llm_provider: ReadOnly[FileContentProvider]
    extra_body: ReadOnly[NotRequired[dict[str, str] | None]]
    extra_headers: ReadOnly[NotRequired[dict[str, str] | None]]
    chunk_size: ReadOnly[NotRequired[int]]
    stream: ReadOnly[NotRequired[bool]]


class FileRetrieveCallOptions(TypedDict, total=False):
    custom_llm_provider: ReadOnly[FileRetrieveProvider]
    extra_body: ReadOnly[NotRequired[dict[str, str] | None]]
    extra_headers: ReadOnly[NotRequired[dict[str, str] | None]]


class FileContentStreamingResult(NamedTuple):
    stream_iterator: Iterator[bytes] | AsyncIterator[bytes]
    headers: Mapping[str, str]
