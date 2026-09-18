from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Literal, NamedTuple

FileContentProvider = Literal[
    "openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "litellm_proxy", "anthropic", "manus"
]


class FileContentStreamingResult(NamedTuple):
    stream_iterator: Iterator[bytes] | AsyncIterator[bytes]
    headers: Mapping[str, str]
