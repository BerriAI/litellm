from collections.abc import AsyncIterator, Iterator
from typing import Literal, NamedTuple

FileContentProvider = Literal["openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic", "manus"]


class FileContentStreamingResult(NamedTuple):
    stream_iterator: Iterator[bytes] | AsyncIterator[bytes]
    headers: dict[str, str]
