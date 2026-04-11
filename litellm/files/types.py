from typing import AsyncIterator, Dict, Iterator, Literal, NamedTuple, Union


FileContentProvider = Literal[
    "openai", "azure", "vertex_ai", "bedrock", "hosted_vllm", "anthropic", "manus"
]


class FileContentStreamingResult(NamedTuple):
    stream_iterator: Union[Iterator[bytes], AsyncIterator[bytes]]
    headers: Dict[str, str]
