from typing import AsyncIterator, Dict, Iterator, NamedTuple, Union


class FileContentStreamingResult(NamedTuple):
    stream_iterator: Union[Iterator[bytes], AsyncIterator[bytes]]
    headers: Dict[str, str]
