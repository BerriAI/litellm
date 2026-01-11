from datetime import datetime
from typing import Protocol


class BlobWriter(Protocol):
    """
    Uniform interface for writing blobs asynchronously to a blob storage.

    Manages retries.
    """

    async def put_object(self, key: str, body: bytes) -> None: ...

    def make_key(self, timestamp: datetime) -> str: ...
