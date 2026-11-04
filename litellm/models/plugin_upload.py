from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class PluginUpload:
    name: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    content: bytes
    created_by: str | None


@dataclass(frozen=True, slots=True)
class PluginArchiveMetadata:
    name: str
    enabled: bool
    source: Mapping[str, str]
    manifest_json: str | None


class PluginUploadStore(Protocol):
    async def create(self, upload: PluginUpload) -> str | None: ...

    async def get_metadata(self, name: str) -> PluginArchiveMetadata | None: ...

    async def get_content(self, metadata: PluginArchiveMetadata) -> bytes | None: ...
