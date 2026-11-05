from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Protocol


def has_agent_skill_metadata(manifest: Mapping[str, object]) -> bool:
    return isinstance(manifest.get("agent_skill"), dict)


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
    id: str | None = None
    version: str | None = None
    description: str | None = None
    has_agent_skill_metadata: bool = False


class PluginUploadStore(Protocol):
    async def create(self, upload: PluginUpload) -> str | None: ...

    async def get_metadata(self, name: str) -> PluginArchiveMetadata | None: ...

    async def get_content(self, metadata: PluginArchiveMetadata) -> bytes | None: ...

    def public_metadata(self) -> AsyncIterator[PluginArchiveMetadata]: ...
