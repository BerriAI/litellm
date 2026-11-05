from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol


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
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PluginPublication:
    name: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    created_by: str | None
    content: bytes | None = None


@dataclass(frozen=True, slots=True)
class PluginPublicationBase:
    id: str
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class PublishedPlugin:
    id: str
    name: str
    version: str
    enabled: bool
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PluginPublicationError:
    kind: Literal["invalid_version", "conflict", "not_found"]
    message: str


@dataclass(frozen=True, slots=True)
class PluginVersion:
    id: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    published_at: datetime | None
    published_by: str | None
    archive_available: bool


class PluginUploadStore(Protocol):
    async def create(self, upload: PluginUpload) -> str | None: ...

    async def publish(
        self, publication: PluginPublication, expected: PluginPublicationBase | None = None
    ) -> PublishedPlugin | PluginPublicationError: ...

    async def versions(self, metadata: PluginArchiveMetadata, limit: int, offset: int) -> tuple[PluginVersion, ...]: ...

    async def version(self, metadata: PluginArchiveMetadata, version_id: str) -> PluginVersion | None: ...

    async def version_content(self, metadata: PluginArchiveMetadata, version_id: str) -> bytes | None: ...

    async def get_metadata(self, name: str) -> PluginArchiveMetadata | None: ...

    async def get_content(self, metadata: PluginArchiveMetadata) -> bytes | None: ...

    def public_metadata(self) -> AsyncIterator[PluginArchiveMetadata]: ...
