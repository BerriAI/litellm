from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal, Protocol


def has_agent_skill_metadata(manifest: Mapping[str, object]) -> bool:
    return isinstance(manifest.get("agent_skill"), dict)


def uploaded_plugin_source(base_url: str, name: str, digest: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "source": "archive",
            "url": f"{base_url.rstrip('/')}/claude-code/plugins/{name}/archive/{digest}.zip",
            "sha256": digest,
        }
    )


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
    created_at: datetime | None = None
    created_by: str | None = None


@dataclass(frozen=True, slots=True)
class PluginReview:
    approved_by: str
    security_scan_passed: bool


@dataclass(frozen=True, slots=True)
class PluginPublication:
    name: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    created_by: str | None
    content: bytes | None = None
    review: PluginReview | None = None


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
    created_at: datetime | None = None
    created_by: str | None = None
    review: PluginReview | None = None
    published_by: str | None = None


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


@dataclass(frozen=True, slots=True)
class PluginSnapshot:
    id: str
    name: str
    version: str | None
    description: str | None
    manifest: Mapping[str, object]
    enabled: bool
    created_at: datetime | None
    updated_at: datetime | None
    created_by: str | None


@dataclass(frozen=True, slots=True)
class PluginDownloader:
    authenticated: bool
    user_id: str | None
    user_email: str | None
    key_alias: str | None
    team_id: str | None
    team_alias: str | None


class PluginDownloadResult(Enum):
    RECORDED = "recorded"
    SKILL_NOT_FOUND = "skill_not_found"


@dataclass(frozen=True, slots=True)
class PluginDownloadEvent:
    id: str
    downloaded_at: datetime
    version: str | None
    digest: str
    authenticated: bool
    user_id: str | None
    user_email: str | None
    key_alias: str | None
    team_id: str | None
    team_alias: str | None


@dataclass(frozen=True, slots=True)
class PluginDownloadsResponse:
    total_downloads: int
    anonymous_downloads: int
    downloads: tuple[PluginDownloadEvent, ...]
    next_cursor: str | None


class PluginUploadStore(Protocol):
    async def create(self, upload: PluginUpload) -> str | None: ...

    async def get_plugin(self, name: str) -> PluginSnapshot | None: ...

    async def publish(
        self, publication: PluginPublication, expected: PluginPublicationBase | None = None
    ) -> PublishedPlugin | PluginPublicationError: ...

    async def versions(self, metadata: PluginArchiveMetadata, limit: int, offset: int) -> tuple[PluginVersion, ...]: ...

    async def version(self, metadata: PluginArchiveMetadata, version_id: str) -> PluginVersion | None: ...

    async def version_content(self, metadata: PluginArchiveMetadata, version_id: str) -> bytes | None: ...

    async def get_metadata(self, name: str) -> PluginArchiveMetadata | None: ...

    async def get_content(self, metadata: PluginArchiveMetadata) -> bytes | None: ...

    def public_metadata(self) -> AsyncIterator[PluginArchiveMetadata]: ...

    async def record_download(
        self, metadata: PluginArchiveMetadata, digest: str, downloader: PluginDownloader
    ) -> PluginDownloadResult: ...

    async def downloads(self, name: str, limit: int, cursor: int | None) -> PluginDownloadsResponse | None: ...

    async def download_counts(self, plugin_ids: tuple[str, ...]) -> Mapping[str, int]: ...
