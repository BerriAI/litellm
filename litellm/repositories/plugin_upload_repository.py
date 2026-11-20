from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # Prisma dynamically exposes its typed table actions
)

from pydantic import TypeAdapter, ValidationError

from litellm.models.plugin_upload import (
    PluginArchiveMetadata,
    PluginDownloader,
    PluginDownloadEvent,
    PluginDownloadResult,
    PluginDownloadsResponse,
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginSnapshot,
    PluginUpload,
    PluginVersion,
    PublishedPlugin,
    has_agent_skill_metadata,
)
from litellm.repositories.plugin_publication import PluginPublicationTransaction, publish_plugin, verified_archive
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ClaudeCodePluginRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.db.create_views import SupportsExecuteRaw
    from litellm.proxy.utils import PrismaClient

_MANIFEST: Final = TypeAdapter(dict[str, object])
_SOURCE: Final = TypeAdapter(dict[str, str])
PAGE_SIZE: Final = 100
_INCREMENT_ANONYMOUS_DOWNLOADS: Final = """
UPDATE "LiteLLM_ClaudeCodePluginTable"
SET "anonymous_download_count" = "anonymous_download_count" + 1
WHERE "id" = $1
"""
_DOWNLOAD_COUNTS_QUERY: Final = """
SELECT p.id, p.anonymous_download_count + COUNT(d.id) AS download_count
FROM "LiteLLM_ClaudeCodePluginTable" p
LEFT JOIN "LiteLLM_ClaudeCodePluginDownload" d ON d.plugin_id = p.id
WHERE p.id = ANY($1::text[])
GROUP BY p.id
"""


@dataclass(frozen=True, slots=True)
class _DownloadCount:
    id: str
    download_count: int


_DOWNLOAD_COUNTS: Final = TypeAdapter(tuple[_DownloadCount, ...])
_VERSIONS: Final = TypeAdapter(tuple[PluginVersion, ...])
_VERSION_QUERY: Final = """
    SELECT v.id, v.version, v.description, COALESCE(v.manifest_json, '{}')::jsonb AS manifest,
           v.published_at, v.published_by, v.archive_available
    FROM "LiteLLM_ClaudeCodePluginVersion" v JOIN "LiteLLM_ClaudeCodePluginTable" p ON p.id = v.plugin_id
    WHERE p.id = $1 AND p.enabled = $2
"""


class _PublicationDatabase(PluginPublicationTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[PluginPublicationTransaction]: ...


class PluginUploadRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._db = cast(  # cast-ok: the generated Prisma transaction delegate is untyped at this boundary
            _PublicationDatabase, prisma_client.writer_db
        )
        self._writer = cast(  # cast-ok: PrismaWrapper dynamically delegates raw statements to its generated client
            "SupportsExecuteRaw", prisma_client.writer_db
        )
        self._plugins = ClaudeCodePluginRepository(prisma_client, use_writer=True).table
        self._archives = cast(  # cast-ok: Prisma exposes generated table actions through an untyped delegate
            TableActions["prisma_models.LiteLLM_ClaudeCodePluginUpload"],
            prisma_client.writer_db.litellm_claudecodepluginupload,
        )
        self._downloads = cast(  # cast-ok: Prisma exposes generated table actions through an untyped delegate
            TableActions["prisma_models.LiteLLM_ClaudeCodePluginDownload"],
            prisma_client.writer_db.litellm_claudecodeplugindownload,
        )

    async def create(self, upload: PluginUpload) -> str | None:
        result: Final = await self.publish(
            PluginPublication(
                name=upload.name,
                version=upload.version,
                description=upload.description,
                manifest=upload.manifest,
                content=upload.content,
                created_by=upload.created_by,
            )
        )
        return result.id if isinstance(result, PublishedPlugin) else None

    async def publish(
        self, publication: PluginPublication, expected: PluginPublicationBase | None = None
    ) -> PublishedPlugin | PluginPublicationError:
        from prisma.errors import UniqueViolationError

        try:
            async with self._db.tx() as tx:
                return await publish_plugin(tx, publication, expected)
        except UniqueViolationError:
            return PluginPublicationError("conflict", "A skill with this name or version already exists")

    async def versions(self, metadata: PluginArchiveMetadata, limit: int, offset: int) -> tuple[PluginVersion, ...]:
        rows: Final = await self._db.query_raw(
            _VERSION_QUERY + " ORDER BY v.published_at DESC NULLS LAST, v.id DESC LIMIT $3 OFFSET $4",
            metadata.id,
            metadata.enabled,
            limit,
            offset,
        )
        return _VERSIONS.validate_python(rows)

    async def version(self, metadata: PluginArchiveMetadata, version_id: str) -> PluginVersion | None:
        rows: Final = await self._db.query_raw(
            _VERSION_QUERY + " AND v.id = $3",
            metadata.id,
            metadata.enabled,
            version_id,
        )
        versions: Final = _VERSIONS.validate_python(rows)
        return versions[0] if versions else None

    async def version_content(self, metadata: PluginArchiveMetadata, version_id: str) -> bytes | None:
        version: Final = await self._db.litellm_claudecodepluginversion.find_first(
            where={
                "id": version_id,
                "plugin_id": metadata.id,
                "plugin": {
                    "is": {
                        "enabled": metadata.enabled,
                    }
                },
            }
        )
        if version is None or version.content is None:
            return None
        return verified_archive(version.content.decode(), version.manifest_json)

    async def get_plugin(self, name: str) -> PluginSnapshot | None:
        row: Final = await self._plugins.find_unique(where={"name": name})
        if row is None:
            return None
        return PluginSnapshot(
            id=row.id,
            name=row.name,
            version=row.version,
            description=row.description,
            manifest=MappingProxyType(_MANIFEST.validate_json(row.manifest_json or "{}")),
            enabled=row.enabled,
            created_at=row.created_at,
            updated_at=row.updated_at,
            created_by=row.created_by,
        )

    async def get_metadata(self, name: str) -> PluginArchiveMetadata | None:
        plugin: Final = await self._plugins.find_unique(where={"name": name})
        if plugin is None:
            return None
        return _archive_metadata(plugin)

    async def public_metadata(self) -> AsyncIterator[PluginArchiveMetadata]:
        after: str | None = None  # rebind-ok: bounded keyset traversal advances after each raw page
        while True:
            plugins: Sequence[prisma_models.LiteLLM_ClaudeCodePluginTable] = await self._plugins.find_many(
                where={"enabled": True} if after is None else {"enabled": True, "name": {"gt": after}},
                take=PAGE_SIZE,
                order={"name": "asc"},
            )
            if not plugins:
                return
            for plugin in plugins:
                try:
                    metadata: PluginArchiveMetadata = _archive_metadata(plugin)
                except ValidationError:
                    continue
                yield metadata
            if len(plugins) < PAGE_SIZE:
                return
            after = plugins[-1].name

    async def get_content(self, metadata: PluginArchiveMetadata) -> bytes | None:
        archive: Final = await self._archives.find_first(
            where={
                "plugin_name": metadata.name,
                "plugin": {
                    "is": {
                        "id": metadata.id,
                        "manifest_json": metadata.manifest_json,
                        "enabled": metadata.enabled,
                    }
                },
            }
        )
        return None if archive is None else archive.content.decode()

    async def record_download(
        self, metadata: PluginArchiveMetadata, digest: str, downloader: PluginDownloader
    ) -> PluginDownloadResult:
        from prisma.errors import ForeignKeyViolationError

        if metadata.id is None:
            return PluginDownloadResult.SKILL_NOT_FOUND
        if not downloader.authenticated:
            # A counter bounds anonymous storage; raw SQL preserves Prisma's @updatedAt metadata.
            affected: Final = await self._writer.execute_raw(_INCREMENT_ANONYMOUS_DOWNLOADS, metadata.id)
            return PluginDownloadResult.RECORDED if affected == 1 else PluginDownloadResult.SKILL_NOT_FOUND
        try:
            await self._downloads.create(
                data={
                    "plugin_id": metadata.id,
                    "version": metadata.version,
                    "digest": digest,
                    "authenticated": downloader.authenticated,
                    "user_id": downloader.user_id,
                    "user_email": downloader.user_email,
                    "key_alias": downloader.key_alias,
                    "team_id": downloader.team_id,
                    "team_alias": downloader.team_alias,
                }
            )
        except ForeignKeyViolationError:
            return PluginDownloadResult.SKILL_NOT_FOUND
        return PluginDownloadResult.RECORDED

    async def downloads(self, name: str, limit: int, cursor: int | None) -> PluginDownloadsResponse | None:
        plugin: Final = await self._plugins.find_unique(where={"name": name})
        if plugin is None:
            return None
        total: Final = await self._downloads.count(where={"plugin_id": plugin.id})
        rows: Final = await self._downloads.find_many(
            where={"plugin_id": plugin.id, **({"id": {"lt": cursor}} if cursor is not None else {})},
            order={"id": "desc"},
            take=limit + 1,
        )
        return PluginDownloadsResponse(
            total_downloads=total + plugin.anonymous_download_count,
            anonymous_downloads=plugin.anonymous_download_count,
            downloads=tuple(
                PluginDownloadEvent(
                    id=str(row.id),
                    downloaded_at=row.downloaded_at,
                    version=row.version,
                    digest=row.digest,
                    authenticated=row.authenticated,
                    user_id=row.user_id,
                    user_email=row.user_email,
                    key_alias=row.key_alias,
                    team_id=row.team_id,
                    team_alias=row.team_alias,
                )
                for row in rows[:limit]
            ),
            next_cursor=str(rows[limit - 1].id) if len(rows) > limit else None,
        )

    async def download_counts(self, plugin_ids: tuple[str, ...]) -> Mapping[str, int]:
        if not plugin_ids:
            return MappingProxyType({})
        rows: Final = _DOWNLOAD_COUNTS.validate_python(await self._db.query_raw(_DOWNLOAD_COUNTS_QUERY, plugin_ids))
        return MappingProxyType({row.id: row.download_count for row in rows})


def _archive_metadata(plugin: "prisma_models.LiteLLM_ClaudeCodePluginTable") -> PluginArchiveMetadata:
    manifest: Final = _MANIFEST.validate_json(plugin.manifest_json or "{}")
    return PluginArchiveMetadata(
        name=plugin.name,
        enabled=plugin.enabled,
        source=MappingProxyType(_SOURCE.validate_python(manifest.get("source", {}))),
        manifest_json=plugin.manifest_json,
        id=plugin.id,
        version=plugin.version,
        description=plugin.description,
        has_agent_skill_metadata=has_agent_skill_metadata(manifest),
        updated_at=plugin.updated_at,
        created_at=plugin.created_at,
        created_by=plugin.created_by,
    )
