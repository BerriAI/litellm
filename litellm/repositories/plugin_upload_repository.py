import json
from collections.abc import AsyncIterator, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Prisma dynamically exposes its typed table actions

from pydantic import TypeAdapter, ValidationError

from litellm.models.plugin_upload import PluginArchiveMetadata, PluginUpload, has_agent_skill_metadata
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ClaudeCodePluginRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

_MANIFEST: Final = TypeAdapter(dict[str, object])
_SOURCE: Final = TypeAdapter(dict[str, str])
PAGE_SIZE: Final = 100


class PluginUploadRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._plugins = ClaudeCodePluginRepository(prisma_client, use_writer=True).table
        self._archives = cast(  # cast-ok: Prisma exposes generated table actions through an untyped delegate
            TableActions["prisma_models.LiteLLM_ClaudeCodePluginUpload"],
            prisma_client.writer_db.litellm_claudecodepluginupload,
        )

    async def create(self, upload: PluginUpload) -> str | None:
        from prisma import Base64
        from prisma.errors import UniqueViolationError

        try:
            plugin: Final = await self._plugins.create(
                data={
                    "name": upload.name,
                    "version": upload.version,
                    "description": upload.description,
                    "manifest_json": json.dumps(dict(upload.manifest)),
                    "created_by": upload.created_by,
                    "uploaded_archive": {"create": {"content": Base64.encode(upload.content)}},
                }
            )
        except UniqueViolationError:
            return None
        return plugin.id

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
                        "manifest_json": metadata.manifest_json,
                        "enabled": metadata.enabled,
                    }
                },
            }
        )
        return None if archive is None else archive.content.decode()


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
    )
