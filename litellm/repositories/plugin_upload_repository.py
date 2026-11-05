import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Prisma dynamically exposes its typed table actions

from pydantic import TypeAdapter

from litellm.models.plugin_upload import PluginArchiveMetadata, PluginUpload
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ClaudeCodePluginRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

_MANIFEST: Final = TypeAdapter(dict[str, object])
_SOURCE: Final = TypeAdapter(dict[str, str])


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
        manifest: Final = _MANIFEST.validate_json(plugin.manifest_json or "{}")
        return PluginArchiveMetadata(
            name=plugin.name,
            enabled=plugin.enabled,
            source=MappingProxyType(_SOURCE.validate_python(manifest.get("source", {}))),
            manifest_json=plugin.manifest_json,
        )

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
