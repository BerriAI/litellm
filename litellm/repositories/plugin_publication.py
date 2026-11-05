import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter, ValidationError
from semver import Version

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginReview,
    PublishedPlugin,
)
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma.models import (
        LiteLLM_ClaudeCodePluginTable,
        LiteLLM_ClaudeCodePluginUpload,
        LiteLLM_ClaudeCodePluginVersion,
    )

_MANIFEST: Final = TypeAdapter(dict[str, object])
_SOURCE: Final = TypeAdapter(dict[str, str])
_REVIEW: Final = TypeAdapter(PluginReview)
_PUBLISHER: Final = TypeAdapter[str | None](str | None)


def read_plugin_review(manifest: Mapping[str, object]) -> PluginReview | None:
    try:
        return _REVIEW.validate_python(manifest.get("_litellm_review"))
    except ValidationError:
        return None


def read_plugin_publisher(manifest: Mapping[str, object]) -> str | None:
    try:
        return _PUBLISHER.validate_python(manifest.get("_litellm_published_by"))
    except ValidationError:
        return None


class PluginPublicationTransaction(Protocol):
    @property
    def litellm_claudecodeplugintable(self) -> TableActions["LiteLLM_ClaudeCodePluginTable"]: ...

    @property
    def litellm_claudecodepluginupload(self) -> TableActions["LiteLLM_ClaudeCodePluginUpload"]: ...

    @property
    def litellm_claudecodepluginversion(self) -> TableActions["LiteLLM_ClaudeCodePluginVersion"]: ...

    async def query_raw(self, query: str, *args: object) -> object: ...


def resolve_plugin_version(
    requested: str | None, current: str | None = None, *, updating: bool = False
) -> str | PluginPublicationError:
    try:
        previous: Final = Version.parse(current) if current is not None else None
    except ValueError:
        return _legacy_version(requested)
    if updating and previous is None:
        return _legacy_version(requested)
    try:
        candidate: Final = Version.parse(requested) if requested is not None else None
    except ValueError:
        return PluginPublicationError("invalid_version", "Version must be a valid semantic version, for example 1.0.1")
    if candidate is None:
        return str(previous.bump_patch()) if previous is not None else "1.0.0"
    if previous is not None and candidate.compare(previous) <= 0:
        return PluginPublicationError("invalid_version", f"Version must be strictly higher than {current}")
    return str(candidate)


def _legacy_version(requested: str | None) -> str | PluginPublicationError:
    if requested is None:
        return PluginPublicationError(
            "conflict",
            "The published version is missing or is not SemVer. Supply an explicit semantic version to update it",
        )
    return resolve_plugin_version(requested)


def verified_archive(content: bytes | None, manifest_json: str | None) -> bytes | None:
    if content is None:
        return None
    manifest: Final = _MANIFEST.validate_json(manifest_json or "{}")
    source: Final = _SOURCE.validate_python(manifest.get("source", {}))
    if source.get("source") != "archive" or source.get("sha256", "").lower() != hashlib.sha256(content).hexdigest():
        return None
    return content


async def _snapshot(
    tx: PluginPublicationTransaction,
    plugin: "LiteLLM_ClaudeCodePluginTable",
    content: bytes | None,
    published_by: str | None,
) -> None:
    from prisma import Base64

    archive: Final = verified_archive(content, plugin.manifest_json)
    await tx.litellm_claudecodepluginversion.create(
        data={
            "plugin_id": plugin.id,
            "version": plugin.version,
            "description": plugin.description,
            "manifest_json": plugin.manifest_json,
            "published_at": plugin.updated_at,
            "published_by": published_by,
            "content": Base64.encode(archive) if archive is not None else None,
            "archive_available": archive is not None,
        }
    )


async def publish_plugin(
    tx: PluginPublicationTransaction,
    publication: PluginPublication,
    expected: PluginPublicationBase | None = None,
) -> PublishedPlugin | PluginPublicationError:
    from prisma import Base64

    await tx.query_raw('SELECT id FROM "LiteLLM_ClaudeCodePluginTable" WHERE name = $1 FOR UPDATE', publication.name)
    current: Final = await tx.litellm_claudecodeplugintable.find_unique(where={"name": publication.name})
    if current is not None and expected is None:
        return PluginPublicationError("conflict", f"A skill named '{publication.name}' already exists")
    if expected is not None and current is None:
        return PluginPublicationError("not_found", f"Plugin '{publication.name}' not found")
    if current is not None and (
        expected is not None and (current.id != expected.id or current.updated_at != expected.updated_at)
    ):
        return PluginPublicationError("conflict", "The published skill changed. Reload it before publishing an update")
    version: Final = resolve_plugin_version(
        publication.version, current.version if current is not None else None, updating=expected is not None
    )
    if isinstance(version, PluginPublicationError):
        return version
    publisher: Final = publication.review.approved_by if publication.review is not None else publication.created_by
    manifest_json: Final = json.dumps(
        {
            **{
                key: value
                for key, value in publication.manifest.items()
                if key not in {"_litellm_review", "_litellm_published_by"}
            },
            "version": version,
            **({"_litellm_published_by": publisher} if publisher is not None else {}),
            **({"_litellm_review": asdict(publication.review)} if publication.review is not None else {}),
        }
    )
    if publication.content is not None and verified_archive(publication.content, manifest_json) is None:
        return PluginPublicationError("conflict", "Uploaded archive does not match its published source digest")
    timestamp: Final = datetime.now(timezone.utc)
    data: Final = {
        "version": version,
        "description": publication.description,
        "manifest_json": manifest_json,
        "files_json": "{}",
        "updated_at": timestamp,
    }
    if current is None:
        created: Final = await tx.litellm_claudecodeplugintable.create(
            data={**data, "name": publication.name, "created_by": publication.created_by}
        )
        if publication.content is not None:
            await tx.litellm_claudecodepluginupload.create(
                data={"plugin_name": publication.name, "content": Base64.encode(publication.content)}
            )
        await _snapshot(tx, created, publication.content, publisher)
        return PublishedPlugin(
            created.id,
            created.name,
            version,
            created.enabled,
            created.updated_at or timestamp,
            created.created_at,
            created.created_by,
            publication.review,
            publisher,
        )
    previous_archive: Final = await tx.litellm_claudecodepluginupload.find_unique(where={"plugin_name": current.name})
    previous_content: Final = (
        verified_archive(previous_archive.content.decode(), current.manifest_json)
        if previous_archive is not None
        else None
    )
    history_count: Final = await tx.litellm_claudecodepluginversion.count(where={"plugin_id": current.id}, take=1)
    if history_count == 0:
        await _snapshot(tx, current, previous_content, None)
    unchanged_content: Final = (
        previous_content
        if _MANIFEST.validate_json(current.manifest_json or "{}").get("source") == publication.manifest.get("source")
        else None
    )
    content: Final = publication.content if publication.content is not None else unchanged_content
    updated: Final = await tx.litellm_claudecodeplugintable.update(where={"id": current.id}, data=data)
    if updated is None:
        raise RuntimeError("Locked plugin disappeared during publication")
    if content is None:
        await tx.litellm_claudecodepluginupload.delete_many(where={"plugin_name": current.name})
    else:
        await tx.litellm_claudecodepluginupload.upsert(
            where={"plugin_name": current.name},
            data={
                "create": {"plugin_name": current.name, "content": Base64.encode(content)},
                "update": {"content": Base64.encode(content)},
            },
        )
    await _snapshot(tx, updated, content, publisher)
    return PublishedPlugin(
        updated.id,
        updated.name,
        version,
        updated.enabled,
        updated.updated_at or timestamp,
        updated.created_at,
        updated.created_by,
        publication.review,
        publisher,
    )
