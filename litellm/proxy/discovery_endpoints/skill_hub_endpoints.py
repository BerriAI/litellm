import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import chain
from typing import Annotated, Final, Literal
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile

import litellm
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory has untyped optional params
)
from litellm.models.plugin_upload import (
    PluginArchiveMetadata,
    PluginDownloader,
    PluginDownloadsResponse,
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginSnapshot,
    PluginUploadStore,
    PluginVersion,
    PublishedPlugin,
    has_agent_skill_metadata,
)
from litellm.models.skill_submission import SkillSubmission, SkillSubmissionStore
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_marketplace import (
    MANIFEST_FIELDS,
    delete_plugin,
    disable_plugin,
    enable_plugin,
    get_plugin_downloads,
    header_key_caller,
    list_visible_plugins,
    plugin_archive_source_matches,
    plugin_archive_url,
    plugin_downloader,
    require_download_recorded,
)
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import resolve_skill_visibility
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.skill_upload_archive import build_uploaded_skill_archive
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.claude_code_marketplace_auth import (
    marketplace_auth_required,
    marketplace_key_header_present,
)
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.deps.repositories import PluginUploadRepositoryDep, SkillSubmissionRepositoryDep
from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    SkillArchive,
    build_agent_skill_archive,
)
from litellm.proxy.discovery_endpoints.skill_hub_mutations import (
    SkillMutationError,
    archive_writer,
    publication_error,
    write_skill,
    write_target,
)
from litellm.proxy.discovery_endpoints.skill_import import (
    MAX_IMPORT_BYTES,
    SkillImportError,
    fetch_skill_source,
    parse_skill_source,
    select_imported_skill,
)
from litellm.repositories.plugin_publication import read_plugin_publisher, read_plugin_review
from litellm.types.proxy.claude_code_endpoints import PluginListItem
from litellm.types.proxy.discovery_endpoints.agent_skills_endpoints import AgentSkillsIndexEntry
from litellm.types.proxy.skill_hub import (
    ImportSkillRequest,
    SkillHubList,
    SkillHubMetadata,
    SkillHubMutationResponse,
    SkillHubPendingSkill,
    SkillHubProposal,
    SkillHubPublication,
    SkillHubRestore,
    SkillHubSkill,
    SkillHubVersion,
    SkillHubVersions,
    SkillSortField,
)

router: Final = APIRouter(prefix="/skill-hub/skills", tags=["Skills Hub"])
_EDITABLE_METADATA: Final = frozenset(SkillHubMetadata.model_fields)
router.add_api_route(
    "/{plugin_name}/downloads", get_plugin_downloads, methods=["GET"], response_model=PluginDownloadsResponse
)


@dataclass(frozen=True, slots=True)
class VisibleSkillArchive:
    archive: SkillArchive
    metadata: PluginArchiveMetadata
    downloader: PluginDownloader


def _url(request: Request, route: str) -> str:
    from litellm.proxy.utils import get_custom_url

    return get_custom_url(request_base_url=str(request.base_url), route=route.lstrip("/"))


def _download_url(request: Request, name: str) -> str:
    return _url(request, f"skill-hub/skills/{quote(name, safe='')}/archive")


def _skill_response(request: Request, plugin: PluginListItem, download_count: int | None = None) -> SkillHubSkill:
    local_archive: Final = plugin_archive_source_matches(
        request,
        plugin.name,
        plugin.source,
        plugin.source.get("sha256", ""),
        has_agent_skill_metadata=plugin.has_agent_skill_metadata,
    )
    return SkillHubSkill(
        id=plugin.id,
        name=plugin.name,
        description=plugin.description,
        version=plugin.version,
        domain=plugin.domain,
        namespace=plugin.namespace,
        keywords=None if plugin.keywords is None else tuple(plugin.keywords),
        updated_at=None if plugin.updated_at is None else datetime.fromisoformat(plugin.updated_at),
        created_at=None if plugin.created_at is None else datetime.fromisoformat(plugin.created_at),
        created_by=plugin.created_by,
        approved_by=plugin.approved_by,
        published_by=plugin.published_by,
        security_scan_passed=plugin.security_scan_passed,
        visibility="public" if plugin.enabled else "private",
        source={"source": "archive", "url": _download_url(request, plugin.name)} if local_archive else plugin.source,
        download_url=_download_url(request, plugin.name) if local_archive else None,
        content_url=_url(request, f"skill-hub/skills/{quote(plugin.name, safe='')}/content") if local_archive else None,
        download_count=download_count,
    )


@router.get("", response_model=SkillHubList)
async def list_skills(
    request: Request,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    query: str = "",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    tag: str = "",
    security_scan_passed: bool | None = None,
    sort_by: SkillSortField = "name",
    sort_order: Literal["asc", "desc"] = "asc",
) -> SkillHubList:
    """Browse the caller's public and granted private skills."""
    catalog: Final = await list_visible_plugins(enabled_only=False, user_api_key_dict=caller)
    needle: Final = query.strip().casefold()
    selected_tag: Final = tag.strip().casefold()
    matched: Final = tuple(
        plugin
        for plugin in sorted(catalog.plugins, key=lambda item: item.name)
        if needle
        in " ".join(
            (
                plugin.name,
                plugin.description or "",
                plugin.domain or "",
                plugin.namespace or "",
                plugin.created_by or "",
                plugin.approved_by or "",
                plugin.published_by or "",
                *(plugin.keywords or ()),
            )
        ).casefold()
        and (not selected_tag or selected_tag in {word.casefold() for word in plugin.keywords or ()})
        and (security_scan_passed is None or plugin.security_scan_passed == security_scan_passed)
    )
    counts: Final[Mapping[str, int]] = (
        await _require_store(uploads).download_counts(tuple(plugin.id for plugin in matched))
        if sort_by == "download_count"
        else {}
    )
    ordered: Final = sorted(
        matched, key=lambda plugin: _skill_sort_key(plugin, sort_by, counts), reverse=sort_order == "desc"
    )
    page: Final = ordered[offset : offset + limit]
    page_counts: Final = (
        await _require_store(uploads).download_counts(tuple(plugin.id for plugin in page))
        if sort_by != "download_count"
        else counts
    )
    return SkillHubList(
        skills=tuple(_skill_response(request, plugin, page_counts.get(plugin.id)) for plugin in page),
        count=len(matched),
        next_offset=offset + limit if offset + limit < len(matched) else None,
        tags=tuple(
            sorted(set(chain.from_iterable(plugin.keywords or () for plugin in catalog.plugins)), key=str.casefold)
        ),
    )


def _skill_sort_key(plugin: PluginListItem, field: SkillSortField, counts: Mapping[str, int]) -> str | float:
    if field == "download_count":
        return counts.get(plugin.id, 0)
    if field == "security_scan_passed":
        return int(plugin.security_scan_passed)
    if field in {"created_at", "updated_at"}:
        value: Final = plugin.created_at if field == "created_at" else plugin.updated_at
        return datetime.fromisoformat(value).timestamp() if value else 0
    return {
        "name": plugin.name,
        "created_by": plugin.created_by or "",
        "published_by": plugin.published_by or "",
        "approved_by": plugin.approved_by or "",
    }[field].casefold()


@router.get("/{name}", response_model=SkillHubSkill)
async def get_skill(
    request: Request,
    name: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubSkill:
    """Read one visible skill and its portable download URL."""
    store: Final = _require_store(uploads)
    metadata: Final = await store.get_metadata(name)
    if metadata is None or not (await resolve_skill_visibility(caller)).allows(metadata):
        raise HTTPException(404, "Skill not found")
    return skill_metadata_response(request, metadata)


def skill_metadata_response(request: Request, metadata: PluginArchiveMetadata) -> SkillHubSkill:
    manifest: Final = MANIFEST_FIELDS.validate_json(metadata.manifest_json or "{}")
    review: Final = read_plugin_review(manifest)
    return _skill_response(
        request,
        PluginListItem.model_validate(
            {
                "id": metadata.id or metadata.name,
                "name": metadata.name,
                "description": metadata.description,
                "version": metadata.version,
                "source": dict(metadata.source),
                "has_agent_skill_metadata": metadata.has_agent_skill_metadata,
                "enabled": metadata.enabled,
                "created_at": metadata.created_at.isoformat() if metadata.created_at is not None else None,
                "created_by": metadata.created_by,
                "approved_by": review.approved_by if review else None,
                "published_by": read_plugin_publisher(manifest),
                "security_scan_passed": review.security_scan_passed if review else False,
                "updated_at": metadata.updated_at.isoformat() if metadata.updated_at is not None else None,
                "domain": manifest.get("domain"),
                "namespace": manifest.get("namespace"),
                "keywords": manifest.get("keywords"),
            }
        ),
    )


def _require_store(uploads: PluginUploadStore | None) -> PluginUploadStore:
    if uploads is None:
        raise HTTPException(503, "Skills Hub requires a connected database")
    return uploads


def _require_admin(caller: UserAPIKeyAuth) -> None:
    if not is_proxy_admin(caller):
        raise HTTPException(403, "Only proxy admins may publish skills")


@router.put("/{name}", response_model=SkillHubMutationResponse)
async def replace_skill_metadata(
    request: Request,
    name: str,
    body: SkillHubMetadata,
    uploads: PluginUploadRepositoryDep,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubMutationResponse:
    """Replace registry metadata, preserving files. Members submit an immutable proposal for admin review."""
    existing: Final = await write_target(name, True, uploads, caller, body.expected_updated_at)
    if isinstance(existing, SkillMutationError):
        raise HTTPException(existing.status, existing.message)
    if existing is None:
        raise HTTPException(404, "Skill not found")
    publication: Final = PluginPublication(
        name=name,
        version=body.version,
        description=body.description,
        manifest={
            **{key: value for key, value in existing.manifest.items() if key not in _EDITABLE_METADATA},
            **body.model_dump(exclude_none=True),
            "name": name,
        },
        created_by=caller.user_id,
    )
    return await _mutate(request, publication, existing, _require_store(uploads), submissions, caller)


@router.post("/{name}/publish", response_model=SkillHubSkill)
async def publish_skill(
    request: Request,
    name: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    _body: SkillHubPublication | None = None,
) -> SkillHubSkill:
    """Make a skill public using the existing Hub publication policy."""
    _require_admin(caller)
    await enable_plugin(plugin_name=name, user_api_key_dict=caller)
    return await get_skill(request, name, uploads, caller)


@router.post("/{name}/unpublish", response_model=SkillHubSkill)
async def unpublish_skill(
    request: Request,
    name: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    _body: SkillHubPublication | None = None,
) -> SkillHubSkill:
    """Remove a skill from public discovery while retaining explicit grants."""
    _require_admin(caller)
    await disable_plugin(plugin_name=name, user_api_key_dict=caller)
    return await get_skill(request, name, uploads, caller)


@router.delete("/{name}", status_code=204, response_class=Response)
async def remove_skill(name: str, caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> Response:
    """Delete a skill and its managed archive."""
    _require_admin(caller)
    await delete_plugin(plugin_name=name, user_api_key_dict=caller)
    return Response(status_code=204)


@router.post("/upload", response_model=SkillHubMutationResponse)
async def upload_skill(
    request: Request,
    uploads: PluginUploadRepositoryDep,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(archive_writer)],
    file: Annotated[UploadFile, File(description="Agent Skills ZIP with SKILL.md and optional supporting files")],
    version: Annotated[str | None, Form()] = None,
) -> SkillHubMutationResponse:
    """Publish a portable skill while retaining Claude marketplace compatibility."""
    store: Final = _require_store(uploads)
    content: Final = await file.read(MAX_IMPORT_BYTES + 1)
    if len(content) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "Skill ZIP must be 10 MiB or smaller")
    archive: Final = await asyncio.to_thread(build_agent_skill_archive, content)
    if archive is None:
        raise HTTPException(422, "Upload a valid Agent Skill with name and description in SKILL.md")
    return await _publish(request, store, submissions, caller, archive, version, None)


def import_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(llm_provider="skill_hub_import").client


@router.post("/import", response_model=SkillHubMutationResponse)
async def import_skill(
    request: Request,
    body: ImportSkillRequest,
    uploads: PluginUploadRepositoryDep,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(archive_writer)],
    client: Annotated[httpx.AsyncClient, Depends(import_http_client)],
) -> SkillHubMutationResponse:
    """Import a public skills.sh or GitHub skill as a portable snapshot."""
    store: Final = _require_store(uploads)
    source: Final = parse_skill_source(body.url, skill=body.skill, ref=body.ref, path=body.path)
    if isinstance(source, SkillImportError):
        raise HTTPException(source.status, source.message)
    content: Final = await fetch_skill_source(source, client)
    if isinstance(content, SkillImportError):
        raise HTTPException(content.status, content.message)
    archive: Final = await asyncio.to_thread(select_imported_skill, content, source)
    if isinstance(archive, SkillImportError):
        raise HTTPException(archive.status, archive.message)
    return await _publish(request, store, submissions, caller, archive, body.version, body.url)


async def _publish(
    request: Request,
    store: PluginUploadStore,
    submissions: SkillSubmissionStore | None,
    caller: UserAPIKeyAuth,
    archive: SkillArchive,
    version: str | None,
    origin: str | None,
) -> SkillHubMutationResponse:
    existing: Final = await write_target(archive.declared_name or "", False, store, caller)
    if isinstance(existing, SkillMutationError):
        raise HTTPException(existing.status, existing.message)
    publication: Final = await prepare_skill_publication(request, archive, version, origin, caller.user_id)
    return await _mutate(request, publication, existing, store, submissions, caller)


async def _mutate(
    request: Request,
    publication: PluginPublication,
    existing: PluginSnapshot | None,
    store: PluginUploadStore,
    submissions: SkillSubmissionStore | None,
    caller: UserAPIKeyAuth,
) -> SkillHubMutationResponse:
    result: Final = await write_skill(publication, existing, store, submissions, caller)
    if isinstance(result, SkillMutationError):
        raise HTTPException(result.status, result.message)
    if isinstance(result, SkillSubmission):
        return SkillHubPendingSkill(submission_id=result.id, proposal=skill_proposal_response(request, result))
    return skill_publication_response(request, publication, result)


def skill_proposal_response(request: Request, submission: SkillSubmission) -> SkillHubProposal:
    proposal: Final = submission.proposal
    return SkillHubProposal.model_validate(
        {
            **proposal.manifest,
            "name": proposal.name,
            "description": proposal.description,
            "version": proposal.version,
            **(
                {
                    "source": {
                        "source": "archive",
                        "url": _url(request, f"skill-hub/submissions/{quote(submission.id, safe='')}/archive"),
                    }
                }
                if submission.has_archive
                else {}
            ),
        }
    )


@router.put("/{name}/upload", response_model=SkillHubMutationResponse)
async def replace_skill_archive(
    request: Request,
    name: str,
    uploads: PluginUploadRepositoryDep,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(archive_writer)],
    file: Annotated[UploadFile, File(description="Replacement Agent Skills ZIP with the same declared skill name")],
    version: Annotated[str | None, Form()] = None,
    expected_updated_at: Annotated[datetime | None, Form()] = None,
) -> SkillHubMutationResponse:
    """Replace skill files, retaining registry fields. Members propose an update against the original editor timestamp."""
    existing: Final = await write_target(name, True, uploads, caller, expected_updated_at)
    if isinstance(existing, SkillMutationError):
        raise HTTPException(existing.status, existing.message)
    if existing is None:
        raise HTTPException(404, "Skill not found")
    content: Final = await file.read(MAX_IMPORT_BYTES + 1)
    if len(content) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "Skill ZIP must be 10 MiB or smaller")
    archive: Final = await asyncio.to_thread(build_agent_skill_archive, content)
    if archive is None or archive.declared_name != name:
        raise HTTPException(422, "Replacement SKILL.md must declare the existing skill name and a description")
    prepared: Final = await prepare_skill_publication(request, archive, version, None, caller.user_id)
    publication: Final = replace(prepared, manifest={**existing.manifest, **prepared.manifest})
    return await _mutate(request, publication, existing, _require_store(uploads), submissions, caller)


def skill_publication_response(
    request: Request, publication: PluginPublication, published: PublishedPlugin
) -> SkillHubSkill:
    return _skill_response(
        request,
        PluginListItem.model_validate(
            {
                "source": publication.manifest.get("source", {}),
                "id": published.id,
                "name": published.name,
                "description": publication.description,
                "version": published.version,
                "enabled": published.enabled,
                "has_agent_skill_metadata": has_agent_skill_metadata(publication.manifest),
                "created_at": published.created_at.isoformat() if published.created_at is not None else None,
                "created_by": published.created_by,
                "approved_by": published.review.approved_by if published.review else None,
                "published_by": published.published_by,
                "security_scan_passed": published.review.security_scan_passed if published.review else False,
                "updated_at": published.updated_at.isoformat(),
                "domain": publication.manifest.get("domain"),
                "namespace": publication.manifest.get("namespace"),
                "keywords": publication.manifest.get("keywords"),
            }
        ),
    )


async def prepare_skill_publication(
    request: Request,
    archive: SkillArchive,
    version: str | None,
    origin: str | None,
    created_by: str | None,
) -> PluginPublication:
    name: Final = archive.declared_name
    if name is None:
        raise HTTPException(422, "SKILL.md must declare a skill name")
    plugin: Final = await asyncio.to_thread(build_uploaded_skill_archive, archive.content, name)
    if plugin is None:
        raise HTTPException(422, "Skill ZIP exceeds the supported archive limits")
    portable: Final = await asyncio.to_thread(build_agent_skill_archive, plugin, plugin_name=name)
    if portable is None or portable.digest != archive.digest:
        raise HTTPException(422, "Skill ZIP exceeds the supported archive limits after compatibility packaging")
    digest: Final = hashlib.sha256(plugin).hexdigest()
    source: Final = {
        "source": "archive",
        "url": plugin_archive_url(request, name, digest),
        "sha256": digest,
    }
    manifest: Final = {
        "name": name,
        "description": archive.declared_description,
        "version": version,
        "source": source,
        "agent_skill": {
            "name": name,
            "description": archive.declared_description,
            "digest": archive.digest,
            "origin": origin,
        },
    }
    return PluginPublication(
        name=name,
        version=version,
        description=archive.declared_description,
        manifest=manifest,
        content=plugin,
        created_by=created_by,
    )


async def _version_metadata(name: str, store: PluginUploadStore, caller: UserAPIKeyAuth) -> PluginArchiveMetadata:
    metadata: Final = await store.get_metadata(name)
    if metadata is None or metadata.id is None or not (await resolve_skill_visibility(caller)).allows(metadata):
        raise HTTPException(404, "Skill not found")
    return metadata


def _version_response(request: Request, name: str, version: PluginVersion) -> SkillHubVersion:
    return SkillHubVersion(
        id=version.id,
        version=version.version,
        description=version.description,
        manifest=version.manifest,
        published_at=version.published_at,
        published_by=version.published_by,
        content_kind="stored_archive" if version.archive_available else "source_reference",
        download_url=(
            _url(request, f"skill-hub/skills/{quote(name, safe='')}/versions/{quote(version.id, safe='')}/archive")
            if version.archive_available
            else None
        ),
    )


@router.get("/{name}/versions", response_model=SkillHubVersions)
async def list_skill_versions(
    request: Request,
    name: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SkillHubVersions:
    """Read published metadata history. External references are not retained artifacts."""
    store: Final = _require_store(uploads)
    metadata: Final = await _version_metadata(name, store, caller)
    versions: Final = await store.versions(metadata, limit + 1, offset)
    return SkillHubVersions(
        versions=tuple(_version_response(request, name, version) for version in versions[:limit]),
        next_offset=offset + limit if len(versions) > limit else None,
    )


@router.get("/{name}/versions/{version_id}", response_model=SkillHubVersion)
async def get_skill_version(
    request: Request,
    name: str,
    version_id: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubVersion:
    """Read a retained publication under the current skill's visibility policy."""
    store: Final = _require_store(uploads)
    metadata: Final = await _version_metadata(name, store, caller)
    version: Final = await store.version(metadata, version_id)
    if version is None:
        raise HTTPException(404, "Skill version not found")
    return _version_response(request, name, version)


@router.get(
    "/{name}/versions/{version_id}/archive",
    response_class=Response,
    responses={200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}},
)
async def download_skill_version(
    name: str,
    version_id: str,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Download a retained publication as a portable ZIP with SKILL.md at its root."""
    store: Final = _require_store(uploads)
    metadata: Final = await _version_metadata(name, store, caller)
    version: Final = await store.version(metadata, version_id)
    if version is None:
        raise HTTPException(404, "Skill version not found")
    content: Final = await store.version_content(metadata, version_id)
    archive: Final = (
        await asyncio.to_thread(build_agent_skill_archive, content, plugin_name=name) if content is not None else None
    )
    if archive is None:
        raise HTTPException(404, "This version has no retained Agent Skills archive")
    require_download_recorded(
        await store.record_download(
            replace(metadata, version=version.version),
            archive.digest.removeprefix("sha256:"),
            plugin_downloader(caller),
        )
    )
    return Response(
        archive.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.zip"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "ETag": f'"{archive.digest}"',
        },
    )


@router.post("/{name}/versions/{version_id}/restore", response_model=SkillHubSkill)
async def restore_skill_version(
    request: Request,
    name: str,
    version_id: str,
    body: SkillHubRestore,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubSkill:
    """Proxy admins may republish retained bytes and metadata as a new version, preserving visibility and grants."""
    _require_admin(caller)
    store: Final = _require_store(uploads)
    metadata: Final = await _version_metadata(name, store, caller)
    if metadata.id != body.expected_skill_id or metadata.updated_at != body.expected_updated_at:
        raise HTTPException(409, "The published skill changed. Reload it before restoring a version")
    version: Final = await store.version(metadata, version_id)
    if version is None:
        raise HTTPException(404, "Skill version not found")
    content: Final = await store.version_content(metadata, version_id) if version.archive_available else None
    if content is None:
        raise HTTPException(409, "This version has no verified retained archive to restore")
    digest: Final = hashlib.sha256(content).hexdigest()
    review: Final = read_plugin_review(version.manifest)
    publication: Final = PluginPublication(
        name=name,
        version=body.version,
        description=version.description,
        manifest={
            **version.manifest,
            "source": {
                "source": "archive",
                "url": plugin_archive_url(request, name, digest),
                "sha256": digest,
            },
        },
        content=content,
        created_by=caller.user_id or "proxy-admin",
        review=replace(review, approved_by=caller.user_id or "proxy-admin") if review is not None else None,
    )
    published: Final = await store.publish(
        publication, PluginPublicationBase(body.expected_skill_id, body.expected_updated_at)
    )
    if isinstance(published, PluginPublicationError):
        error: Final = publication_error(published)
        raise HTTPException(error.status, error.message)
    return skill_publication_response(request, publication, published)


async def _visible_archive(request: Request, name: str, uploads: PluginUploadStore | None) -> VisibleSkillArchive:
    anonymous_allowed: Final = litellm.public_skills_index is True and not marketplace_auth_required()
    caller: Final = (
        await header_key_caller(request) if marketplace_key_header_present(request) or not anonymous_allowed else None
    )
    store: Final = _require_store(uploads)
    metadata: Final = await store.get_metadata(name)
    if metadata is None or not (await resolve_skill_visibility(caller)).allows(metadata):
        raise HTTPException(404, "Skill not found")
    if not plugin_archive_source_matches(
        request,
        name,
        metadata.source,
        metadata.source.get("sha256", ""),
        has_agent_skill_metadata=metadata.has_agent_skill_metadata,
    ):
        raise HTTPException(404, "This skill no longer refers to a managed archive")
    content: Final = await store.get_content(metadata)
    archive: Final = (
        await asyncio.to_thread(build_agent_skill_archive, content, plugin_name=name) if content is not None else None
    )
    if archive is None:
        raise HTTPException(404, "This skill has no stored Agent Skills archive")
    return VisibleSkillArchive(archive, metadata, plugin_downloader(caller))


@router.get(
    "/{name}/archive",
    response_class=Response,
    responses={200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}},
)
async def download_skill(
    request: Request, name: str, uploads: PluginUploadRepositoryDep, digest: str | None = None
) -> Response:
    """Download the complete portable skill, with SKILL.md at the ZIP root."""
    visible: Final = await _visible_archive(request, name, uploads)
    archive: Final = visible.archive
    if digest is not None and digest != archive.digest:
        raise HTTPException(404, "Skill archive version not found")
    require_download_recorded(
        await _require_store(uploads).record_download(
            visible.metadata, archive.digest.removeprefix("sha256:"), visible.downloader
        )
    )
    return Response(
        archive.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.zip"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "ETag": f'"{archive.digest}"',
        },
    )


@router.get("/{name}/content", response_class=Response, responses={200: {"content": {"text/markdown": {}}}})
async def skill_content(request: Request, name: str, uploads: PluginUploadRepositoryDep) -> Response:
    """Read the original SKILL.md without downloading supporting files."""
    visible: Final = await _visible_archive(request, name, uploads)
    return Response(
        visible.archive.manifest_content,
        media_type="text/markdown",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


async def public_hub_entries(request: Request, store: PluginUploadStore | None) -> tuple[AgentSkillsIndexEntry, ...]:
    if store is None or marketplace_auth_required():
        return ()
    entries: Final[list[AgentSkillsIndexEntry]] = []  # mutable-ok: bounded inventory avoids copying on each append
    async for item in store.public_metadata():
        entry: AgentSkillsIndexEntry | None = _index_entry(request, item)
        if entry is None:
            continue
        entries.append(entry)
        if len(entries) == 1000:
            break
    return tuple(entries)


def _index_entry(request: Request, item: PluginArchiveMetadata) -> AgentSkillsIndexEntry | None:
    manifest: Final = MANIFEST_FIELDS.validate_json(item.manifest_json or "{}")
    raw: Final = manifest.get("agent_skill")
    if (
        not isinstance(raw, dict)
        or not item.enabled
        or not plugin_archive_source_matches(
            request,
            item.name,
            item.source,
            item.source.get("sha256", ""),
            has_agent_skill_metadata=item.has_agent_skill_metadata,
        )
    ):
        return None
    skill: Final = MANIFEST_FIELDS.validate_python(raw)
    digest: Final = skill.get("digest")
    description: Final = item.description or skill.get("description")
    if not isinstance(digest, str) or not isinstance(description, str):
        return None
    return AgentSkillsIndexEntry(
        name=item.name,
        type="archive",
        description=description,
        url=f"{_download_url(request, item.name)}?digest={quote(digest, safe=':')}",
        digest=digest,
    )
