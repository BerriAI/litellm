"""
CLAUDE CODE MARKETPLACE

Provides a registry/discovery layer for Claude Code plugins.
Plugins are stored as metadata + git source references in LiteLLM database.
Actual plugin files are hosted on GitHub/GitLab/Bitbucket.

Endpoints:
/claude-code/marketplace.json  - GET  - List plugins for Claude Code discovery
/claude-code/plugins           - POST - Register a new plugin (create-only)
/claude-code/plugins           - GET  - List plugins (admin)
/claude-code/plugins/{name}    - GET  - Get plugin details
/claude-code/plugins/{name}    - PUT  - Update an existing plugin
/claude-code/plugins/{name}/enable  - POST - Enable a plugin
/claude-code/plugins/{name}/disable - POST - Disable a plugin
/claude-code/plugins/{name}/approve - POST - Approve a submitted plugin (admin)
/claude-code/plugins/{name}/reject  - POST - Reject a submitted plugin (admin)
/claude-code/plugins/{name}    - DELETE - Delete a plugin

Skills registered by a non-admin are submissions: they are stored with
approval_status="pending_review" and disabled until an administrator approves
them, so only approved skills reach marketplace.json and the public Skill Hub.
This mirrors the MCP server and guardrail submission flows.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Annotated, Final, Protocol, TypedDict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
)
from litellm.repositories.table_repositories import ClaudeCodePluginRepository
from litellm.types.proxy.claude_code_endpoints import (
    SKILL_ACTIVE,
    SKILL_PENDING_REVIEW,
    SKILL_REJECTED,
    ApprovePluginRequest,
    ListPluginsResponse,
    PluginListItem,
    PluginResponse,
    PluginSpec,
    RegisterPluginRequest,
    RegisterPluginResponse,
    RejectPluginRequest,
    ReviewPluginResponse,
    SkillApprovalStatus,
    UpdatePluginRequest,
)

router: Final = APIRouter()


class _PluginRecord(Protocol):
    id: str
    name: str
    version: str | None
    description: str | None
    manifest_json: str | None
    enabled: bool
    approval_status: str | None
    review_notes: str | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    created_by: str | None


class _MarketplaceEntry(TypedDict, total=False):
    name: str
    source: object
    version: str
    description: str
    author: object
    homepage: object
    keywords: object
    category: object


def published_skill_filter() -> dict[str, object]:  # mutable-ok: prisma query arguments must be plain dicts
    """Where-clause for the skills served to users: admin-approved and enabled."""
    return {"enabled": True, "approval_status": SKILL_ACTIVE}  # mutable-ok: prisma query arguments are dicts


def _submitter_edit_resets_review(*, is_admin: bool) -> Mapping[str, object]:
    if is_admin:
        return MappingProxyType({})
    return MappingProxyType({"approval_status": SKILL_PENDING_REVIEW, "enabled": False})


def _manifest_fingerprint(manifest_json: str | None) -> str:
    """Names the exact submitted content, so an approval can be tied to the manifest the administrator read."""
    return hashlib.sha256((manifest_json or "").encode()).hexdigest()


def _as_approval_status(raw: str | None) -> SkillApprovalStatus:
    """Rows written before approval existed, and any hand-edited row, read back as active."""
    match raw:
        case "pending_review":
            return SKILL_PENDING_REVIEW
        case "rejected":
            return SKILL_REJECTED
        case _:
            return SKILL_ACTIVE


def _caller_can_see(plugin: "_PluginRecord", user_api_key_dict: UserAPIKeyAuth) -> bool:
    return (
        is_proxy_admin(user_api_key_dict)
        or _as_approval_status(plugin.approval_status) == SKILL_ACTIVE
        or plugin.created_by in get_resource_owner_scopes(user_api_key_dict)
    )


def _list_plugins_filter(
    *,
    enabled_only: bool,
    approval_status: SkillApprovalStatus | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> dict[str, object]:  # mutable-ok: prisma query arguments must be plain dicts
    status_terms: Final[tuple[tuple[str, object], ...]] = (
        *((("enabled", True),) if enabled_only else ()),
        *((("approval_status", approval_status),) if approval_status is not None else ()),
    )
    if is_proxy_admin(user_api_key_dict):
        return dict(status_terms)  # mutable-ok: prisma query arguments are dicts

    owner_scopes: Final = get_resource_owner_scopes(user_api_key_dict)
    own_skills: Final = ({"created_by": {"in": owner_scopes}},) if owner_scopes else ()  # mutable-ok: prisma dicts
    visible: Final = [{"approval_status": SKILL_ACTIVE}, *own_skills]  # mutable-ok: prisma query arguments are dicts
    return dict((*status_terms, ("OR", visible)))  # mutable-ok: prisma query arguments are dicts


async def _get_prisma_client() -> object:
    """Get the prisma client from proxy_server."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )
    return prisma_client


@router.get(
    "/claude-code/marketplace.json",
    tags=["Claude Code Marketplace"],
)
async def get_marketplace():
    """
    Serve marketplace.json for Claude Code plugin discovery.

    This endpoint is accessed by Claude Code CLI when users run:
    - claude plugin marketplace add <url>
    - claude plugin install <name>@<marketplace>

    Returns:
        Marketplace catalog with list of available plugins and their git sources.

    Example:
        ```bash
        claude plugin marketplace add http://localhost:4000/claude-code/marketplace.json
        claude plugin install my-plugin@litellm
        ```
    """
    try:
        prisma_client: Final = await _get_prisma_client()

        plugins: Final[Sequence[_PluginRecord]] = await ClaudeCodePluginRepository(prisma_client).table.find_many(
            where=published_skill_filter()
        )

        plugin_list: Final = []
        for plugin in plugins:
            try:
                manifest: Mapping[str, object] = json.loads(plugin.manifest_json or "{}")
            except json.JSONDecodeError:
                verbose_proxy_logger.warning("Plugin %s has invalid manifest JSON, skipping", plugin.name)
                continue

            # Source must be specified for URL-based marketplaces
            if "source" not in manifest:
                verbose_proxy_logger.warning("Plugin %s has no source field, skipping", plugin.name)
                continue

            entry: _MarketplaceEntry = {
                "name": plugin.name,
                "source": manifest["source"],
            }

            if plugin.version:
                entry["version"] = plugin.version
            if plugin.description:
                entry["description"] = plugin.description
            if "author" in manifest:
                entry["author"] = manifest["author"]
            if "homepage" in manifest:
                entry["homepage"] = manifest["homepage"]
            if "keywords" in manifest:
                entry["keywords"] = manifest["keywords"]
            if "category" in manifest:
                entry["category"] = manifest["category"]

            plugin_list.append(entry)

        marketplace: Final = {
            "name": "litellm",
            "owner": {"name": "LiteLLM", "email": "support@litellm.ai"},
            "plugins": plugin_list,
        }

        return JSONResponse(content=marketplace)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error generating marketplace: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Failed to generate marketplace: {e}"},
        )


# Allowlist for git-subdir paths: one or more segments separated by '/'.
# Each segment must start with an alphanumeric character and contain only
# alphanumeric characters, dots, hyphens, and underscores.
# This implicitly blocks '..', leading '/', backslashes, and percent-encoded sequences.
_VALID_GIT_SUBDIR_PATH_RE: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)*$")


def _validate_plugin_source(source: Mapping[str, str]) -> None:
    """Validate plugin source format, raising HTTPException on invalid input."""
    source_type: Final = source.get("source")
    if source_type == "github":
        if "repo" not in source:
            raise HTTPException(
                status_code=400,
                detail={"error": "GitHub source must include 'repo' field (e.g., 'org/repo')"},
            )
    elif source_type == "url":
        if "url" not in source:
            raise HTTPException(
                status_code=400,
                detail={"error": "URL source must include 'url' field (e.g., 'https://github.com/org/repo.git')"},
            )
    elif source_type == "git-subdir":
        if not source.get("url"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "git-subdir source must include 'url' field (e.g., 'https://github.com/org/repo.git')"
                },
            )
        if not source.get("path"):
            raise HTTPException(
                status_code=400,
                detail={"error": "git-subdir source must include 'path' field (e.g., 'plugins/plugin-name')"},
            )
        if not _VALID_GIT_SUBDIR_PATH_RE.match(source["path"]):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "git-subdir 'path' must be a relative path of the form 'segment/segment' (alphanumeric, dots, hyphens, underscores only)"
                },
            )
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "source.source must be 'github', 'url', or 'git-subdir'"},
        )


def _build_plugin_manifest(name: str, spec: PluginSpec) -> Mapping[str, object]:
    """Build the stored manifest dict shared by plugin create and update."""
    dumped: Final[Mapping[str, object]] = spec.model_dump(exclude_none=True)
    return {"name": name, **{key: value for key, value in dumped.items() if value and key != "name"}}


def _error_response(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": message})


def _name_conflict_error(name: str) -> HTTPException:
    return _error_response(
        409, f"A skill named '{name}' already exists. Update the existing skill instead of adding it again."
    )


@router.post(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RegisterPluginResponse,
)
async def register_plugin(
    request: RegisterPluginRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Register a new plugin in the LiteLLM marketplace.

    LiteLLM acts as a registry/discovery layer. Plugins are hosted on
    GitHub/GitLab/Bitbucket. Claude Code will clone from the git source
    when users install.

    This endpoint is create-only and never overwrites. If a plugin with
    the same name already exists it returns 409 Conflict; use
    PUT /claude-code/plugins/{plugin_name} to update an existing plugin.

    Callers that are not proxy admins are self-service submitters: the skill
    is stored with approval_status=pending_review and stays disabled until an
    admin approves it via POST /claude-code/plugins/{plugin_name}/approve.

    Parameters:
        - name: Plugin name (kebab-case)
        - source: Git source reference (github, url, or git-subdir format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)

    Returns:
        Registration status ("created" for admins, "submitted_for_review" otherwise)
        and plugin information.

    Example:
        ```bash
        curl -X POST http://localhost:4000/claude-code/plugins \\
          -H "Authorization: Bearer sk-..." \\
          -H "Content-Type: application/json" \\
          -d '{
            "name": "my-plugin",
            "source": {"source": "github", "repo": "org/my-plugin"},
            "version": "1.0.0",
            "description": "My awesome plugin"
          }'
        ```
    """
    from prisma.errors import UniqueViolationError

    try:
        prisma_client: Final = await _get_prisma_client()

        if not re.match(r"^[a-z0-9-]+$", request.name):
            raise HTTPException(
                status_code=400,
                detail={"error": "Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"},
            )

        _validate_plugin_source(request.source)

        existing: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": request.name}
        )
        if existing:
            raise _name_conflict_error(request.name)

        manifest: Final[Mapping[str, object]] = _build_plugin_manifest(request.name, request)
        submitted_for_review: Final = not is_proxy_admin(user_api_key_dict)
        approval_status: Final[SkillApprovalStatus] = SKILL_PENDING_REVIEW if submitted_for_review else SKILL_ACTIVE
        owner_scope: Final[str | None] = user_api_key_dict.user_id or get_primary_resource_owner_scope(
            user_api_key_dict
        )
        if submitted_for_review and owner_scope is None:
            raise _error_response(
                403,
                "Cannot submit a skill for review without an identity to attribute it to. "
                "Use a key that carries a user, team, or organization.",
            )

        try:
            plugin: Final[_PluginRecord] = await ClaudeCodePluginRepository(prisma_client).table.create(
                data={
                    "name": request.name,
                    "version": request.version,
                    "description": request.description,
                    "manifest_json": json.dumps(manifest),
                    "files_json": "{}",
                    "enabled": not submitted_for_review,
                    "approval_status": approval_status,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "created_by": owner_scope,
                }
            )
        except UniqueViolationError:
            raise _name_conflict_error(request.name)

        verbose_proxy_logger.info("Plugin %s created with approval_status=%s", request.name, approval_status)

        return RegisterPluginResponse(
            status="success",
            action="submitted_for_review" if submitted_for_review else "created",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                source=request.source,
                enabled=plugin.enabled,
                approval_status=approval_status,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error registering plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": f"Registration failed: {e}"},
        )


@router.get(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ListPluginsResponse,
)
async def list_plugins(
    enabled_only: bool = False,
    approval_status: SkillApprovalStatus | None = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    List plugins in the marketplace.

    Admins see every skill, including submissions awaiting review. Everyone
    else sees approved skills plus their own submissions, so a submitter can
    track the status of what they sent in.

    Parameters:
        - enabled_only: If true, only return enabled plugins
        - approval_status: Filter to one approval state, e.g. `pending_review` for the admin review queue

    Returns:
        List of plugins with their metadata and review state.
    """
    try:
        prisma_client: Final = await _get_prisma_client()

        plugins: Final[Sequence[_PluginRecord]] = await ClaudeCodePluginRepository(prisma_client).table.find_many(
            where=_list_plugins_filter(
                enabled_only=enabled_only,
                approval_status=approval_status,
                user_api_key_dict=user_api_key_dict,
            )
        )

        plugin_list: Final = []
        for p in plugins:
            # Parse manifest to get additional fields
            manifest = json.loads(p.manifest_json) if p.manifest_json else {}

            plugin_list.append(
                PluginListItem(
                    id=p.id,
                    name=p.name,
                    version=p.version,
                    description=p.description,
                    source=manifest.get("source", {}),
                    author=manifest.get("author"),
                    homepage=manifest.get("homepage"),
                    keywords=manifest.get("keywords"),
                    category=manifest.get("category"),
                    domain=manifest.get("domain"),
                    namespace=manifest.get("namespace"),
                    enabled=p.enabled,
                    approval_status=_as_approval_status(p.approval_status),
                    manifest_fingerprint=_manifest_fingerprint(p.manifest_json),
                    review_notes=p.review_notes,
                    reviewed_by=p.reviewed_by,
                    reviewed_at=p.reviewed_at.isoformat() if p.reviewed_at else None,
                    created_by=p.created_by,
                    created_at=p.created_at.isoformat() if p.created_at else None,
                    updated_at=p.updated_at.isoformat() if p.updated_at else None,
                )
            )

        # Sort by created_at descending (newest first)
        plugin_list.sort(key=lambda x: x.created_at or "", reverse=True)

        return ListPluginsResponse(
            plugins=plugin_list,
            count=len(plugin_list),
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error listing plugins: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.get(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get details of a specific plugin.

    Parameters:
        - plugin_name: The name of the plugin

    Returns:
        Plugin details including source and metadata.
    """
    try:
        prisma_client: Final = await _get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )

        if not plugin or not _caller_can_see(plugin, user_api_key_dict):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        manifest: Final[Mapping[str, object]] = json.loads(plugin.manifest_json or "{}") if plugin.manifest_json else {}

        return {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "description": plugin.description,
            "source": manifest.get("source"),
            "author": manifest.get("author"),
            "homepage": manifest.get("homepage"),
            "keywords": manifest.get("keywords"),
            "category": manifest.get("category"),
            "enabled": plugin.enabled,
            "approval_status": _as_approval_status(plugin.approval_status),
            "manifest_fingerprint": _manifest_fingerprint(plugin.manifest_json),
            "review_notes": plugin.review_notes,
            "reviewed_by": plugin.reviewed_by,
            "reviewed_at": plugin.reviewed_at.isoformat() if plugin.reviewed_at else None,
            "created_at": plugin.created_at.isoformat() if plugin.created_at else None,
            "updated_at": plugin.updated_at.isoformat() if plugin.updated_at else None,
            "created_by": plugin.created_by,
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.put(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RegisterPluginResponse,
)
async def update_plugin(
    plugin_name: str,
    request: UpdatePluginRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Update an existing plugin in the LiteLLM marketplace.

    The plugin is identified by its name in the path, which is the resource
    identity and cannot be changed here. This is a full replace, not a merge:
    the manifest is rebuilt from the request body, so any optional field left
    out is reset to its default (e.g. an omitted version is cleared, not kept).
    Send the full desired state.

    Returns 404 if no plugin with the given name exists, and the same 404 for
    a skill the caller cannot see, so the status code never reveals that a
    pending or rejected submission is sitting under that name; use
    POST /claude-code/plugins to create a new plugin.

    Admins can update any skill and the review state is left untouched. A
    submitter can only update their own skill, and doing so sends it back to
    pending review, since the content an admin approved has changed.

    Parameters:
        - plugin_name: Name of the plugin to update (path parameter)
        - source: Git source reference (github, url, or git-subdir format)
        - version: Semantic version (optional)
        - description: Plugin description (optional)
        - author: Author information (optional)
        - homepage: Plugin homepage URL (optional)
        - keywords: Search keywords (optional)
        - category: Plugin category (optional)

    Returns:
        Update status (action is always "updated") and plugin information.

    Example:
        ```bash
        curl -X PUT http://localhost:4000/claude-code/plugins/my-plugin \\
          -H "Authorization: Bearer sk-..." \\
          -H "Content-Type: application/json" \\
          -d '{
            "source": {"source": "github", "repo": "org/my-plugin"},
            "version": "2.0.0",
            "description": "My awesome plugin"
          }'
        ```
    """
    from prisma.errors import PrismaError

    try:
        prisma_client: Final = await _get_prisma_client()

        _validate_plugin_source(request.source)

        existing: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}  # mutable-ok: prisma query arguments must be plain dicts
        )
        if not existing or not _caller_can_see(existing, user_api_key_dict):
            raise _error_response(404, f"Plugin '{plugin_name}' not found")

        is_admin: Final = is_proxy_admin(user_api_key_dict)
        if not is_admin and existing.created_by not in get_resource_owner_scopes(user_api_key_dict):
            raise _error_response(403, "Only proxy admins or the submitter can update this skill")

        manifest: Final[Mapping[str, object]] = _build_plugin_manifest(plugin_name, request)

        plugin: Final[_PluginRecord] = await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},  # mutable-ok: prisma query arguments must be plain dicts
            data={  # mutable-ok: prisma query arguments must be plain dicts
                "version": request.version,
                "description": request.description,
                "manifest_json": json.dumps(manifest),
                "files_json": "{}",
                "updated_at": datetime.now(timezone.utc),
                **_submitter_edit_resets_review(is_admin=is_admin),
            },
        )

        verbose_proxy_logger.info("Plugin %s updated successfully", plugin_name)

        return RegisterPluginResponse(
            status="success",
            action="updated",
            plugin=PluginResponse(
                id=plugin.id,
                name=plugin.name,
                version=plugin.version,
                description=plugin.description,
                source=request.source,
                enabled=plugin.enabled,
                approval_status=_as_approval_status(plugin.approval_status),
            ),
        )

    except HTTPException:
        raise
    except PrismaError as e:
        verbose_proxy_logger.exception("Error updating plugin: %s", e)
        raise _error_response(500, f"Update failed: {e}")


def _stale_review_error(plugin_name: str) -> HTTPException:
    return _error_response(
        409,
        f"Skill '{plugin_name}' is no longer the submission that was reviewed. "
        "Read it again and review the current content.",
    )


async def _record_review(
    *,
    plugin_name: str,
    approval_status: SkillApprovalStatus,
    review_notes: str | None,
    reviewed_fingerprint: str | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> ReviewPluginResponse:
    if not is_proxy_admin(user_api_key_dict):
        raise _error_response(403, "Admin access required to review submitted skills")

    prisma_client: Final = await _get_prisma_client()
    repository: Final = ClaudeCodePluginRepository(prisma_client)

    existing: Final[_PluginRecord | None] = await repository.table.find_unique(
        where={"name": plugin_name}  # mutable-ok: prisma query arguments must be plain dicts
    )
    if not existing:
        raise _error_response(404, f"Plugin '{plugin_name}' not found")

    if _as_approval_status(existing.approval_status) == approval_status:
        raise _error_response(400, f"Skill '{plugin_name}' is already {approval_status}")

    if approval_status == SKILL_REJECTED and _as_approval_status(existing.approval_status) == SKILL_ACTIVE:
        raise _error_response(
            400,
            f"Skill '{plugin_name}' is already approved. Disable it to unpublish it. "
            "Rejecting applies to a skill awaiting review.",
        )

    publishes: Final = approval_status == SKILL_ACTIVE
    if publishes and _manifest_fingerprint(existing.manifest_json) != reviewed_fingerprint:
        raise _stale_review_error(plugin_name)

    reviewed_at: Final = datetime.now(timezone.utc)
    reviewed_rows: Final[int] = await repository.table.update_many(
        where={  # mutable-ok: prisma query arguments must be plain dicts
            "name": plugin_name,
            **({"manifest_json": existing.manifest_json} if publishes else {}),
        },
        data={  # mutable-ok: prisma query arguments must be plain dicts
            "approval_status": approval_status,
            "review_notes": review_notes,
            "reviewed_by": user_api_key_dict.user_id,
            "reviewed_at": reviewed_at,
            "enabled": publishes,
            "updated_at": reviewed_at,
        },
    )
    if reviewed_rows == 0:
        raise _stale_review_error(plugin_name)

    verbose_proxy_logger.info("Plugin %s reviewed: approval_status=%s", plugin_name, approval_status)

    return ReviewPluginResponse(
        status="success",
        name=existing.name,
        approval_status=approval_status,
        enabled=publishes,
        reviewed_by=user_api_key_dict.user_id,
        reviewed_at=reviewed_at.isoformat(),
        review_notes=review_notes,
    )


@router.post(
    "/claude-code/plugins/{plugin_name}/approve",
    tags=["Claude Code Marketplace"],  # mutable-ok: FastAPI route decorators take lists
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route decorators take lists
    response_model=ReviewPluginResponse,
)
async def approve_plugin(
    plugin_name: str,
    request: ApprovePluginRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Approve a submitted skill (admin only).

    Approving sets approval_status=active and publishes the skill to
    marketplace.json and the public Skill Hub.

    reviewed_fingerprint is the manifest_fingerprint returned by
    GET /claude-code/plugins and GET /claude-code/plugins/{plugin_name}. It
    ties the approval to the content that was read, so a submitter cannot get
    an edit published by landing it between the read and the approval.

    Example:
        ```bash
        FP=$(curl -s http://localhost:4000/claude-code/plugins/my-skill \\
          -H "Authorization: Bearer sk-admin-..." | jq -r .manifest_fingerprint)
        curl -X POST http://localhost:4000/claude-code/plugins/my-skill/approve \\
          -H "Authorization: Bearer sk-admin-..." \\
          -H "Content-Type: application/json" \\
          -d "{\\"reviewed_fingerprint\\": \\"$FP\\"}"
        ```
    """
    return await _record_review(
        plugin_name=plugin_name,
        approval_status=SKILL_ACTIVE,
        review_notes=request.review_notes,
        reviewed_fingerprint=request.reviewed_fingerprint,
        user_api_key_dict=user_api_key_dict,
    )


@router.post(
    "/claude-code/plugins/{plugin_name}/reject",
    tags=["Claude Code Marketplace"],  # mutable-ok: FastAPI route decorators take lists
    dependencies=[Depends(user_api_key_auth)],  # mutable-ok: FastAPI route decorators take lists
    response_model=ReviewPluginResponse,
)
async def reject_plugin(
    plugin_name: str,
    request: RejectPluginRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
):
    """
    Reject a submitted skill (admin only).

    The row is kept unpublished so the submitter can read review_notes and fix the submission.
    An already-approved skill cannot be rejected, since that would hide it from everyone but its
    submitter rather than merely unpublishing it. Disable it instead.

    Example:
        ```bash
        curl -X POST http://localhost:4000/claude-code/plugins/my-skill/reject \\
          -H "Authorization: Bearer sk-admin-..." \\
          -H "Content-Type: application/json" \\
          -d '{"review_notes": "point the source at the skill folder"}'
        ```
    """
    return await _record_review(
        plugin_name=plugin_name,
        approval_status=SKILL_REJECTED,
        review_notes=request.review_notes,
        reviewed_fingerprint=None,
        user_api_key_dict=user_api_key_dict,
    )


@router.post(
    "/claude-code/plugins/{plugin_name}/enable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def enable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Enable a disabled plugin. Proxy admins only.

    A skill that has not been approved cannot be enabled here: approve it
    through POST /claude-code/plugins/{plugin_name}/approve instead, so the
    reviewer is recorded on the row.

    Parameters:
        - plugin_name: The name of the plugin to enable
    """
    if not is_proxy_admin(user_api_key_dict):
        raise _error_response(403, "Only proxy admins can publish skills")

    try:
        prisma_client: Final = await _get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        if _as_approval_status(plugin.approval_status) != SKILL_ACTIVE:
            raise _error_response(
                409,
                f"Skill '{plugin_name}' is awaiting review. Approve it via "
                f"POST /claude-code/plugins/{plugin_name}/approve",
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": True, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info("Plugin %s enabled", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' enabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error enabling plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.post(
    "/claude-code/plugins/{plugin_name}/disable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def disable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Disable a plugin without deleting it. Proxy admins only.

    Parameters:
        - plugin_name: The name of the plugin to disable
    """
    if not is_proxy_admin(user_api_key_dict):
        raise _error_response(403, "Only proxy admins can unpublish skills")

    try:
        prisma_client: Final = await _get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        await ClaudeCodePluginRepository(prisma_client).table.update(
            where={"name": plugin_name},
            data={"enabled": False, "updated_at": datetime.now(timezone.utc)},
        )

        verbose_proxy_logger.info("Plugin %s disabled", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' disabled"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error disabling plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.delete(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a plugin from the marketplace. Admins can delete any skill; a
    submitter can withdraw one they submitted.

    Parameters:
        - plugin_name: The name of the plugin to delete
    """
    try:
        prisma_client: Final = await _get_prisma_client()

        plugin: Final[_PluginRecord | None] = await ClaudeCodePluginRepository(prisma_client).table.find_unique(
            where={"name": plugin_name}
        )
        if not plugin or not _caller_can_see(plugin, user_api_key_dict):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Plugin '{plugin_name}' not found"},
            )

        if not is_proxy_admin(user_api_key_dict) and plugin.created_by not in get_resource_owner_scopes(
            user_api_key_dict
        ):
            raise _error_response(403, "Only proxy admins or the submitter can delete this skill")

        await ClaudeCodePluginRepository(prisma_client).table.delete(where={"name": plugin_name})

        verbose_proxy_logger.info("Plugin %s deleted", plugin_name)
        return {"status": "success", "message": f"Plugin '{plugin_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting plugin: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
