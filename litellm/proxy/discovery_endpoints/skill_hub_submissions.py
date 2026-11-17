import asyncio
from dataclasses import replace
from typing import Annotated, Final

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile

from litellm.models.plugin_upload import PluginPublication, PluginPublicationError, PluginUploadStore
from litellm.models.skill_submission import (
    SkillSubmission,
    SkillSubmissionStore,
    SubmissionConflict,
    SubmissionLimitExceeded,
    SubmissionMissing,
    SubmissionStatus,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import resolve_skill_visibility
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import PluginUploadRepositoryDep, SkillSubmissionRepositoryDep
from litellm.proxy.discovery_endpoints.agent_skills_archive import build_agent_skill_archive
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import (
    prepare_skill_publication,
    skill_metadata_response,
    skill_proposal_response,
    skill_publication_response,
)
from litellm.proxy.discovery_endpoints.skill_hub_mutations import (
    SkillMutationError,
    archive_writer,
    publication_error,
    review_skill,
    submission_editor,
    submission_owner,
    write_target,
)
from litellm.proxy.discovery_endpoints.skill_import import MAX_IMPORT_BYTES
from litellm.repositories.plugin_publication import resolve_plugin_version
from litellm.types.proxy.skill_hub import (
    SkillHubMetadata,
    SkillHubRejection,
    SkillHubReviewResponse,
    SkillHubSubmission,
    SkillHubSubmissionDetail,
    SkillHubSubmissionEdit,
    SkillHubSubmissionList,
    SkillHubSubmissionRevision,
)

router: Final = APIRouter(prefix="/skill-hub/submissions", tags=["Skills Hub"])


def _store(submissions: SkillSubmissionStore | None) -> SkillSubmissionStore:
    if submissions is None:
        raise HTTPException(503, "Skills Hub requires a connected database")
    return submissions


def _owner(caller: UserAPIKeyAuth) -> str | None:
    owner: Final = submission_owner(caller)
    if isinstance(owner, SkillMutationError):
        raise HTTPException(owner.status, owner.message)
    return owner


def _response(request: Request, submission: SkillSubmission, caller: UserAPIKeyAuth) -> SkillHubSubmission:
    return SkillHubSubmission(
        id=submission.id,
        skill_name=submission.plugin_name,
        operation=submission.operation,
        status=submission.status,
        proposal=skill_proposal_response(request, submission),
        has_archive=submission.has_archive,
        submitted_by=submission.submitted_by,
        created_at=submission.created_at,
        base_skill_id=submission.base_plugin_id,
        base_updated_at=submission.base_updated_at,
        reviewed_by=submission.reviewed_by,
        reviewed_at=submission.reviewed_at,
        rejection_reason=submission.rejection_reason,
        revision=submission.revision,
        can_edit=submission.status == "pending_review" and submission_editor(caller) == submission.submitted_by,
    )


@router.get("", response_model=SkillHubSubmissionList)
async def list_submissions(
    request: Request,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    status: SubmissionStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SkillHubSubmissionList:
    """Read a page of owned proposals. Proxy admins and admin viewers can read every owner's proposals."""
    owner: Final = _owner(caller)
    page: Final = await _store(submissions).list(owner, status, limit, offset)
    return SkillHubSubmissionList(
        submissions=tuple(_response(request, row, caller) for row in page.submissions),
        count=len(page.submissions),
        has_more=page.has_more,
    )


@router.get("/{submission_id}", response_model=SkillHubSubmissionDetail)
async def get_submission(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubSubmissionDetail:
    """Read a proposal revision and its currently visible published skill."""
    owner: Final = _owner(caller)
    submission: Final = await _store(submissions).get(submission_id, owner)
    if submission is None:
        raise HTTPException(404, "Skill submission not found")
    if uploads is None:
        raise HTTPException(503, "Skills Hub requires a connected database")
    current: Final = await uploads.get_metadata(submission.plugin_name)
    visible: Final = current is not None and (owner is None or (await resolve_skill_visibility(caller)).allows(current))
    return SkillHubSubmissionDetail(
        submission=_response(request, submission, caller),
        current_skill=skill_metadata_response(request, current) if current is not None and visible else None,
    )


@router.get(
    "/{submission_id}/archive",
    response_class=Response,
    responses={200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}},
)
async def download_submission(
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Download an owned proposal as a private portable ZIP. Legacy nonportable archives return 404."""
    owner: Final = _owner(caller)
    store: Final = _store(submissions)
    submission: Final = await store.get(submission_id, owner)
    if submission is None:
        raise HTTPException(404, "Skill submission not found")
    content: Final = await store.get_archive(submission_id, owner)
    archive: Final = (
        await asyncio.to_thread(build_agent_skill_archive, content, plugin_name=submission.plugin_name)
        if content is not None
        else None
    )
    if archive is None:
        raise HTTPException(404, "This submission has no portable Agent Skills archive")
    return Response(
        archive.content,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{submission.plugin_name}.zip"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "ETag": f'"{archive.digest}"',
        },
    )


async def _review(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionStore | None,
    caller: UserAPIKeyAuth,
    approve: bool,
    reason: str | None,
    expected_revision: int | None,
) -> SkillHubReviewResponse:
    from litellm.proxy.utils import get_custom_url

    result: Final = await review_skill(
        submission_id, submissions, caller, approve, reason, get_custom_url(str(request.base_url)), expected_revision
    )
    if isinstance(result, SkillMutationError):
        raise HTTPException(result.status, result.message)
    proposal: Final = result.submission.proposal
    publication: Final = PluginPublication(
        name=proposal.name,
        version=proposal.version,
        description=proposal.description,
        manifest={
            **proposal.manifest,
            **({"source": dict(result.published_source)} if result.published_source else {}),
        },
        created_by=result.submission.submitted_by,
    )
    return SkillHubReviewResponse(
        action="approved" if approve else "rejected",
        submission=_response(request, result.submission, caller),
        skill=None if result.plugin is None else skill_publication_response(request, publication, result.plugin),
    )


@router.put("/{submission_id}/approve", response_model=SkillHubReviewResponse)
async def approve_submission(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    body: SkillHubSubmissionRevision | None = None,
) -> SkillHubReviewResponse:
    """Publish a pending proposal and record its approval atomically. Only proxy admins may review."""
    return await _review(
        request, submission_id, submissions, caller, True, None, None if body is None else body.expected_revision
    )


@router.put("/{submission_id}/reject", response_model=SkillHubReviewResponse)
async def reject_submission(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    body: SkillHubRejection | None = None,
) -> SkillHubReviewResponse:
    """Reject a pending proposal without changing the published skill. Only proxy admins may review."""
    return await _review(
        request,
        submission_id,
        submissions,
        caller,
        False,
        None if body is None else body.reason,
        None if body is None else body.expected_revision,
    )


def _editor(caller: UserAPIKeyAuth) -> str:
    owner: Final = submission_editor(caller)
    if isinstance(owner, SkillMutationError):
        raise HTTPException(owner.status, owner.message)
    return owner


def _changed(
    result: SkillSubmission | SubmissionMissing | SubmissionConflict | SubmissionLimitExceeded,
) -> SkillSubmission:
    if isinstance(result, SubmissionMissing):
        raise HTTPException(404, "Skill submission not found")
    if isinstance(result, SubmissionConflict):
        raise HTTPException(409, result.message)
    if isinstance(result, SubmissionLimitExceeded):
        raise HTTPException(429, result.message)
    return result


async def _editable(
    submission_id: str,
    owner: str,
    store: SkillSubmissionStore,
    revision: int,
    version: str | None,
    uploads: PluginUploadStore | None,
    caller: UserAPIKeyAuth,
) -> SkillSubmission:
    submission: Final = await store.get(submission_id, owner)
    if submission is None:
        raise HTTPException(404, "Skill submission not found")
    if submission.status != "pending_review" or submission.revision != revision:
        raise HTTPException(409, "This submission changed or is no longer pending. Reload it before trying again.")
    existing: Final = await write_target(
        submission.plugin_name, submission.operation == "update", uploads, caller, submission.base_updated_at
    )
    if isinstance(existing, SkillMutationError):
        raise HTTPException(existing.status, existing.message)
    validation: Final = resolve_plugin_version(
        version, None if existing is None else existing.version, updating=existing is not None
    )
    if isinstance(validation, PluginPublicationError):
        error: Final = publication_error(validation)
        raise HTTPException(error.status, error.message)
    return submission


@router.put("/{submission_id}", response_model=SkillHubSubmission)
async def edit_submission(
    request: Request,
    submission_id: str,
    body: SkillHubSubmissionEdit,
    submissions: SkillSubmissionRepositoryDep,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubSubmission:
    """Replace a pending proposal's metadata, preserving files. Only its original submitting user may edit."""
    owner: Final = _editor(caller)
    store: Final = _store(submissions)
    submission: Final = await _editable(
        submission_id, owner, store, body.expected_revision, body.version, uploads, caller
    )
    publication: Final = PluginPublication(
        name=submission.plugin_name,
        version=body.version,
        description=body.description,
        manifest={
            **{
                key: value
                for key, value in submission.proposal.manifest.items()
                if key not in SkillHubMetadata.model_fields
            },
            **body.model_dump(exclude={"expected_revision"}, exclude_none=True),
        },
        created_by=owner,
    )
    result: Final = await store.edit(submission_id, owner, body.expected_revision, publication)
    return _response(request, _changed(result), caller)


@router.put("/{submission_id}/upload", response_model=SkillHubSubmission)
async def edit_submission_archive(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    uploads: PluginUploadRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(archive_writer)],
    file: Annotated[UploadFile, File(description="Replacement Agent Skills ZIP with the same declared skill name")],
    expected_revision: Annotated[int, Form(ge=0)],
    version: Annotated[str | None, Form()] = None,
) -> SkillHubSubmission:
    """Replace a pending proposal's files using the upload limits. Only its original submitting user may edit."""
    owner: Final = _editor(caller)
    store: Final = _store(submissions)
    submission: Final = await _editable(submission_id, owner, store, expected_revision, version, uploads, caller)
    content: Final = await file.read(MAX_IMPORT_BYTES + 1)
    if len(content) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "Skill ZIP must be 10 MiB or smaller")
    archive: Final = await asyncio.to_thread(build_agent_skill_archive, content)
    if archive is None or archive.declared_name != submission.plugin_name:
        raise HTTPException(422, "Replacement SKILL.md must declare the existing skill name and a description")
    prepared: Final = await prepare_skill_publication(request, archive, version, None, owner)
    publication: Final = replace(prepared, manifest={**submission.proposal.manifest, **prepared.manifest})
    result: Final = await store.edit(submission_id, owner, expected_revision, publication)
    return _response(request, _changed(result), caller)


@router.put("/{submission_id}/withdraw", response_model=SkillHubSubmission)
async def withdraw_submission(
    request: Request,
    submission_id: str,
    body: SkillHubSubmissionRevision,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubSubmission:
    """Withdraw a pending proposal without publishing it. Only its original submitting user may withdraw."""
    owner: Final = _editor(caller)
    result: Final = await _store(submissions).withdraw(submission_id, owner, body.expected_revision)
    return _response(request, _changed(result), caller)
