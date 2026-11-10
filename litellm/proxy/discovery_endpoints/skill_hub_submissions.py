import asyncio
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from litellm.models.plugin_upload import PluginPublication
from litellm.models.skill_submission import SkillSubmission, SkillSubmissionStore, SubmissionStatus
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import resolve_skill_visibility
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import PluginUploadRepositoryDep, SkillSubmissionRepositoryDep
from litellm.proxy.discovery_endpoints.agent_skills_archive import build_agent_skill_archive
from litellm.proxy.discovery_endpoints.skill_hub_endpoints import (
    skill_metadata_response,
    skill_proposal_response,
    skill_publication_response,
)
from litellm.proxy.discovery_endpoints.skill_hub_mutations import SkillMutationError, review_skill, submission_owner
from litellm.types.proxy.skill_hub import (
    SkillHubRejection,
    SkillHubReviewResponse,
    SkillHubSubmission,
    SkillHubSubmissionDetail,
    SkillHubSubmissionList,
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


def _response(request: Request, submission: SkillSubmission) -> SkillHubSubmission:
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
        submissions=tuple(_response(request, row) for row in page.submissions),
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
    """Read an immutable proposal and its currently visible published skill."""
    owner: Final = _owner(caller)
    submission: Final = await _store(submissions).get(submission_id, owner)
    if submission is None:
        raise HTTPException(404, "Skill submission not found")
    if uploads is None:
        raise HTTPException(503, "Skills Hub requires a connected database")
    current: Final = await uploads.get_metadata(submission.plugin_name)
    visible: Final = current is not None and (owner is None or (await resolve_skill_visibility(caller)).allows(current))
    return SkillHubSubmissionDetail(
        submission=_response(request, submission),
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
) -> SkillHubReviewResponse:
    from litellm.proxy.utils import get_custom_url

    result: Final = await review_skill(
        submission_id, submissions, caller, approve, reason, get_custom_url(str(request.base_url))
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
        submission=_response(request, result.submission),
        skill=None if result.plugin is None else skill_publication_response(request, publication, result.plugin),
    )


@router.put("/{submission_id}/approve", response_model=SkillHubReviewResponse)
async def approve_submission(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SkillHubReviewResponse:
    """Publish a pending proposal and record its approval atomically. Only proxy admins may review."""
    return await _review(request, submission_id, submissions, caller, True, None)


@router.put("/{submission_id}/reject", response_model=SkillHubReviewResponse)
async def reject_submission(
    request: Request,
    submission_id: str,
    submissions: SkillSubmissionRepositoryDep,
    caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    body: SkillHubRejection | None = None,
) -> SkillHubReviewResponse:
    """Reject a pending proposal without changing the published skill. Only proxy admins may review."""
    return await _review(request, submission_id, submissions, caller, False, None if body is None else body.reason)
