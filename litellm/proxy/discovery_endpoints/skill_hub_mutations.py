import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Final

from fastapi import Depends, HTTPException
from typing_extensions import assert_never

from litellm.models.plugin_upload import (
    PluginPublication,
    PluginPublicationBase,
    PluginPublicationError,
    PluginSnapshot,
    PluginUploadStore,
    PublishedPlugin,
)
from litellm.models.skill_submission import (
    SkillSubmission,
    SkillSubmissionStore,
    SubmissionConflict,
    SubmissionLimitExceeded,
    SubmissionMissing,
    SubmissionReviewed,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.anthropic_endpoints.claude_code_endpoints.claude_code_skill_access import resolve_skill_visibility
from litellm.proxy.auth.auth_exception_handler import DB_UNAVAILABLE_FALLBACK_USER_ID
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.repositories.plugin_publication import resolve_plugin_version
from litellm.types.proxy.auth.user_roles import LitellmUserRoles


@dataclass(frozen=True, slots=True)
class SkillMutationError:
    status: int
    message: str


def writer_error(caller: UserAPIKeyAuth) -> SkillMutationError | None:
    if is_proxy_admin(caller):
        return None
    if (
        not caller.user_id
        or caller.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID
        or caller.user_role in (None, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
    ):
        return SkillMutationError(403, "Submitting a skill requires a writable user identity.")
    return None


def submission_owner(caller: UserAPIKeyAuth) -> str | None | SkillMutationError:
    if is_proxy_admin(caller) or caller.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
        return None
    if not caller.user_id or caller.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID:
        return SkillMutationError(403, "Reading submissions requires a user identity.")
    return caller.user_id


def submission_editor(caller: UserAPIKeyAuth) -> str | SkillMutationError:
    error: Final = writer_error(caller)
    if error is not None:
        return error
    if not caller.user_id or caller.user_id == DB_UNAVAILABLE_FALLBACK_USER_ID:
        return SkillMutationError(403, "Editing submissions requires the original submitting user's identity.")
    return caller.user_id


async def archive_writer(caller: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> UserAPIKeyAuth:
    from litellm.proxy.proxy_server import proxy_logging_obj

    error: Final = writer_error(caller)
    if error is not None:
        raise HTTPException(error.status, error.message)
    limiter: Final = proxy_logging_obj.get_proxy_hook("parallel_request_limiter")
    if not isinstance(limiter, PROXY_MaxParallelRequestsHandler_v3):
        raise HTTPException(503, "Skills Hub publishing requires the default v3 rate limiter")
    key_identity: Final = caller.api_key or caller.token
    identity: Final = (
        f"user:{caller.user_id}" if caller.user_id else f"key:{key_identity}" if key_identity else "local-admin"
    )
    result: Final = await limiter.atomic_check_and_increment_by_n(
        descriptors=[
            {
                "key": "skill_hub_publish",
                "value": hashlib.sha256(identity.encode()).hexdigest(),
                "rate_limit": {"requests_per_unit": 10, "window_size": 60},
            }
        ],
        increments=[{"requests": 1}],
        parent_otel_span=caller.parent_otel_span,
    )
    if result["overall_code"] == "OVER_LIMIT":
        raise HTTPException(
            429, "Skills Hub allows 10 uploads or imports per minute per user", headers={"Retry-After": "60"}
        )
    return caller


async def write_target(
    name: str,
    updating: bool,
    uploads: PluginUploadStore | None,
    caller: UserAPIKeyAuth,
    expected_updated_at: datetime | None = None,
) -> PluginSnapshot | SkillMutationError | None:
    error: Final = writer_error(caller)
    if error is not None:
        return error
    if uploads is None:
        return SkillMutationError(503, "Skills Hub requires a connected database")
    existing: Final = await uploads.get_plugin(name)
    if updating and (existing is None or not (await resolve_skill_visibility(caller)).allows(existing)):
        return SkillMutationError(404, "Skill not found")
    if existing is not None and expected_updated_at is not None and existing.updated_at != expected_updated_at:
        return SkillMutationError(409, "The published skill changed. Reload it before submitting.")
    if not updating and existing is not None:
        return SkillMutationError(409, f"A skill named '{name}' already exists. Update the existing skill instead.")
    return existing


def publication_error(error: PluginPublicationError) -> SkillMutationError:
    return SkillMutationError({"invalid_version": 400, "conflict": 409, "not_found": 404}[error.kind], error.message)


async def write_skill(
    publication: PluginPublication,
    existing: PluginSnapshot | None,
    uploads: PluginUploadStore,
    submissions: SkillSubmissionStore | None,
    caller: UserAPIKeyAuth,
) -> PublishedPlugin | SkillSubmission | SkillMutationError:
    error: Final = writer_error(caller)
    if error is not None:
        return error
    validation: Final = resolve_plugin_version(
        publication.version, None if existing is None else existing.version, updating=existing is not None
    )
    if isinstance(validation, PluginPublicationError):
        return publication_error(validation)
    expected: Final = None if existing is None else PluginPublicationBase(existing.id, existing.updated_at)
    if is_proxy_admin(caller):
        published: Final = await uploads.publish(publication, expected)
        return publication_error(published) if isinstance(published, PluginPublicationError) else published
    if submissions is None or not caller.user_id:
        return SkillMutationError(503, "Skills Hub requires a connected database")
    submission: Final = await submissions.submit(publication, caller.user_id, expected)
    return (
        SkillMutationError(429, submission.message) if isinstance(submission, SubmissionLimitExceeded) else submission
    )


async def review_skill(
    submission_id: str,
    submissions: SkillSubmissionStore | None,
    caller: UserAPIKeyAuth,
    approve: bool,
    reason: str | None,
    publication_base_url: str | None = None,
    expected_revision: int | None = None,
) -> SubmissionReviewed | SkillMutationError:
    if not is_proxy_admin(caller):
        return SkillMutationError(403, "Only proxy admins may review skill submissions.")
    if submissions is None:
        return SkillMutationError(503, "Skills Hub requires a connected database")
    result: Final = await submissions.review(
        submission_id,
        caller.user_id or "proxy-admin",
        approve,
        reason,
        publication_base_url=publication_base_url,
        expected_revision=expected_revision,
    )
    match result:
        case SubmissionMissing():
            return SkillMutationError(404, "Skill submission not found")
        case SubmissionConflict(message=message):
            return SkillMutationError(409, message)
        case PluginPublicationError(kind="invalid_version") as error:
            return publication_error(error)
        case PluginPublicationError(message=message):
            return SkillMutationError(409, message)
        case SubmissionReviewed():
            return result
        case _:
            assert_never(result)
