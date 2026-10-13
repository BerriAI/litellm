"""Kubernetes-style impersonation for service keys: a key with the `impersonate` permission acts as a named user."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
from litellm.proxy.auth.auth_checks import get_user_role
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

IMPERSONATE_USER_HEADER: Final = "X-LiteLLM-Impersonate-User"
IMPERSONATE_EMAIL_HEADER: Final = "X-LiteLLM-Impersonate-Email"
IMPERSONATE_PERMISSION: Final = "impersonate"


@dataclass(frozen=True, slots=True)
class ImpersonationRequest:
    user_id: str
    email: str | None


@dataclass(frozen=True, slots=True)
class ImpersonationForbidden:
    detail: str


@dataclass(frozen=True, slots=True)
class ImpersonatedUserNotFound:
    user_id: str


ImpersonationError: TypeAlias = ImpersonationForbidden | ImpersonatedUserNotFound


class _KeyPermissions(BaseModel):
    """Typed view of the key's `permissions` JSON column, read off the auth object by attribute."""

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="ignore")

    permissions: Mapping[str, object] = {}


class _ActorRoutes(BaseModel):
    """Typed view of the service key's route allowlist, kept as a ceiling on the impersonated request."""

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="ignore")

    allowed_routes: tuple[str, ...] | None = None


class _SubjectGrants(BaseModel):
    """Typed view of the untyped list/dict columns on the user row."""

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="ignore")

    models: tuple[str, ...] = ()
    model_max_budget: Mapping[str, object] | None = None


class UserLookup(Protocol):
    async def __call__(self, user_id: str) -> LiteLLM_UserTable | None: ...


def read_impersonation_request(headers: Mapping[str, str]) -> ImpersonationRequest | None:
    user_id: Final = headers.get(IMPERSONATE_USER_HEADER, "").strip()
    email: Final = headers.get(IMPERSONATE_EMAIL_HEADER, "").strip()
    if not user_id:
        return None
    return ImpersonationRequest(user_id=user_id, email=email or None)


def key_may_impersonate(permissions: Mapping[str, object]) -> bool:
    return permissions.get(IMPERSONATE_PERMISSION) is True


def actor_label(actor: UserAPIKeyAuth) -> str:
    return actor.key_alias or actor.token or "unknown-key"


async def impersonate(
    actor: UserAPIKeyAuth,
    request: ImpersonationRequest,
    lookup: UserLookup,
) -> UserAPIKeyAuth | ImpersonationError:
    if not key_may_impersonate(_KeyPermissions.model_validate(actor).permissions):
        return ImpersonationForbidden(f"key {actor_label(actor)} lacks the {IMPERSONATE_PERMISSION!r} permission")
    subject: Final = await lookup(request.user_id)
    if subject is None or not _email_matches(request.email, subject.user_email):
        return ImpersonatedUserNotFound(request.user_id)
    return _as_subject(actor, subject)


def _email_matches(claimed: str | None, actual: str | None) -> bool:
    return claimed is None or actual is None or claimed.casefold() == actual.casefold()


def _as_subject(actor: UserAPIKeyAuth, subject: LiteLLM_UserTable) -> UserAPIKeyAuth:
    """Only the key's identity and its route allowlist survive, the allowlist stays a ceiling on what the control
    plane may do here. Team, models and permissions come from the subject, so a user never inherits what the
    service key may do. Budgets stay off the key fields, the user budget check reads the
    subject's own row by ``user_id``, a shared service key must not pool every subject's spend."""
    grants: Final = _SubjectGrants.model_validate(subject)
    routes: Final = _ActorRoutes.model_validate(actor).allowed_routes
    return UserAPIKeyAuth(
        api_key=actor.api_key,
        token=actor.token,
        key_alias=actor.key_alias,
        allowed_routes=None if routes is None else list(routes),
        user_id=subject.user_id,
        user_email=subject.user_email,
        user_role=get_user_role(subject) or LitellmUserRoles.INTERNAL_USER,
        models=list(grants.models),
        user_tpm_limit=subject.tpm_limit,
        user_rpm_limit=subject.rpm_limit,
        user_model_max_budget=grants.model_max_budget,
        parent_otel_span=actor.parent_otel_span,
        impersonated_by=actor_label(actor),
    )
