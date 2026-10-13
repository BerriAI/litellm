"""Kubernetes-style impersonation for service keys: a key with the `impersonate` permission acts as a named user."""

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles
from litellm.proxy.auth.auth_checks import get_user_role
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

IMPERSONATE_USER_HEADER: Final = "X-LiteLLM-Impersonate-User"
IMPERSONATE_EMAIL_HEADER: Final = "X-LiteLLM-Impersonate-Email"
IMPERSONATE_ENCODING_HEADER: Final = "X-LiteLLM-Impersonate-Encoding"
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

    @field_validator("permissions", mode="before")
    @classmethod
    def _default_permissions(cls, value: object) -> object:
        return {} if value is None else value


class _SubjectGrants(BaseModel):
    """Typed view of the untyped list/dict columns on the user row."""

    model_config = ConfigDict(frozen=True, from_attributes=True, extra="ignore")

    models: tuple[str, ...] = ()
    model_max_budget: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None

    @field_validator("models", mode="before")
    @classmethod
    def _default_models(cls, value: object) -> object:
        return () if value is None else value

    @field_validator("model_max_budget", mode="before")
    @classmethod
    def _default_model_max_budget(cls, value: object) -> object:
        return None if value is None else value

    @field_validator("metadata", mode="before")
    @classmethod
    def _default_metadata(cls, value: object) -> object:
        return None if value is None else value


class UserLookup(Protocol):
    async def __call__(self, user_id: str) -> LiteLLM_UserTable | None: ...


def impersonation_headers(user_id: str, email: str | None) -> Mapping[str, str]:
    encoded: Final = any(
        bool(value) and (not value.isascii() or not value.isprintable()) for value in (user_id, email or "")
    )
    return MappingProxyType(
        {
            IMPERSONATE_USER_HEADER: b64encode(user_id.encode("utf-8")).decode("ascii") if encoded else user_id,
            **(
                {IMPERSONATE_EMAIL_HEADER: b64encode(email.encode("utf-8")).decode("ascii") if encoded else email}
                if email is not None
                else {}
            ),
            **({IMPERSONATE_ENCODING_HEADER: "base64-utf8"} if encoded else {}),
        }
    )


def _read_identity(value: str, encoding: str | None) -> str:
    if encoding is None:
        return value
    if encoding == "base64-utf8":
        try:
            return b64decode(value, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="Invalid impersonation identity encoding") from None
    raise HTTPException(status_code=400, detail="Unsupported impersonation identity encoding")


def read_impersonation_request(headers: Mapping[str, str]) -> ImpersonationRequest | None:
    user_id: Final = headers.get(IMPERSONATE_USER_HEADER, "").strip()
    email: Final = headers.get(IMPERSONATE_EMAIL_HEADER, "").strip()
    if not user_id:
        return None
    encoding: Final = headers.get(IMPERSONATE_ENCODING_HEADER)
    return ImpersonationRequest(_read_identity(user_id, encoding), _read_identity(email, encoding) or None)


def key_may_impersonate(permissions: Mapping[str, object]) -> bool:
    return permissions.get(IMPERSONATE_PERMISSION) is True


def actor_label(actor: UserAPIKeyAuth) -> str:
    return actor.key_alias or actor.token or "unknown-key"


async def impersonate(
    actor: UserAPIKeyAuth,
    request: ImpersonationRequest,
    lookup: UserLookup,
    *,
    actor_is_master_key: bool,
) -> UserAPIKeyAuth | ImpersonationError:
    if not actor_is_master_key and actor.team_blocked:
        return ImpersonationForbidden(f"team {actor.team_id!r} is blocked")
    if not (actor_is_master_key or key_may_impersonate(_KeyPermissions.model_validate(actor).permissions)):
        return ImpersonationForbidden(f"key {actor_label(actor)} lacks the {IMPERSONATE_PERMISSION!r} permission")
    subject: Final = await lookup(request.user_id)
    if subject is None or not _email_matches(request.email, subject.user_email):
        return ImpersonatedUserNotFound(request.user_id)
    grants: Final = _SubjectGrants.model_validate(subject)
    if grants.metadata is not None and grants.metadata.get("scim_active") is False:
        return ImpersonationForbidden(f"user {request.user_id!r} is deactivated via SCIM")
    return _as_subject(actor, subject, grants)


def _email_matches(claimed: str | None, actual: str | None) -> bool:
    return claimed is None or actual is None or claimed.casefold() == actual.casefold()


def _as_subject(actor: UserAPIKeyAuth, subject: LiteLLM_UserTable, grants: _SubjectGrants) -> UserAPIKeyAuth:
    """Only the key identity survives, while subject grants and budgets come from the user row."""
    return UserAPIKeyAuth(
        api_key=None,
        token=actor.token,
        key_alias=actor.key_alias,
        allowed_routes=None,
        user_id=subject.user_id,
        user_email=subject.user_email,
        user_role=get_user_role(subject) or LitellmUserRoles.INTERNAL_USER,
        models=list(grants.models),
        user_tpm_limit=subject.tpm_limit,
        user_rpm_limit=subject.rpm_limit,
        user_max_budget=subject.max_budget,
        user_spend=subject.spend,
        user_model_max_budget=grants.model_max_budget,
        parent_otel_span=actor.parent_otel_span,
    ).model_copy(update={"impersonated_by": actor_label(actor)})
