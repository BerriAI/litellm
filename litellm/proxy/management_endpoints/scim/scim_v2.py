"""
✨ SCIM v2 Endpoints for LiteLLM Proxy using Internal User/Team Management

This is an enterprise feature and requires a premium license.
"""

import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, overload

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict, assert_never

import litellm
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    NewTeamRequest,
    NewUserRequest,
    NewUserResponse,
    ProxyErrorTypes,
    ProxyException,
    TeamMemberAddRequest,
    TeamMemberDeleteRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import _delete_cache_key_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import _safe_get_request_headers
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user
from litellm.proxy.management_endpoints.scim.scim_transformations import (
    ScimTransformations,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    new_team,
    team_member_add,
    team_member_delete,
)
from litellm.proxy.utils import (
    PrismaClient,
    _premium_user_check,
    handle_exception_on_proxy,
)
from litellm.repositories.table_repositories import (
    InvitationLinkRepository,
    OrganizationMembershipRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.proxy.management_endpoints.scim_v2 import *

if TYPE_CHECKING:
    from prisma.models import LiteLLM_VerificationToken as PrismaVerificationToken


class _UserTableClient(Protocol):
    async def find_first(self, where: Mapping[str, object]) -> LiteLLM_UserTable | None: ...

    async def find_unique(self, where: Mapping[str, object]) -> LiteLLM_UserTable | None: ...

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        skip: int | None = None,
        take: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> Sequence[LiteLLM_UserTable]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> LiteLLM_UserTable: ...

    async def delete(self, where: Mapping[str, object]) -> LiteLLM_UserTable | None: ...

    async def count(self, where: Mapping[str, object] | None = None) -> int: ...


class _TeamTableClient(Protocol):
    async def find_unique(self, where: Mapping[str, object]) -> LiteLLM_TeamTable | None: ...

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        skip: int | None = None,
        take: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> Sequence[LiteLLM_TeamTable]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> LiteLLM_TeamTable: ...

    async def delete(self, where: Mapping[str, object]) -> LiteLLM_TeamTable | None: ...

    async def count(self, where: Mapping[str, object] | None = None) -> int: ...


class _VerificationTokenTableClient(Protocol):
    async def find_many(self, where: Mapping[str, object] | None = None) -> "Sequence[PrismaVerificationToken]": ...

    async def update(
        self, where: Mapping[str, object], data: Mapping[str, object]
    ) -> "PrismaVerificationToken | None": ...


class _UserReferencingTableClient(Protocol):
    async def delete_many(self, where: Mapping[str, object]) -> int: ...


@overload
def _table(repository: UserRepository) -> _UserTableClient: ...


@overload
def _table(repository: TeamRepository) -> _TeamTableClient: ...


@overload
def _table(repository: VerificationTokenRepository) -> _VerificationTokenTableClient: ...


@overload
def _table(
    repository: InvitationLinkRepository | OrganizationMembershipRepository | TeamMembershipRepository,
) -> _UserReferencingTableClient: ...


def _table(
    repository: UserRepository
    | TeamRepository
    | VerificationTokenRepository
    | InvitationLinkRepository
    | OrganizationMembershipRepository
    | TeamMembershipRepository,
) -> object:
    return repository.table


class UserProvisionerHelpers:
    """Helper methods for user provisioning operations."""

    @staticmethod
    async def handle_existing_user_by_email(
        prisma_client: PrismaClient,
        new_user_request: NewUserRequest,
        admin_group: str | None = None,
    ) -> SCIMUser | None:
        """
        Check if a user with the given email already exists and update them if found.

        The matched row keeps its existing user_id even when the SCIM userName differs.
        Virtual keys, team rosters, team/organization memberships and spend logs all
        reference that id, so re-keying the user row would strand every one of them and
        make removals against rosters holding the old id no-op. SCIM ids are opaque to
        the client, which reads the stable id back from the response.

        When admin_group is configured the resolved global role on new_user_request
        is persisted too, so re-upserting an existing email demotes a user who is no
        longer in the admin group instead of leaving the stale role.

        IdPs like Entra manage membership exclusively through /Groups and never send
        ``groups`` on POST /Users, so a request without teams means "unspecified",
        not "remove from every team": existing memberships are preserved then.

        Args:
            prisma_client: Database client
            new_user_request: New user request data
            admin_group: Configured SCIM admin group, or None to leave role untouched

        Returns:
            SCIMUser if user was updated, None if no existing user found
        """
        if not new_user_request.user_email:
            return None

        existing_user: Final = await _table(UserRepository(prisma_client)).find_first(
            where={"user_email": new_user_request.user_email}
        )

        if not existing_user:
            return None

        requested_teams: Final = list(dict.fromkeys(new_user_request.teams or []))
        new_teams: Final = requested_teams if requested_teams else list(existing_user.teams or [])

        if new_user_request.user_id != existing_user.user_id:
            verbose_proxy_logger.info(
                "SCIM: email %s already provisioned as user_id=%s, keeping that id instead of re-keying to %s",
                new_user_request.user_email,
                existing_user.user_id,
                new_user_request.user_id,
            )

        await _handle_team_membership_changes(
            user_id=existing_user.user_id,
            existing_teams=existing_user.teams or [],
            new_teams=new_teams,
        )

        updated_user: Final = await _table(UserRepository(prisma_client)).update(
            where={"user_id": existing_user.user_id},
            data={
                "user_email": new_user_request.user_email,
                "user_alias": new_user_request.user_alias,
                "teams": new_teams,
                "metadata": safe_dumps(new_user_request.metadata),
                **({"user_role": new_user_request.user_role} if admin_group is not None else {}),
            },
        )

        return await ScimTransformations.transform_litellm_user_to_scim_user(updated_user)


class ScimUserData(TypedDict):
    """Typed structure for extracted SCIM user data."""

    user_email: str | None
    user_alias: str | None
    sso_user_id: str | None
    teams: list[str]
    given_name: str | None
    family_name: str | None
    active: bool | None
    enterprise: SCIMEnterpriseUser | None
    entitlements: list[SCIMMultiValuedAttribute] | None
    roles: list[SCIMMultiValuedAttribute] | None


class GroupMemberExtractionResult(BaseModel):
    """Result of extracting and processing group members.

    ``all_member_ids`` is deduped order-preserving; ``existing_member_ids`` is not,
    so a repeated resolved id appears once in the former and twice in the latter.
    """

    existing_member_ids: list[str]
    created_users: list[NewUserResponse]
    all_member_ids: list[str]  # existing + newly created


scim_router: Final = APIRouter(
    prefix="/scim/v2",
    tags=["✨ SCIM v2 (Enterprise Only)"],
    dependencies=[Depends(_premium_user_check)],
)


# Helper functions for common operations
async def _get_prisma_client_or_raise_exception():
    """Check if database is connected and raise HTTPException if not."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail={"error": "No database connected"})
    return prisma_client


async def _check_user_exists(user_id: str) -> LiteLLM_UserTable:
    """Check if user exists and return user, raise 404 if not found."""
    prisma_client: Final = await _get_prisma_client_or_raise_exception()

    user: Final = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": user_id})

    if not user:
        raise HTTPException(status_code=404, detail={"error": f"User not found with ID: {user_id}"})

    return user


async def _check_team_exists(team_id: str) -> LiteLLM_TeamTable:
    """Check if team exists and return team, raise 404 if not found."""
    prisma_client: Final = await _get_prisma_client_or_raise_exception()

    team: Final = await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id})

    if not team:
        raise HTTPException(status_code=404, detail={"error": f"Group not found with ID: {team_id}"})

    return team


def _extract_scim_user_data(user: SCIMUser) -> ScimUserData:
    """Extract common data from SCIMUser object."""
    user_email = None
    if user.emails and len(user.emails) > 0:
        user_email = user.emails[0].value

    user_alias = None
    if user.name and user.name.givenName:
        user_alias = user.name.givenName

    teams = []
    if user.groups:
        teams = [group.value for group in user.groups]

    return {
        "user_email": user_email,
        "user_alias": user_alias,
        "sso_user_id": user.externalId,
        "teams": teams,
        "given_name": user.name.givenName if user.name else None,
        "family_name": user.name.familyName if user.name else None,
        "active": user.active,
        "enterprise": user.enterprise_user,
        "entitlements": user.entitlements,
        "roles": user.roles,
    }


def _build_scim_metadata(
    given_name: str | None,
    family_name: str | None,
    active: bool | None = None,
    enterprise: SCIMEnterpriseUser | None = None,
    entitlements: list[SCIMMultiValuedAttribute] | None = None,
    roles: list[SCIMMultiValuedAttribute] | None = None,
) -> dict[str, object]:
    """Build metadata dictionary with SCIM data."""
    metadata: Final[dict[str, object]] = {
        "scim_metadata": LiteLLM_UserScimMetadata(
            givenName=given_name,
            familyName=family_name,
        ).model_dump()
    }

    if active is not None:
        metadata["scim_active"] = active

    if enterprise is not None:
        metadata[SCIM_ENTERPRISE_METADATA_KEY] = enterprise.model_dump(by_alias=True, exclude_none=True)

    if entitlements is not None:
        metadata[SCIM_ENTITLEMENTS_METADATA_KEY] = [e.model_dump(exclude_none=True) for e in entitlements]

    if roles is not None:
        metadata[SCIM_ROLES_METADATA_KEY] = [r.model_dump(exclude_none=True) for r in roles]

    return metadata


async def _get_scim_upsert_user_setting() -> bool:
    """
    Get the scim_upsert_user setting from litellm_settings.

    Returns:
        True if scim_upsert_user is not set or is True (default behavior),
        False if scim_upsert_user is explicitly set to False (SCIM 2.0 strict mode)
    """
    try:
        from litellm.proxy.proxy_server import proxy_config

        config: Final = await proxy_config.get_config()
        litellm_settings: Final = config.get("litellm_settings", {}) or {}
        scim_upsert_user: Final = litellm_settings.get("scim_upsert_user", True)

        # Default to True if not set (backward compatibility)
        return bool(scim_upsert_user)
    except Exception as e:
        verbose_proxy_logger.warning("Error reading scim_upsert_user setting, defaulting to True: %s", e)
        # Default to True for backward compatibility
        return True


ScimUserRole = Literal[
    LitellmUserRoles.PROXY_ADMIN,
    LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    LitellmUserRoles.INTERNAL_USER,
    LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
]


def _default_scim_user_role() -> ScimUserRole:
    """Non-admin default role for SCIM-provisioned users."""
    if litellm.default_internal_user_params:
        configured_role: Final = litellm.default_internal_user_params.get("user_role")
        if configured_role is not None:
            return configured_role
    return LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


async def _get_scim_admin_group() -> str | None:
    """
    Get the scim_admin_group setting from litellm_settings.

    Returns the configured admin group identifier, or None when unset so callers
    leave a user's global role untouched (default-safe).
    """
    try:
        from litellm.proxy.proxy_server import proxy_config

        config: Final = await proxy_config.get_config()
        litellm_settings: Final = config.get("litellm_settings", {}) or {}
        return litellm_settings.get("scim_admin_group") or None
    except Exception as e:
        verbose_proxy_logger.warning("Error reading scim_admin_group setting, defaulting to None: %s", e)
        return None


def _resolve_scim_user_role(
    groups: list[SCIMUserGroup],
    admin_group: str | None,
    default_role: ScimUserRole,
) -> LitellmUserRoles | None:
    """
    Resolve a user's global proxy role from their SCIM groups.

    Returns None when no admin group is configured, signalling callers to leave
    the role unchanged. Otherwise grants PROXY_ADMIN when any group matches the
    admin group by value or display, and falls back to the non-admin default.
    """
    if admin_group is None:
        return None
    for group in groups:
        if group.value == admin_group or group.display == admin_group:
            return LitellmUserRoles.PROXY_ADMIN
    return default_role


async def _scim_groups_from_team_ids(prisma_client: PrismaClient, team_ids: list[str]) -> list[SCIMUserGroup]:
    """
    Build SCIMUserGroup objects from team ids, populating display from each
    team's alias so admin-group matching by display name works the same way it
    does on PUT (where SCIM groups carry display names natively).
    """
    teams: Final = [
        await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id}) for team_id in team_ids
    ]
    return [
        SCIMUserGroup(
            value=team_id,
            display=team.team_alias if team is not None else None,
        )
        for team_id, team in zip(team_ids, teams)
    ]


async def _recompute_scim_member_roles(prisma_client: PrismaClient, user_ids: Iterable[str]) -> None:
    """
    Recompute and persist each user's global proxy role from their resulting team
    membership. No-op unless scim_admin_group is configured, so a SCIM group write
    that drops a member from the admin group demotes them just like the user
    endpoints do, and the role is left untouched when the feature is off.
    """
    admin_group: Final = await _get_scim_admin_group()
    if admin_group is None:
        return

    default_role: Final = _default_scim_user_role()
    for user_id in user_ids:
        user = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": user_id})
        if user is None:
            continue
        resolved_role = _resolve_scim_user_role(
            await _scim_groups_from_team_ids(prisma_client, user.teams or []),
            admin_group,
            default_role,
        )
        await _table(UserRepository(prisma_client)).update(
            where={"user_id": user_id},
            data={"user_role": resolved_role},
        )


class _ResolvedUserMember(NamedTuple):
    user_id: str


class _SkippedGroupMember(NamedTuple):
    value: str
    reason: Literal["nested_group", "non_user_type", "existing_team"]


class _UnknownMember(NamedTuple):
    value: str


class _AmbiguousMember(NamedTuple):
    value: str


_ClassifiedGroupMember = Union[_ResolvedUserMember, _SkippedGroupMember, _UnknownMember, _AmbiguousMember]


class _PartitionedMembers(NamedTuple):
    resolved_ids: tuple[str, ...]
    skipped: tuple[_SkippedGroupMember, ...]
    unknown_ids: tuple[str, ...]
    ambiguous_values: tuple[str, ...]


def _member_value(member: SCIMMember) -> str:
    """A member id is opaque to us but has to be there; an empty one is a client error."""
    if not member.value or not member.value.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid member: user ID cannot be empty."},
        )
    return member.value


def _normalized_member_type(member: SCIMMember) -> str | None:
    """The canonical ``type`` a member declares, lowercased; blank or absent means none."""
    normalized: Final = (member.type or "").strip().lower()
    return normalized or None


_JSON_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])


def _json_object_fields(raw: object) -> Mapping[str, object] | None:
    """A typed, read-only view of a JSON object, or None when it is not one."""
    try:
        return _JSON_OBJECT_ADAPTER.validate_python(raw)
    except ValidationError:
        return None


def _team_metadata_has_scim_provenance(team_metadata: object) -> bool:
    """Whether a group write from the identity provider left its mark on this team.

    ``SCIM_TEAM_DATA_METADATA_KEY`` counts because PUT has been writing it since
    long before the explicit marker, so a team the identity provider already
    syncs is recognized without waiting to be written again.
    """
    fields: Final = _json_object_fields(team_metadata)
    if fields is None:
        return False
    return bool(fields.get(SCIM_MANAGED_TEAM_METADATA_KEY)) or fields.get(SCIM_TEAM_DATA_METADATA_KEY) is not None


class _CaseInsensitiveMatch(TypedDict):
    equals: ReadOnly[str]
    mode: ReadOnly[str]


async def _users_named_by_member_value(
    value: str, prisma_client: PrismaClient, *, take: int | None = 2
) -> tuple[str, ...]:
    """Every user id this member value names, by SSO identity or by email.

    Both fields are searched in one pass, because searching either first would hide a
    value that names one account by its SSO identity and another by its email, and
    hand the group to whichever field was searched first.

    They are not compared alike. An email is matched the way ``new_user`` matches one
    before it accepts a new account, case-insensitively: matching more strictly than
    the layer that would reject the placeholder is what turned a member id whose
    casing differed from the stored email into a 500 on the whole push. An SSO
    identity is matched exactly, because OIDC defines ``sub`` as case-sensitive and
    nothing folds its case on the way in, so treating two subjects that differ in case
    as one would hand the group to an account the provider never named.

    ``take`` bounds the read for a caller that only needs to know whether the value
    names one account or several; ``user_email`` carries no index, so letting the scan
    stop early is worth the two rows. A caller that has to know *which* accounts, as a
    removal does, passes None. That set is the accounts sharing one identity, which is
    a handful at worst.
    """
    subject: Final = value.strip()
    email: Final[_CaseInsensitiveMatch] = {"equals": subject, "mode": "insensitive"}
    rows: Final = await _table(UserRepository(prisma_client)).find_many(
        # mutable-ok: the Prisma serializer requires concrete dicts and a concrete list
        where={"OR": [{"sso_user_id": subject}, {"user_email": email}]},
        take=take,
    )
    return tuple(dict.fromkeys(row.user_id for row in rows))


async def _classify_group_member(member: SCIMMember, prisma_client: PrismaClient) -> _ClassifiedGroupMember:
    """
    Decide what a single SCIM group member refers to.

    A LiteLLM team only holds users, so a member is dropped when it declares a type
    other than ``User`` or when its id names an existing team. Both of those checks
    are placed around the user lookup rather than before it, because the id of a
    real user is the one thing that outranks them:

    - ``"type": "Group"`` (what Entra sends for a nested group) is dropped without
      a lookup. This bug provisioned nested group GUIDs as users, so those rows
      exist in the wild and would otherwise resolve as members all over again.
    - any other unrecognized type is dropped only after the user lookup misses.
      Clients do send non-canonical types on real members (RFC 7643 defines
      ``direct`` for ``User.groups``), and dropping a live user over one would
      revoke that user's team access on the next full sync.
    - an id that names an existing team is dropped only when the member arrives
      untyped, which is how Okta sends nested groups, and only when that team is
      one the identity provider writes. An id the IdP called a User is a user
      even if some team happens to share the id, and a team created here rather
      than through SCIM is not evidence of anything about the member.

    When those checks miss on an otherwise user-shaped member, its value is looked
    up as an SSO identity or an email, and a match resolves to that user's
    ``user_id``. A value that names more than one account is ambiguous rather than
    unknown: it names a real person we cannot identify, so it is neither guessed at
    nor provisioned.

    An exact ``user_id`` hit is checked the same way rather than trusted outright. A
    value can be one account's id and another's SSO identity or email, and taking the
    id on sight would hand the group to whichever account happened to be keyed by it.
    The placeholders this bug provisioned are that shape exactly, since they are keyed
    by the very id the provider keeps pushing, so on a tenant that already has them
    the membership is refused and named rather than silently landing on the
    placeholder again.
    """
    value: Final = _member_value(member)
    member_type: Final = _normalized_member_type(member)

    if member_type == "group":
        return _SkippedGroupMember(value=value, reason="nested_group")

    user: Final = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": value})
    if user is not None:
        shared_with: Final = tuple(
            other for other in await _users_named_by_member_value(value, prisma_client) if other != value
        )
        if shared_with:
            verbose_proxy_logger.warning(
                "SCIM: group member '%s' is one account's user id and is also account '%s' by SSO identity or email, "
                "so the membership cannot be attributed. A placeholder an earlier release provisioned under this id "
                "looks exactly like this and should be deleted so the real account can be matched",
                value,
                shared_with[0],
            )
            return _AmbiguousMember(value=value)
        return _ResolvedUserMember(user_id=value)

    if member_type is not None and member_type != "user":
        return _SkippedGroupMember(value=value, reason="non_user_type")

    if member_type is None:
        team: Final = await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": value})
        if team is not None and _team_metadata_has_scim_provenance(team.metadata):
            return _SkippedGroupMember(value=value, reason="existing_team")

    named: Final = await _users_named_by_member_value(value, prisma_client)
    if len(named) == 1:
        verbose_proxy_logger.info(
            "SCIM: group member '%s' matched user_id '%s' by SSO identity or email",
            value,
            named[0],
        )
        return _ResolvedUserMember(user_id=named[0])
    if len(named) > 1:
        verbose_proxy_logger.warning(
            "SCIM: group member '%s' names more than one account by SSO identity or email and cannot be resolved "
            "unambiguously",
            value,
        )
        return _AmbiguousMember(value=value)

    return _UnknownMember(value=value)


def _bucketed_member(entry: _ClassifiedGroupMember) -> _PartitionedMembers:
    """The single-member partition one classified entry contributes."""
    match entry:
        case _ResolvedUserMember(user_id=user_id):
            return _PartitionedMembers(resolved_ids=(user_id,), skipped=(), unknown_ids=(), ambiguous_values=())
        case _SkippedGroupMember():
            return _PartitionedMembers(resolved_ids=(), skipped=(entry,), unknown_ids=(), ambiguous_values=())
        case _UnknownMember(value=value):
            return _PartitionedMembers(resolved_ids=(), skipped=(), unknown_ids=(value,), ambiguous_values=())
        case _AmbiguousMember(value=value):
            return _PartitionedMembers(resolved_ids=(), skipped=(), unknown_ids=(), ambiguous_values=(value,))
        case _:
            assert_never(entry)


def _partition_classified_members(classified: Iterable[_ClassifiedGroupMember]) -> _PartitionedMembers:
    """Split classified members into the buckets the resolver acts on, keeping request order."""
    bucketed: Final = tuple(_bucketed_member(entry) for entry in classified)
    return _PartitionedMembers(
        resolved_ids=tuple(chain.from_iterable(bucket.resolved_ids for bucket in bucketed)),
        skipped=tuple(chain.from_iterable(bucket.skipped for bucket in bucketed)),
        unknown_ids=tuple(chain.from_iterable(bucket.unknown_ids for bucket in bucketed)),
        ambiguous_values=tuple(chain.from_iterable(bucket.ambiguous_values for bucket in bucketed)),
    )


def _admitted_member_id(entry: _ClassifiedGroupMember, created_ids: frozenset[str]) -> str | None:
    match entry:
        case _ResolvedUserMember(user_id=user_id):
            return user_id
        case _UnknownMember(value=value):
            return value if value in created_ids else None
        case _SkippedGroupMember() | _AmbiguousMember():
            return None
        case _:
            assert_never(entry)


def _admitted_member_ids(classified: Iterable[_ClassifiedGroupMember], created_ids: frozenset[str]) -> tuple[str, ...]:
    """Member ids that survive resolution, in the order the request listed them.

    An id the request repeats is one member: the roster these ids are written to
    holds one row per member, and a second creation attempt for the same id fails
    against the real unique constraint even though the first one succeeded.
    """
    return tuple(
        dict.fromkeys(
            member_id for entry in classified if (member_id := _admitted_member_id(entry, created_ids)) is not None
        )
    )


class _UserIdWhere(TypedDict):
    user_id: ReadOnly[str]


class _ScimErrorDetail(TypedDict):
    error: ReadOnly[str]


async def _ensure_group_member_user(
    user_id: str,
    created_via: str,
    prisma_client: PrismaClient,
) -> NewUserResponse | None:
    """The created user, or None when the id already resolves to a user row (a
    concurrent provisioning request won the creation race after our lookup missed).

    Raises:
        HTTPException: 500 when the user can neither be created nor found. The
        request has to fail so the identity provider retries, instead of recording
        success for a member the roster silently dropped.
    """
    created: Final = await _create_user_if_not_exists(user_id=user_id, created_via=created_via)
    if created is not None:
        return created
    where: Final[_UserIdWhere] = {"user_id": user_id}
    existing: Final = await _table(UserRepository(prisma_client)).find_unique(where=where)
    if existing is not None:
        return None
    detail: Final[_ScimErrorDetail] = {
        "error": f"Failed to create user '{user_id}' while provisioning group membership."
    }
    raise HTTPException(status_code=500, detail=detail)


def _roster_entries_named_by(value: str, roster: frozenset[str], resolved: tuple[str, ...]) -> tuple[str, ...]:
    """The members of this group a removal value names.

    Both ways of naming one count together. The id as written counts when the roster
    holds it verbatim, which is how an earlier release recorded a member it could not
    match, and the accounts it resolves to count when they are on the roster. Counting
    only the resolved ones would let a value that is one member's canonical id and
    another member's email revoke both, since each looks singular on its own.
    """
    return tuple(
        dict.fromkeys(
            chain(
                (value,) if value in roster else (),
                (user_id for user_id in resolved if user_id in roster),
            )
        )
    )


async def _member_ids_to_drop(
    members: Sequence[SCIMMember], roster: frozenset[str], prisma_client: PrismaClient
) -> frozenset[str]:
    """The members a ``remove`` clears, one per id the request names.

    The roster holds canonical user ids, so a directory that added someone by their
    email or SSO identity has to be able to remove them by that same value, and a
    member an earlier release recorded under the raw id has to stay removable by it.

    Ambiguity is a property of the table as it stands, not of the value, so a value
    that named one person when they were admitted can name two later. Resolving a
    removal against the whole table would then drop nobody while answering 200, and
    the person the directory just took out of the group would keep the team. So a
    removal keeps only the accounts already on the roster: one is unambiguous however
    many strangers share the address, none means there is nothing to revoke, and only
    a value naming two of this group's own members is genuinely undecidable. That last
    case fails rather than reporting a removal it did not perform, or revoking both.

    Raises:
        HTTPException: 400 when a member id names more than one current member.
    """
    written: Final = frozenset(_member_value(member) for member in members)
    matched: Final = tuple(
        [
            (
                value,
                _roster_entries_named_by(
                    value, roster, await _users_named_by_member_value(value, prisma_client, take=None)
                ),
            )
            for value in sorted(written)
        ]
    )
    undecidable: Final = tuple(value for value, entries in matched if len(entries) > 1)
    if undecidable:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Member ID '{undecidable[0]}' names more than one member of this group, so the removal "
                "cannot be attributed. Send the LiteLLM user ID as the member value, or resolve the duplicate."
            },
        )
    return frozenset(chain.from_iterable(entries for _, entries in matched))


async def _resolve_group_member_ids(
    members: Sequence[SCIMMember],
    created_via: str,
    prisma_client: PrismaClient,
) -> GroupMemberExtractionResult:
    """
    Resolve SCIM group members to LiteLLM user ids, dropping members that are not users.

    Member ids are matched by ``user_id`` first, then by SSO identity or email. An
    id that resolves to nothing is created when litellm_settings.scim_upsert_user is
    True (default) and rejected per SCIM 2.0 otherwise. Removals do not come through
    here: they resolve through ``_member_ids_to_drop`` instead, which neither creates
    a user nor fails on an id it cannot place.

    Raises:
        HTTPException: 400 when a member id is empty, when a member id names more
        than one user, or when scim_upsert_user is False and a member id is neither
        an existing user, an existing team, nor a member declared to be something
        other than a user. 500 when a member's user row can neither be created nor
        found.
    """
    classified: Final = tuple([await _classify_group_member(member, prisma_client) for member in members])
    partition: Final = _partition_classified_members(classified)

    for skipped in partition.skipped:
        verbose_proxy_logger.info(
            "SCIM: ignoring non-user group member '%s' (%s); LiteLLM teams contain users only",
            skipped.value,
            skipped.reason,
        )

    if partition.ambiguous_values:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Member ID '{partition.ambiguous_values[0]}' names more than one LiteLLM user, so the "
                "group membership cannot be attributed. Resolve the duplicate, which for an id that also matches a "
                "SCIM-provisioned placeholder means deleting that placeholder."
            },
        )

    if partition.unknown_ids and not await _get_scim_upsert_user_setting():
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"User with ID '{partition.unknown_ids[0]}' does not exist. "
                "Please create the user first via POST /Users before adding to group."
            },
        )

    unique_unknown_ids: Final = tuple(dict.fromkeys(partition.unknown_ids))
    for user_id in unique_unknown_ids:
        verbose_proxy_logger.warning(
            "SCIM: creating placeholder user for group member '%s'; matched no user by user_id, sso_user_id or "
            "user_email. An SSO-provisioned user's real account stays teamless if this is a mismatch",
            user_id,
        )

    creations: Final = tuple(
        [
            (
                user_id,
                await _ensure_group_member_user(user_id=user_id, created_via=created_via, prisma_client=prisma_client),
            )
            for user_id in unique_unknown_ids
        ]
    )
    created_users: Final = tuple(created for _, created in creations if created is not None)

    return GroupMemberExtractionResult(
        existing_member_ids=partition.resolved_ids,
        created_users=created_users,
        all_member_ids=_admitted_member_ids(classified, frozenset(unique_unknown_ids)),
    )


async def _extract_group_member_ids(group: SCIMGroup) -> GroupMemberExtractionResult:
    """
    Extract member IDs from SCIMGroup, validating that all users exist.

    Behavior depends on litellm_settings.scim_upsert_user:
    - If True (default): Creates users that don't exist (backward compatible)
    - If False: Rejects non-existent users per SCIM 2.0 protocol

    Returns:
        GroupMemberExtractionResult with existing members, created users, and all member IDs

    Raises:
        HTTPException: If scim_upsert_user is False and any member user does not exist (400 Bad Request)
    """
    prisma_client: Final = await _get_prisma_client_or_raise_exception()
    return await _resolve_group_member_ids(
        members=group.members or [],
        created_via="scim_group_membership",
        prisma_client=prisma_client,
    )


async def _get_team_members_display(member_ids: list[str]) -> list[SCIMMember]:
    """Get SCIMMember objects with display names for a list of member IDs."""
    prisma_client: Final = await _get_prisma_client_or_raise_exception()
    members: Final[list[SCIMMember]] = []

    for member_id in member_ids:
        user = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": member_id})
        if user:
            display_name = user.user_email or user.user_id
            members.append(SCIMMember(value=user.user_id, display=display_name, type="User"))

    return members


async def _handle_team_membership_changes(
    user_id: str,
    existing_teams: list[str],
    new_teams: list[str],
) -> None:
    """Handle adding/removing user from teams based on changes.

    Roster write failures propagate so the SCIM endpoint returns an error the IdP
    retries, instead of persisting a ``teams`` array the roster never received.
    """
    existing_teams_set: Final = set(existing_teams)
    new_teams_set: Final = set(new_teams)

    teams_to_add: Final = new_teams_set - existing_teams_set
    teams_to_remove: Final = existing_teams_set - new_teams_set

    if teams_to_add or teams_to_remove:
        await patch_team_membership(
            user_id=user_id,
            teams_ids_to_add_user_to=list(teams_to_add),
            teams_ids_to_remove_user_from=list(teams_to_remove),
            raise_on_error=True,
        )


SCIM_BLOCKED_METADATA_KEY: Final = "scim_blocked"


def _key_was_scim_blocked(metadata: object) -> bool:
    """True if a verification token carries the SCIM-block marker in metadata."""
    return isinstance(metadata, dict) and metadata.get(SCIM_BLOCKED_METADATA_KEY) is True


async def _set_user_keys_blocked(user_id: str, blocked: bool) -> int:
    """
    Block or unblock virtual keys owned by a user and invalidate them in the
    in-memory/redis caches so the change takes effect immediately.

    Each key SCIM blocks is tagged with ``metadata.scim_blocked = True``. On
    reactivation we only unblock keys carrying that marker, so a key an admin
    blocked manually for unrelated reasons is left alone.

    Returns the number of keys whose state was flipped.
    """
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    prisma_client: Final = await _get_prisma_client_or_raise_exception()

    if blocked:
        # `blocked` is a nullable column with no default, so existing rows
        # typically hold NULL; treat NULL as "not blocked" since SQL equality
        # on NULL would otherwise silently skip them.
        candidates = await _table(VerificationTokenRepository(prisma_client)).find_many(
            where={
                "user_id": user_id,
                "OR": [{"blocked": False}, {"blocked": None}],
            },
        )
        affected_keys = candidates
    else:
        candidates = await _table(VerificationTokenRepository(prisma_client)).find_many(
            where={"user_id": user_id, "blocked": True},
        )
        affected_keys = [k for k in candidates if _key_was_scim_blocked(k.metadata)]

    if not affected_keys:
        return 0

    for key_row in affected_keys:
        current_metadata: dict[str, object] = dict(key_row.metadata) if isinstance(key_row.metadata, dict) else {}
        if blocked:
            new_metadata = {**current_metadata, SCIM_BLOCKED_METADATA_KEY: True}
        else:
            new_metadata = {k: v for k, v in current_metadata.items() if k != SCIM_BLOCKED_METADATA_KEY}
        await _table(VerificationTokenRepository(prisma_client)).update(
            where={"token": key_row.token},
            data={"blocked": blocked, "metadata": safe_dumps(new_metadata)},
        )

    for key_row in affected_keys:
        await _delete_cache_key_object(
            hashed_token=key_row.token,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    verbose_proxy_logger.info(
        "SCIM: %s %d virtual key(s) for user_id=%s",
        "blocked" if blocked else "unblocked",
        len(affected_keys),
        user_id,
    )
    return len(affected_keys)


async def _delete_rows_referencing_user(prisma_client: PrismaClient, *, user_id: str) -> None:
    """Drop rows whose foreign keys reference ``LiteLLM_UserTable.user_id``.

    Required before deleting the user row itself, otherwise Postgres rejects
    the user delete with an FK constraint violation (e.g.
    ``LiteLLM_InvitationLink_user_id_fkey``).
    """
    await _table(InvitationLinkRepository(prisma_client)).delete_many(
        where={
            "OR": [
                {"user_id": user_id},
                {"created_by": user_id},
                {"updated_by": user_id},
            ]
        }
    )
    await _table(OrganizationMembershipRepository(prisma_client)).delete_many(where={"user_id": user_id})
    await _table(TeamMembershipRepository(prisma_client)).delete_many(where={"user_id": user_id})


def _scim_active_value(metadata: Mapping[str, object] | None) -> bool | None:
    """Read the SCIM active flag from a user's metadata dict, if present."""
    if not metadata:
        return None
    value: Final = metadata.get("scim_active")
    if value is None:
        return None
    return bool(value)


def _user_scim_active(user: LiteLLM_UserTable) -> bool | None:
    """Read the SCIM active flag off a user row's metadata, if present."""
    metadata: Final[dict[str, object] | None] = user.metadata
    return _scim_active_value(metadata)


async def _create_user_if_not_exists(user_id: str, created_via: str = "scim_group") -> NewUserResponse | None:
    """
    Helper function to create a user if they don't exist.

    Args:
        user_id: The user ID to create
        created_via: Context for where the user was created from

    Returns:
        LiteLLM_UserTable if user was created, None if creation failed
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user

    try:
        # Get default role for new internal users
        default_role: (
            Literal[
                LitellmUserRoles.PROXY_ADMIN,
                LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
                LitellmUserRoles.INTERNAL_USER,
                LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            ]
            | None
        ) = LitellmUserRoles.INTERNAL_USER_VIEW_ONLY
        if litellm.default_internal_user_params:
            default_role = litellm.default_internal_user_params.get("user_role")

        new_user_request: Final = NewUserRequest(
            user_id=user_id,
            user_email=user_id,  # We don't have email from group membership
            user_alias=None,
            teams=[],  # Teams will be added separately
            metadata={"created_via": created_via},
            auto_create_key=False,
            user_role=default_role,
        )

        created_user: Final = await new_user(data=new_user_request)
        verbose_proxy_logger.info("Created user %s via %s", user_id, created_via)
        return created_user

    except Exception as e:
        verbose_proxy_logger.exception("Failed to create user %s: %s", user_id, e)
        return None


async def _get_team_member_user_ids_from_team(team: LiteLLM_TeamTable) -> list[str]:
    """
    Get the IDs of the members from a team.

    Use one source of truth for the member IDs: team.members_with_roles

    """
    member_user_ids: Final[list[str]] = []
    for member in team.members_with_roles or []:
        if hasattr(member, "user_id") and member.user_id is not None:
            member_user_ids.append(member.user_id)
        elif isinstance(member, dict) and "user_id" in member:
            user_id = member.get("user_id")
            if user_id is not None:
                member_user_ids.append(user_id)
    return member_user_ids


# Dependency to set the correct SCIM Content-Type
async def set_scim_content_type(response: Response):
    """Sets the Content-Type header to application/scim+json"""
    # Check if content type is already application/json, only override in that case
    # Avoids overriding for non-JSON responses or already correct types if they were set manually
    response.headers["Content-Type"] = "application/scim+json"


def _get_resource_types(base_url: str = "/scim/v2") -> Sequence[SCIMResourceType]:
    """Return the list of SCIM ResourceType definitions per RFC 7643 Section 6."""
    return [
        SCIMResourceType(
            id="User",
            name="User",
            description="User Account",
            endpoint="/Users",
            schema_="urn:ietf:params:scim:schemas:core:2.0:User",
            meta={
                "location": f"{base_url}/ResourceTypes/User",
                "resourceType": "ResourceType",
            },
        ),
        SCIMResourceType(
            id="Group",
            name="Group",
            description="Group",
            endpoint="/Groups",
            schema_="urn:ietf:params:scim:schemas:core:2.0:Group",
            meta={
                "location": f"{base_url}/ResourceTypes/Group",
                "resourceType": "ResourceType",
            },
        ),
    ]


def _get_schemas() -> Sequence[SCIMSchema]:
    """Return the list of SCIM Schema definitions per RFC 7643 Section 7."""
    return [
        SCIMSchema(
            id="urn:ietf:params:scim:schemas:core:2.0:User",
            name="User",
            description="User Account",
            attributes=[
                SCIMSchemaAttribute(
                    name="userName",
                    type="string",
                    multiValued=False,
                    description="Unique identifier for the User.",
                    required=True,
                    mutability="readWrite",
                    returned="default",
                    uniqueness="server",
                ),
                SCIMSchemaAttribute(
                    name="name",
                    type="complex",
                    multiValued=False,
                    description="The components of the user's real name.",
                    required=False,
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="givenName",
                            type="string",
                            description="The given name of the User.",
                        ),
                        SCIMSchemaAttribute(
                            name="familyName",
                            type="string",
                            description="The family name of the User.",
                        ),
                        SCIMSchemaAttribute(
                            name="formatted",
                            type="string",
                            description="The full name.",
                        ),
                    ],
                ),
                SCIMSchemaAttribute(
                    name="displayName",
                    type="string",
                    multiValued=False,
                    description="The name of the User, suitable for display.",
                ),
                SCIMSchemaAttribute(
                    name="emails",
                    type="complex",
                    multiValued=True,
                    description="Email addresses for the user.",
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="value",
                            type="string",
                            description="Email address value.",
                        ),
                        SCIMSchemaAttribute(
                            name="type",
                            type="string",
                            description="Type of email (work, home, etc.).",
                        ),
                        SCIMSchemaAttribute(
                            name="primary",
                            type="boolean",
                            description="Whether this is the primary email.",
                        ),
                    ],
                ),
                SCIMSchemaAttribute(
                    name="active",
                    type="boolean",
                    multiValued=False,
                    description="Whether the user account is active.",
                ),
                SCIMSchemaAttribute(
                    name="groups",
                    type="complex",
                    multiValued=True,
                    description="Groups to which the user belongs.",
                    mutability="readOnly",
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="value",
                            type="string",
                            description="Group identifier.",
                        ),
                        SCIMSchemaAttribute(
                            name="display",
                            type="string",
                            description="Group display name.",
                        ),
                    ],
                ),
                SCIMSchemaAttribute(
                    name="entitlements",
                    type="complex",
                    multiValued=True,
                    description="A list of entitlements for the user.",
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="value",
                            type="string",
                            description="The value of an entitlement.",
                        ),
                        SCIMSchemaAttribute(
                            name="display",
                            type="string",
                            description="A human-readable name for the entitlement.",
                        ),
                        SCIMSchemaAttribute(
                            name="type",
                            type="string",
                            description="A label indicating the entitlement's function.",
                        ),
                        SCIMSchemaAttribute(
                            name="primary",
                            type="boolean",
                            description="Whether this is the primary entitlement.",
                        ),
                    ],
                ),
                SCIMSchemaAttribute(
                    name="roles",
                    type="complex",
                    multiValued=True,
                    description="A list of roles for the user.",
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="value",
                            type="string",
                            description="The value of a role.",
                        ),
                        SCIMSchemaAttribute(
                            name="display",
                            type="string",
                            description="A human-readable name for the role.",
                        ),
                        SCIMSchemaAttribute(
                            name="type",
                            type="string",
                            description="A label indicating the role's function.",
                        ),
                        SCIMSchemaAttribute(
                            name="primary",
                            type="boolean",
                            description="Whether this is the primary role.",
                        ),
                    ],
                ),
            ],
            meta={
                "location": "/scim/v2/Schemas/urn:ietf:params:scim:schemas:core:2.0:User",
                "resourceType": "Schema",
            },
        ),
        SCIMSchema(
            id="urn:ietf:params:scim:schemas:core:2.0:Group",
            name="Group",
            description="Group",
            attributes=[
                SCIMSchemaAttribute(
                    name="displayName",
                    type="string",
                    multiValued=False,
                    description="A human-readable name for the Group.",
                    required=True,
                    mutability="readWrite",
                    returned="default",
                    uniqueness="none",
                ),
                SCIMSchemaAttribute(
                    name="members",
                    type="complex",
                    multiValued=True,
                    description="A list of members of the Group.",
                    subAttributes=[
                        SCIMSchemaAttribute(
                            name="value",
                            type="string",
                            description="Member identifier.",
                        ),
                        SCIMSchemaAttribute(
                            name="display",
                            type="string",
                            description="Member display name.",
                        ),
                        SCIMSchemaAttribute(
                            name="type",
                            type="string",
                            description=(
                                'The type of member; canonical values are "User" and "Group". '
                                "Only members of type User are honored, LiteLLM teams contain users only."
                            ),
                        ),
                    ],
                ),
            ],
            meta={
                "location": "/scim/v2/Schemas/urn:ietf:params:scim:schemas:core:2.0:Group",
                "resourceType": "Schema",
            },
        ),
    ]


@scim_router.get(
    "",
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
@scim_router.get(
    "/",
    status_code=200,
    include_in_schema=False,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_scim_base(request: Request):
    """
    Base SCIM v2 endpoint for resource discovery per RFC 7644 Section 4.

    Returns a ListResponse of ResourceTypes supported by this SCIM service provider.
    Identity providers (Okta, Azure AD, etc.) use this endpoint for resource discovery.
    """
    verbose_proxy_logger.debug(
        "SCIM base resource discovery request: method=%s url=%s",
        request.method,
        request.url,
    )
    base_url: Final = str(request.base_url).rstrip("/") + "/scim/v2"
    resource_types: Final = _get_resource_types(base_url)
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(resource_types),
        "Resources": [rt.model_dump() for rt in resource_types],
    }


@scim_router.get(
    "/ResourceTypes",
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_resource_types(request: Request):
    """
    SCIM ResourceTypes endpoint per RFC 7644 Section 4.

    Returns a ListResponse of all resource types supported by this service provider.
    """
    verbose_proxy_logger.debug(
        "SCIM ResourceTypes request: method=%s url=%s",
        request.method,
        request.url,
    )
    base_url: Final = str(request.base_url).rstrip("/") + "/scim/v2"
    resource_types: Final = _get_resource_types(base_url)
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(resource_types),
        "Resources": [rt.model_dump() for rt in resource_types],
    }


@scim_router.get(
    "/ResourceTypes/{resource_type_id}",
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_resource_type(
    request: Request,
    resource_type_id: str = Path(..., title="ResourceType ID"),
):
    """
    Get a single ResourceType by ID per RFC 7644.
    """
    verbose_proxy_logger.debug("SCIM ResourceType request for id=%s", resource_type_id)
    base_url: Final = str(request.base_url).rstrip("/") + "/scim/v2"
    resource_types: Final = _get_resource_types(base_url)
    for rt in resource_types:
        if rt.id == resource_type_id:
            return rt.model_dump()
    raise HTTPException(
        status_code=404,
        detail={"error": f"ResourceType not found: {resource_type_id}"},
    )


@scim_router.get(
    "/Schemas",
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_schemas(request: Request):
    """
    SCIM Schemas endpoint per RFC 7643 Section 7.

    Returns a ListResponse of all schemas supported by this service provider.
    """
    verbose_proxy_logger.debug(
        "SCIM Schemas request: method=%s url=%s",
        request.method,
        request.url,
    )
    schemas: Final = _get_schemas()
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": len(schemas),
        "Resources": [s.model_dump() for s in schemas],
    }


@scim_router.get(
    "/Schemas/{schema_id:path}",
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_schema(
    request: Request,
    schema_id: str = Path(..., title="Schema URI"),
):
    """
    Get a single Schema by its URI per RFC 7643 Section 7.
    """
    verbose_proxy_logger.debug("SCIM Schema request for id=%s", schema_id)
    schemas: Final = _get_schemas()
    for s in schemas:
        if s.id == schema_id:
            return s.model_dump()
    raise HTTPException(
        status_code=404,
        detail={"error": f"Schema not found: {schema_id}"},
    )


@scim_router.get(
    "/ServiceProviderConfig",
    response_model=SCIMServiceProviderConfig,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_service_provider_config(request: Request):
    """Return SCIM Service Provider Configuration."""
    verbose_proxy_logger.debug(
        "SCIM ServiceProviderConfig request: method=%s url=%s headers=%s",
        request.method,
        request.url,
        _safe_get_request_headers(request),
    )
    meta: Final = {
        "resourceType": "ServiceProviderConfig",
        "location": str(request.url),
    }
    return SCIMServiceProviderConfig(meta=meta)


def _parse_scim_eq_filter(scim_filter: str) -> tuple[str, str] | None:
    """Parse the SCIM equality filters Okta uses before user lifecycle changes."""
    match: Final = re.match(
        r"""\s*([\w.]+)\s+eq\s+(['"]?)(.*?)\2\s*$""",
        scim_filter,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).lower(), match.group(3)


# User Endpoints
@scim_router.get(
    "/Users",
    response_model=SCIMListResponse,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_users(
    startIndex: int = Query(1, ge=1),
    count: int = Query(10, ge=1, le=100),
    filter: str | None = Query(None),
):
    """
    Get a list of users according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM GET USERS request: startIndex=%s count=%s filter=%s",
        startIndex,
        count,
        filter,
    )
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        # Parse filter if provided (basic support)
        where_conditions: Final[dict[str, object]] = {}
        if filter:
            # Okta locates users by userName before deprovisioning. LiteLLM
            # exposes SCIM userName from user_email, while older SCIM-created
            # users may still have user_id == userName, so support both.
            parsed_filter: Final = _parse_scim_eq_filter(filter)
            if parsed_filter:
                filter_attribute, filter_value = parsed_filter
                if filter_attribute == "username":
                    where_conditions["OR"] = [
                        {"user_email": filter_value},
                        {"user_id": filter_value},
                    ]
                elif filter_attribute == "emails.value":
                    where_conditions["user_email"] = filter_value

        # Get users from database
        users: Final[Sequence[LiteLLM_UserTable]] = await _table(UserRepository(prisma_client)).find_many(
            where=where_conditions,
            skip=(startIndex - 1),
            take=count,
            order={"created_at": "desc"},
        )

        # Get total count for pagination
        total_count: Final = await _table(UserRepository(prisma_client)).count(where=where_conditions)

        # Convert to SCIM format
        scim_users: Final[list[SCIMUser]] = []
        for user in users:
            scim_user = await ScimTransformations.transform_litellm_user_to_scim_user(user=user)
            scim_users.append(scim_user)

        return SCIMListResponse(
            totalResults=total_count,
            startIndex=startIndex,
            itemsPerPage=min(count, len(scim_users)),
            Resources=scim_users,
        )

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.get(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_user(
    user_id: str = Path(..., title="User ID"),
):
    """
    Get a single user by ID according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug("SCIM GET USER request for user_id=%s", user_id)
    try:
        user: Final = await _check_user_exists(user_id)

        # Convert to SCIM format
        scim_user: Final = await ScimTransformations.transform_litellm_user_to_scim_user(user)
        return scim_user

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.post(
    "/Users",
    response_model=SCIMUser,
    status_code=201,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def create_user(
    user: SCIMUser = Body(...),
):
    """
    Create a user according to SCIM v2 protocol
    """
    try:
        verbose_proxy_logger.debug("SCIM CREATE USER request: %s", user.model_dump())
        prisma_client: Final = await _get_prisma_client_or_raise_exception()

        # Extract data from SCIM user
        user_data: Final = _extract_scim_user_data(user)

        # Check if user already exists
        if user.userName:
            existing_user = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": user.userName})
            if existing_user:
                raise HTTPException(
                    status_code=409,
                    detail={"error": f"User already exists with username: {user.userName}"},
                )

        # Create user in database
        user_id: Final = user.userName or str(uuid.uuid4())
        metadata: Final = _build_scim_metadata(
            user_data["given_name"],
            user_data["family_name"],
            enterprise=user_data["enterprise"],
            entitlements=user_data["entitlements"],
            roles=user_data["roles"],
        )

        default_role: Final = _default_scim_user_role()
        admin_group: Final = await _get_scim_admin_group()
        resolved_role: Final = _resolve_scim_user_role(user.groups or [], admin_group, default_role)

        new_user_request: Final = NewUserRequest(
            user_id=user_id,
            user_email=user_data["user_email"],
            user_alias=user_data["user_alias"],
            teams=user_data["teams"],
            metadata=metadata,
            auto_create_key=False,
            user_role=resolved_role if admin_group is not None else default_role,
        )

        # Check if user with email already exists and update if found
        existing_user_scim: Final = await UserProvisionerHelpers.handle_existing_user_by_email(
            prisma_client=prisma_client,
            new_user_request=new_user_request,
            admin_group=admin_group,
        )

        if existing_user_scim:
            return existing_user_scim

        created_user: Final = await new_user(
            data=new_user_request,
        )

        scim_user: Final = await ScimTransformations.transform_litellm_user_to_scim_user(user=created_user)
        return scim_user
    except HTTPException as e:  # allow exceptions like SCIMUserAlreadyExists to be raised
        raise e
    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.put(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def update_user(
    user_id: str = Path(..., title="User ID"),
    user: SCIMUser = Body(...),
):
    """
    Update a user according to SCIM v2 protocol (full replacement)
    """
    verbose_proxy_logger.debug(
        "SCIM PUT USER request for user_id=%s: %s",
        user_id,
        user.model_dump(),
    )

    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_user: Final = await _check_user_exists(user_id)

        prev_active: Final = _user_scim_active(existing_user)

        user_data: Final = _extract_scim_user_data(user)

        # SCIM PUT may legally omit `active` (full-replace with the field absent).
        # Pydantic fills the model default, so distinguish "client sent active"
        # from "client omitted it" via model_fields_set, and preserve the prior
        # SCIM active state when omitted — otherwise a vanilla PUT to a
        # deactivated user would silently re-enable them and unblock their keys.
        client_set_active: Final = "active" in user.model_fields_set
        scim_active_for_metadata: Final = user_data["active"] if client_set_active else prev_active

        metadata: Final = _build_scim_metadata(
            user_data["given_name"],
            user_data["family_name"],
            scim_active_for_metadata,
            enterprise=user_data["enterprise"],
            entitlements=user_data["entitlements"],
            roles=user_data["roles"],
        )

        await _handle_team_membership_changes(
            user_id=user_id,
            existing_teams=existing_user.teams or [],
            new_teams=user_data["teams"],
        )

        update_data: Final = {
            "user_email": user_data["user_email"],
            "user_alias": user_data["user_alias"],
            "sso_user_id": user_data["sso_user_id"],
            "teams": user_data["teams"],
            "metadata": safe_dumps(metadata),
        }

        admin_group: Final = await _get_scim_admin_group()
        if admin_group is not None:
            update_data["user_role"] = _resolve_scim_user_role(
                user.groups or [], admin_group, _default_scim_user_role()
            )

        updated_user: Final = await _table(UserRepository(prisma_client)).update(
            where={"user_id": user_id},
            data=update_data,
        )

        if client_set_active:
            new_active: Final = _scim_active_value(metadata)
            if new_active is not None and new_active != (True if prev_active is None else prev_active):
                await _set_user_keys_blocked(user_id=user_id, blocked=not new_active)

        # Convert back to SCIM format
        scim_user: Final = await ScimTransformations.transform_litellm_user_to_scim_user(updated_user)

        return scim_user

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.delete(
    "/Users/{user_id}",
    status_code=204,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_user(
    user_id: str = Path(..., title="User ID"),
):
    """
    Delete a user according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug("SCIM DELETE USER request for user_id=%s", user_id)
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_user: Final = await _check_user_exists(user_id)

        # Get teams user belongs to
        found_teams: Final = tuple(
            [
                await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id})
                for team_id in existing_user.teams or []
            ]
        )
        teams: Final = tuple(team for team in found_teams if team)

        # Remove user from all teams
        for team in teams:
            current_members: Sequence[str] = team.members or []
            if user_id in current_members:
                new_members = [m for m in current_members if m != user_id]
                await _table(TeamRepository(prisma_client)).update(
                    where={"team_id": team.team_id}, data={"members": new_members}
                )

            team_row = LiteLLM_TeamTable.model_validate(team.model_dump())
            if any(member.user_id == user_id for member in team_row.members_with_roles or []):
                await team_member_delete(
                    data=TeamMemberDeleteRequest(team_id=team_row.team_id, user_id=user_id),
                    user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
                )

        await _set_user_keys_blocked(user_id=user_id, blocked=True)

        await _delete_rows_referencing_user(prisma_client, user_id=user_id)

        # Delete user
        await _table(UserRepository(prisma_client)).delete(where={"user_id": user_id})

        return Response(status_code=204)
    except Exception as e:
        raise handle_exception_on_proxy(e)


def _parse_member_entry(entry: object) -> SCIMMember | None:
    """Parse one entry of a SCIM patch value, or None when it carries no id."""
    if isinstance(entry, str):
        return SCIMMember(value=entry)

    fields: Final = _json_object_fields(entry)
    if fields is None:
        return None

    entry_value: Final = fields.get("value")
    if not entry_value:
        return None

    entry_display: Final = fields.get("display")
    entry_type: Final = fields.get("type")
    return SCIMMember(
        value=str(entry_value),
        display=str(entry_display) if entry_display is not None else None,
        type=entry_type if isinstance(entry_type, str) else None,
    )


def _parse_member_entries(value: object) -> tuple[SCIMMember, ...]:
    """Parse a SCIM patch value into members, keeping each entry's ``type``.

    PATCH bodies bypass SCIMGroup parsing (SCIMPatchOperation.value is untyped),
    so member objects arrive as raw dicts and the ``type`` that marks a nested
    group would otherwise be lost.
    """
    entries: Final[tuple[object, ...]] = tuple(value) if isinstance(value, list) else (value,)
    return tuple(member for member in (_parse_member_entry(entry) for entry in entries) if member is not None)


def _extract_group_values(value: object) -> list[str]:
    """Return group ids from a SCIM patch value."""
    return [member.value for member in _parse_member_entries(value)]


def _extract_ids_from_path_filter(path: str | None, attribute: str) -> list[str]:
    """Return ids from a SCIM filtered path like ``members[value eq "id"]``.

    Okta commonly sends membership removals as a filtered path and omits the
    request body ``value``, so the id lives only inside the ``[value eq "..."]``
    filter. The ``eq`` operator is matched case-insensitively per the SCIM
    spec; the id keeps its original case. Per the SCIM filter grammar the
    compared value must be quoted (single or double), so malformed unquoted
    filters yield no id. A quoted id may contain escaped quotes and
    backslashes (``\\"`` and ``\\\\``), which are unescaped before use.
    ``path`` must be the raw, case-preserving path from the patch op.
    """
    if not path:
        return []
    match: Final = re.match(
        rf"""\s*{re.escape(attribute)}\s*\[\s*value\s+eq\s+(['"])((?:\\.|[^\\])*?)\1\s*\]\s*$""",
        path,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    extracted: Final = re.sub(r"\\(.)", r"\1", match.group(2))
    return [extracted] if extracted else []


def _handle_displayname_update(op_type: str, value: object, update_data: dict[str, object]) -> None:
    """Handle displayname updates."""
    if op_type == "remove":
        update_data["user_alias"] = None
    else:
        update_data["user_alias"] = str(value)


def _handle_externalid_update(op_type: str, value: object, update_data: dict[str, object]) -> None:
    """Handle externalid updates."""
    if op_type == "remove":
        update_data["sso_user_id"] = None
    else:
        update_data["sso_user_id"] = str(value)


def _handle_active_update(op_type: str, value: object, metadata: dict[str, object]) -> None:
    """Handle active status updates."""
    if op_type == "remove":
        metadata.pop("scim_active", None)
    else:
        bool_val = value
        if isinstance(value, str):
            bool_val = value.lower() == "true"
        else:
            bool_val = bool(value)
        metadata["scim_active"] = bool_val


def _handle_name_update(path: str, op_type: str, value: object, scim_metadata: dict[str, object]) -> None:
    """Handle name field updates (givenName, familyName)."""
    if path == "name.givenname":
        if op_type == "remove":
            scim_metadata.pop("givenName", None)
        else:
            scim_metadata["givenName"] = str(value)
    elif path == "name.familyname":
        if op_type == "remove":
            scim_metadata.pop("familyName", None)
        else:
            scim_metadata["familyName"] = str(value)


def _handle_group_operations(op_type: str, value: object, teams_set: set[str], path: str | None) -> set[str] | None:
    """Handle group/team membership operations."""
    group_values = _extract_group_values(value)
    if not group_values and value is None:
        group_values = _extract_ids_from_path_filter(path, "groups")
    if op_type == "replace":
        return set(group_values)
    elif op_type == "add":
        teams_set.update(group_values)
    elif op_type == "remove":
        for gid in group_values:
            teams_set.discard(gid)
    return None


def _multi_valued_attribute_base(path: str) -> str:
    """The attribute name a SCIM path targets, stripped of any value filter or sub-attribute."""
    return path.split("[", 1)[0].split(".", 1)[0]


def _handle_multi_valued_attribute_update(path: str, op_type: str, value: object, metadata: dict[str, object]) -> None:
    """Handle add/replace/remove for the entitlements and roles multi-valued attributes."""
    base: Final = _multi_valued_attribute_base(path)
    metadata_key: Final = SCIM_MULTI_VALUED_ATTRIBUTE_METADATA_KEYS[base]
    if path != base:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Filtered or sub-attribute paths are not supported for {base}; PATCH the full attribute"},
        )

    if op_type == "remove":
        metadata.pop(metadata_key, None)
        return

    if value is None:
        raise HTTPException(
            status_code=400,
            detail={"error": f"The {op_type} operation on {base} requires a 'value' member (RFC 7644 Section 3.5.2)"},
        )

    normalized: Final = value if isinstance(value, list) else [value]
    try:
        attrs: Final = SCIM_MULTI_VALUED_LIST_ADAPTER.validate_python(normalized)
    except ValidationError:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid value for {base}: expected a list of objects with a 'value' sub-attribute"},
        )

    dumped: Final = [attr.model_dump(exclude_none=True) for attr in attrs]
    existing: Final = metadata.get(metadata_key)
    if op_type == "add" and isinstance(existing, list):
        metadata[metadata_key] = existing + dumped
        return
    metadata[metadata_key] = dumped


def _handle_generic_metadata(path: str, op_type: str, value: object, metadata: dict[str, object]) -> None:
    """Handle generic metadata operations for unknown paths."""
    if op_type == "remove":
        metadata.pop(path, None)
    else:
        metadata[path] = value


def _apply_patch_ops(
    existing_user: LiteLLM_UserTable,
    patch_ops: SCIMPatchOp,
) -> tuple[dict[str, object], set[str]]:
    """Apply patch operations and return update data and final team set."""
    update_data: Final[dict[str, object]] = {}
    metadata: Final = existing_user.metadata or {}
    scim_metadata: Final = metadata.get("scim_metadata", {})

    teams_set: Final[set[str]] = set(existing_user.teams or [])
    replace_team_set: set[str] | None = None

    for op in patch_ops.Operations:
        path = (op.path or "").lower()
        value = op.value
        op_type = op.op

        # Handle SCIM operations without path where value contains the fields
        if not path and isinstance(value, dict):
            for key, val in value.items():
                key_lower = key.lower()
                if key_lower == "active":
                    _handle_active_update(op_type, val, metadata)
                elif key_lower == "displayname":
                    _handle_displayname_update(op_type, val, update_data)
                elif key_lower == "externalid":
                    _handle_externalid_update(op_type, val, update_data)
                elif key_lower in SCIM_MULTI_VALUED_ATTRIBUTE_METADATA_KEYS:
                    _handle_multi_valued_attribute_update(key_lower, op_type, val, metadata)
                elif key_lower == "name" and isinstance(val, dict):
                    for name_key, name_val in val.items():
                        name_key_lower = name_key.lower()
                        if name_key_lower in ("givenname", "familyname"):
                            _handle_name_update(
                                f"name.{name_key_lower}",
                                op_type,
                                name_val,
                                scim_metadata,
                            )
            continue

        if path == "displayname":
            _handle_displayname_update(op_type, value, update_data)
        elif path == "externalid":
            _handle_externalid_update(op_type, value, update_data)
        elif path == "active":
            _handle_active_update(op_type, value, metadata)
        elif path in ("name.givenname", "name.familyname"):
            _handle_name_update(path, op_type, value, scim_metadata)
        elif _multi_valued_attribute_base(path) in SCIM_MULTI_VALUED_ATTRIBUTE_METADATA_KEYS:
            _handle_multi_valued_attribute_update(path, op_type, value, metadata)
        elif path.startswith("groups"):
            new_replace_set = _handle_group_operations(op_type, value, teams_set, op.path)
            if new_replace_set is not None:
                replace_team_set = new_replace_set
        else:
            _handle_generic_metadata(path, op_type, value, metadata)

    final_team_set: Final = replace_team_set if replace_team_set is not None else teams_set
    metadata["scim_metadata"] = scim_metadata
    update_data["metadata"] = metadata
    return update_data, final_team_set


def _is_user_not_in_team_error(exc: HTTPException) -> bool:
    """True when team_member_delete reports the user was already absent from the
    team, which is the idempotent no-op case for a removal."""
    detail: Final = exc.detail
    return isinstance(detail, dict) and detail.get("error") == "User not found in team"


@dataclass(frozen=True, slots=True)
class RosterWriteFailure:
    description: str
    status_code: int


def _roster_write_status(exc: Exception) -> int:
    if isinstance(exc, HTTPException):
        return exc.status_code
    if isinstance(exc, ProxyException):
        return int(exc.code) if exc.code.isdigit() else 500
    return 500


class SCIMRosterSyncError(Exception):
    """Every roster write in the batch was attempted; these are the ones that did not land.

    Rolling the successful ones back is not safe, since the compensating write can fail
    too and can strip a membership that pre-dated the push. Naming the exact failures
    instead lets the IdP's next push, which is idempotent, close the gap. handle_exception_on_proxy
    reads ``status_code`` off this, so a unanimous failure keeps its own status and a mixed
    batch reports 500.
    """

    def __init__(self, failures: tuple[RosterWriteFailure, ...], attempted: int) -> None:
        statuses: Final = frozenset(failure.status_code for failure in failures)
        self.failures: Final[tuple[RosterWriteFailure, ...]] = failures
        self.status_code: Final[int] = next(iter(statuses)) if len(statuses) == 1 else 500
        super().__init__(
            f"SCIM roster sync failed on {len(failures)} of {attempted} team membership writes, "
            f"leaving the roster partially updated. Retry the push to reconcile it. "
            f"Failed writes: {'; '.join(failure.description for failure in failures)}"
        )


async def _attempt_roster_write(label: str, write: Callable[[], Awaitable[object]]) -> tuple[RosterWriteFailure, ...]:
    """Run one roster write and return what failed, so the caller can keep going."""
    try:
        await write()
    except SCIMRosterSyncError as e:
        return e.failures
    except Exception as e:  # noqa: BLE001  # this boundary turns any write failure into a value so the batch continues
        verbose_proxy_logger.exception("SCIM roster write failed (%s): %s", label, e)
        return (RosterWriteFailure(description=f"{label}: {e}", status_code=_roster_write_status(e)),)
    return ()


async def _collect_roster_write_failures(
    writes: Sequence[tuple[str, Callable[[], Awaitable[object]]]],
) -> tuple[RosterWriteFailure, ...]:
    per_write: Final = tuple([await _attempt_roster_write(label, write) for label, write in writes])
    return tuple(chain.from_iterable(per_write))


async def _add_user_to_team(user_id: str, team_id: str) -> None:
    try:
        await team_member_add(
            data=TeamMemberAddRequest(
                team_id=team_id,
                member=Member(user_id=user_id, role="user"),
            ),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
    except ProxyException as e:
        if e.type != ProxyErrorTypes.team_member_already_in_team:
            raise
        verbose_proxy_logger.debug("User %s is already in team %s, skipping add", user_id, team_id)


async def _remove_user_from_team(user_id: str, team_id: str) -> None:
    try:
        await team_member_delete(
            data=TeamMemberDeleteRequest(team_id=team_id, user_id=user_id),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )
    except HTTPException as e:
        if not _is_user_not_in_team_error(e):
            raise
        verbose_proxy_logger.debug("User %s is not in team %s, skipping remove", user_id, team_id)


async def patch_team_membership(
    user_id: str,
    teams_ids_to_add_user_to: list[str],
    teams_ids_to_remove_user_from: list[str],
    raise_on_error: bool = False,
) -> bool:
    """
    Add or remove user from teams

    Handles duplicate membership gracefully (idempotent operation).
    A user already being in a team (on add) or already absent from it (on
    remove) is treated as a no-op, not an error.

    Every team is attempted before anything is reported, so one failing team cannot
    strand the others unattempted. When ``raise_on_error`` is True the writes that did
    not land are reported together, instead of a teams array the roster never received
    being persisted as a success.
    """
    writes: Final = tuple(
        chain(
            (
                (f"add {user_id} to {team_id}", partial(_add_user_to_team, user_id, team_id))
                for team_id in teams_ids_to_add_user_to
            ),
            (
                (f"remove {user_id} from {team_id}", partial(_remove_user_from_team, user_id, team_id))
                for team_id in teams_ids_to_remove_user_from
            ),
        )
    )
    failures: Final = await _collect_roster_write_failures(writes)
    if failures and raise_on_error:
        raise SCIMRosterSyncError(failures, attempted=len(writes))

    return True


@scim_router.patch(
    "/Users/{user_id}",
    response_model=SCIMUser,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def patch_user(
    user_id: str = Path(..., title="User ID"),
    patch_ops: SCIMPatchOp = Body(...),
):
    """
    Patch a user according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM PATCH USER request for user_id=%s: %s",
        user_id,
        patch_ops.model_dump(),
    )

    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_user: Final = await _check_user_exists(user_id)

        prev_active: Final = _user_scim_active(existing_user)

        update_data, final_team_set = _apply_patch_ops(
            existing_user=existing_user,
            patch_ops=patch_ops,
        )

        patched_metadata: Final = update_data.get("metadata")
        new_active: Final = _scim_active_value(patched_metadata if isinstance(patched_metadata, Mapping) else None)

        # Handle team membership changes
        await _handle_team_membership_changes(
            user_id=user_id,
            existing_teams=existing_user.teams or [],
            new_teams=list(final_team_set),
        )

        update_data["teams"] = list(final_team_set)

        admin_group: Final = await _get_scim_admin_group()
        if admin_group is not None:
            update_data["user_role"] = _resolve_scim_user_role(
                await _scim_groups_from_team_ids(prisma_client, list(final_team_set)),
                admin_group,
                _default_scim_user_role(),
            )

        # Serialize metadata to JSON string for Prisma to avoid GraphQL parsing issues
        if "metadata" in update_data and isinstance(update_data["metadata"], dict):
            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            update_data["metadata"] = safe_dumps(update_data["metadata"])

        updated_user: Final = await _table(UserRepository(prisma_client)).update(
            where={"user_id": user_id},
            data=update_data,
        )

        if new_active is not None and new_active != (True if prev_active is None else prev_active):
            await _set_user_keys_blocked(user_id=user_id, blocked=not new_active)

        scim_user: Final = await ScimTransformations.transform_litellm_user_to_scim_user(updated_user)

        return scim_user

    except Exception as e:
        raise handle_exception_on_proxy(e)


class _TeamWhereConditions(TypedDict, total=False):
    """The team columns SCIM GET /Groups can filter on, as Prisma where-conditions."""

    team_alias: str


# Group Endpoints
@scim_router.get(
    "/Groups",
    response_model=SCIMListResponse,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_groups(
    startIndex: int = Query(1, ge=1),
    count: int = Query(10, ge=1, le=100),
    filter: str | None = Query(None),
):
    """
    Get a list of groups according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM GET GROUPS request: startIndex=%s count=%s filter=%s",
        startIndex,
        count,
        filter,
    )
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        # Parse filter if provided (basic support)
        where_conditions: Final[_TeamWhereConditions] = {}
        if filter:
            # Very basic filter support - only handling displayName eq
            if "displayName eq" in filter:
                team_alias = filter.split("displayName eq ")[1].strip("\"'")
                where_conditions["team_alias"] = team_alias

        # Get teams from database
        teams: Final = await _table(TeamRepository(prisma_client)).find_many(
            where=where_conditions,
            skip=(startIndex - 1),
            take=count,
            order={"created_at": "desc"},
        )

        # Get total count for pagination
        total_count: Final = await _table(TeamRepository(prisma_client)).count(where=where_conditions)

        # Convert to SCIM format
        scim_groups: Final[list[SCIMGroup]] = []
        for team in teams:
            # Get team members with display names. members_with_roles is the
            # source of truth; the legacy `members` column is not populated by
            # team creation, so reading it here would report an empty member
            # list to the IdP and trigger repeated re-provisioning.
            members = await _get_team_members_display(await _get_team_member_user_ids_from_team(team))
            verbose_proxy_logger.debug("SCIM GET GROUPS members: %s", members)
            team_alias = getattr(team, "team_alias", team.team_id)
            team_created_at = team.created_at.isoformat() if team.created_at else None
            team_updated_at = team.updated_at.isoformat() if team.updated_at else None

            scim_group = SCIMGroup(
                schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
                id=team.team_id,
                displayName=team_alias,
                members=members,
                meta={
                    "resourceType": "Group",
                    "created": team_created_at,
                    "lastModified": team_updated_at,
                },
            )
            scim_groups.append(scim_group)

        verbose_proxy_logger.debug("SCIM GET GROUPS response: %s", scim_groups)
        return SCIMListResponse(
            totalResults=total_count,
            startIndex=startIndex,
            itemsPerPage=min(count, len(scim_groups)),
            Resources=scim_groups,
        )

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.get(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def get_group(
    group_id: str = Path(..., title="Group ID"),
):
    """
    Get a single group by ID according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug("SCIM GET GROUP request for group_id=%s", group_id)
    try:
        team: Final = await _check_team_exists(group_id)

        scim_group: Final = await ScimTransformations.transform_litellm_team_to_scim_group(team)
        verbose_proxy_logger.debug("SCIM GET GROUP response: %s", scim_group)
        return scim_group

    except Exception as e:
        raise handle_exception_on_proxy(e)


def _new_team_request_with_defaults(
    team_id: str,
    team_alias: str | None,
    members_with_roles: Sequence[Member],
) -> NewTeamRequest:
    """Build the SCIM group's team request, applying litellm.default_team_params
    (including models) the same way SSO auto-created teams do."""
    default_params: Final = litellm.default_team_params
    defaults: Final[Mapping[str, object]] = (
        deepcopy(default_params)
        if isinstance(default_params, dict)
        else default_params.model_dump(exclude_none=True)
        if default_params is not None
        else {}
    )
    default_metadata: Final = defaults.get("metadata")
    metadata: Final = {
        **(default_metadata if isinstance(default_metadata, dict) else {}),
        SCIM_MANAGED_TEAM_METADATA_KEY: True,
    }
    return NewTeamRequest.model_validate(
        {
            **defaults,
            "team_id": team_id,
            "team_alias": team_alias,
            "members_with_roles": members_with_roles,
            "metadata": metadata,
        }
    )


@scim_router.post(
    "/Groups",
    response_model=SCIMGroup,
    status_code=201,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def create_group(
    group: SCIMGroup = Body(...),
):
    """
    Create a group according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM CREATE GROUP request: %s",
        group.model_dump(),
    )
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()

        # Generate ID if not provided
        team_id: Final = group.id or group.externalId or str(uuid.uuid4())

        # Check if team already exists
        existing_team: Final = await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": team_id})

        if existing_team:
            raise HTTPException(
                status_code=409,
                detail={"error": f"Group already exists with ID: {team_id}"},
            )

        # Extract and validate group members (all users must exist)
        member_result: Final = await _extract_group_member_ids(group)
        members_with_roles = [Member(user_id=member_id, role="user") for member_id in member_result.all_member_ids]

        # Create team in database
        created_team: Final = await new_team(
            data=_new_team_request_with_defaults(
                team_id=team_id,
                team_alias=group.displayName,
                members_with_roles=members_with_roles,
            ),
            http_request=Request(scope={"type": "http", "path": "/scim/v2/Groups"}),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

        await _recompute_scim_member_roles(prisma_client, member_result.all_member_ids)

        scim_group: Final = await ScimTransformations.transform_litellm_team_to_scim_group(created_team)
        return scim_group
    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.put(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def update_group(
    group_id: str = Path(..., title="Group ID"),
    group: SCIMGroup = Body(...),
):
    """
    Update a group according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM PUT GROUP request for group_id=%s: %s",
        group_id,
        group.model_dump(),
    )
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_team: Final = await _check_team_exists(group_id)

        # Extract and validate group members (all users must exist)
        member_result: Final = await _extract_group_member_ids(group)
        verbose_proxy_logger.debug("SCIM PUT GROUP all_member_ids: %s", member_result.all_member_ids)
        verbose_proxy_logger.debug("SCIM PUT GROUP created_users: %s", len(member_result.created_users))

        # Prepare update data
        existing_metadata: Final = existing_team.metadata if existing_team.metadata else {}
        updated_metadata: Final = {
            **existing_metadata,
            SCIM_TEAM_DATA_METADATA_KEY: group.model_dump(),
            SCIM_MANAGED_TEAM_METADATA_KEY: True,
        }

        update_data: Final = {
            "team_alias": group.displayName,
            "metadata": safe_dumps(updated_metadata),
        }

        # Update team in database
        updated_team: Final = await _table(TeamRepository(prisma_client)).update(
            where={"team_id": group_id},
            data=update_data,
        )

        # Handle user-team relationship changes
        current_members: Final = set(await _get_team_member_user_ids_from_team(existing_team))
        verbose_proxy_logger.debug("SCIM PUT GROUP current_members: %s", current_members)
        final_members: Final = set(member_result.all_member_ids)
        verbose_proxy_logger.debug("SCIM PUT GROUP final_members: %s", final_members)

        await _handle_group_membership_changes(
            group_id=group_id,
            current_members=current_members,
            final_members=final_members,
        )

        # A rename can flip whether this group matches scim_admin_group by display
        # name, so retained members must be re-resolved too, not just the ones whose
        # membership changed.
        alias_changed: Final = existing_team.team_alias != group.displayName
        await _recompute_scim_member_roles(
            prisma_client,
            (current_members | final_members if alias_changed else current_members ^ final_members),
        )

        # Convert to SCIM format and return
        scim_group: Final = await ScimTransformations.transform_litellm_team_to_scim_group(updated_team)
        return scim_group

    except Exception as e:
        raise handle_exception_on_proxy(e)


@scim_router.delete(
    "/Groups/{group_id}",
    status_code=204,
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_group(
    group_id: str = Path(..., title="Group ID"),
):
    """
    Delete a group according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug("SCIM DELETE GROUP request for group_id=%s", group_id)
    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_team: Final = await _check_team_exists(group_id)

        member_ids: Final = await _get_team_member_user_ids_from_team(existing_team)

        # For each member, remove this team from their teams list
        for member_id in member_ids:
            user = await _table(UserRepository(prisma_client)).find_unique(where={"user_id": member_id})
            if user:
                current_teams = user.teams or []
                if group_id in current_teams:
                    new_teams = [t for t in current_teams if t != group_id]
                    await _table(UserRepository(prisma_client)).update(
                        where={"user_id": member_id}, data={"teams": new_teams}
                    )

        await _recompute_scim_member_roles(prisma_client, member_ids)

        # Delete team
        await _table(TeamRepository(prisma_client)).delete(where={"team_id": group_id})

        return Response(status_code=204)

    except Exception as e:
        raise handle_exception_on_proxy(e)


async def _process_group_patch_operations(
    patch_ops: SCIMPatchOp, existing_team: LiteLLM_TeamTable, prisma_client: PrismaClient
) -> tuple[dict[str, object], set[str], set[str] | None]:
    """Process patch operations for a group and return update data, final members
    and, when the request contained a member ``replace`` op, the absolute target
    roster it declared (``None`` otherwise).

    ``add``/``remove`` are deltas relative to the current roster, but ``replace``
    is absolute: it declares the roster is exactly this set, so the caller must
    reconcile against it as a set-to-target rather than rebasing it onto a
    concurrently-mutated roster.

    A ``remove`` drops the ids it names without resolving them first. Removal is
    idempotent and cannot put anything on a roster, while resolving would make it
    conditional on what the id turns out to be and leave members we should never
    have admitted - the phantom users this endpoint used to create for nested
    groups - impossible to clean up.
    """
    update_data: Final[dict[str, object]] = {}

    # Create a fresh copy of existing metadata to avoid Prisma issues
    metadata: Final = {**(existing_team.metadata or {}), SCIM_MANAGED_TEAM_METADATA_KEY: True}

    # Track member changes. members_with_roles is the source of truth for team
    # membership; the legacy `members` column is not populated by team creation
    # or the real team endpoints, so seeding from it would make an `add`/`remove`
    # operation recompute the member set from an empty base and silently drop
    # everyone already in the team.
    current_members: Final = set(await _get_team_member_user_ids_from_team(existing_team))
    final_members = current_members.copy()

    # Process each patch operation
    for op in patch_ops.Operations:
        path = (op.path or "").lower()
        value = op.value
        op_type = op.op

        if path == "displayname":
            if op_type == "remove":
                update_data["team_alias"] = None
            else:
                update_data["team_alias"] = str(value)
        elif path == "externalid":
            if op_type == "remove":
                metadata.pop("externalId", None)
            else:
                metadata["externalId"] = str(value)
        elif path.startswith("members"):
            # Handle member operations
            patched_members = (
                _parse_member_entries(value)
                if value is not None
                else tuple(
                    SCIMMember(value=member_id) for member_id in _extract_ids_from_path_filter(op.path, "members")
                )
            )

            if op_type == "remove":
                final_members = final_members - await _member_ids_to_drop(
                    patched_members, frozenset(final_members), prisma_client
                )
            else:
                member_result = await _resolve_group_member_ids(
                    members=patched_members,
                    created_via="scim_group_patch",
                    prisma_client=prisma_client,
                )
                if op_type == "replace":
                    final_members = set(member_result.all_member_ids)
                elif op_type == "add":
                    final_members = final_members | set(member_result.all_member_ids)
        else:
            # Handle other generic metadata
            if op_type == "remove":
                metadata.pop(path, None)
            else:
                metadata[path] = value

    update_data["metadata"] = metadata

    member_replace_present: Final = any(
        op.op == "replace" and (op.path or "").lower().startswith("members") for op in patch_ops.Operations
    )
    replace_target: Final = set(final_members) if member_replace_present else None

    return update_data, final_members, replace_target


async def _apply_group_patch_updates(group_id: str, update_data: dict[str, object], prisma_client: PrismaClient):
    """Apply the group's metadata/displayName patch updates to the database.

    Membership itself is not written here; it is reconciled onto the source of
    truth (members_with_roles and each member's user.teams) by
    _handle_group_membership_changes via team_member_add/team_member_delete.
    Writing the legacy `members` column here too would create a second, unread
    copy of membership that could drift from the source of truth.
    """
    if "metadata" in update_data and isinstance(update_data["metadata"], dict):
        update_data["metadata"] = safe_dumps(update_data["metadata"])

    if update_data:
        return await TeamRepository(prisma_client).table.update(
            where={"team_id": group_id},
            data=update_data,
        )
    return await TeamRepository(prisma_client).table.find_unique(where={"team_id": group_id})


async def _handle_group_membership_changes(group_id: str, current_members: set[str], final_members: set[str]) -> None:
    """Reconcile the group roster, attempting every member before reporting failures.

    Aborting on the first failure would leave the remaining members unattempted on top
    of unrolled-back, so every member is written and the ones that failed are named for
    the IdP's next push to reconcile.
    """
    members_to_add: Final = sorted(final_members - current_members)
    members_to_remove: Final = sorted(current_members - final_members)

    verbose_proxy_logger.debug("members_to_add: %s", members_to_add)
    verbose_proxy_logger.debug("members_to_remove: %s", members_to_remove)

    writes: Final = tuple(
        chain(
            (
                (
                    f"add {member_id} to {group_id}",
                    partial(
                        patch_team_membership,
                        user_id=member_id,
                        teams_ids_to_add_user_to=[group_id],
                        teams_ids_to_remove_user_from=[],
                        raise_on_error=True,
                    ),
                )
                for member_id in members_to_add
            ),
            (
                (
                    f"remove {member_id} from {group_id}",
                    partial(
                        patch_team_membership,
                        user_id=member_id,
                        teams_ids_to_add_user_to=[],
                        teams_ids_to_remove_user_from=[group_id],
                        raise_on_error=True,
                    ),
                )
                for member_id in members_to_remove
            ),
        )
    )
    failures: Final = await _collect_roster_write_failures(writes)
    if failures:
        raise SCIMRosterSyncError(failures, attempted=len(writes))


@scim_router.patch(
    "/Groups/{group_id}",
    response_model=SCIMGroup,
    status_code=200,
    dependencies=[Depends(user_api_key_auth), Depends(set_scim_content_type)],
)
async def patch_group(
    group_id: str = Path(..., title="Group ID"),
    patch_ops: SCIMPatchOp = Body(...),
):
    """
    Patch a group according to SCIM v2 protocol
    """
    verbose_proxy_logger.debug(
        "SCIM PATCH GROUP request for group_id=%s: %s",
        group_id,
        patch_ops.model_dump(),
    )

    try:
        prisma_client: Final = await _get_prisma_client_or_raise_exception()
        existing_team: Final = await _check_team_exists(group_id)

        # Process patch operations
        update_data, final_members, replace_target = await _process_group_patch_operations(
            patch_ops, existing_team, prisma_client
        )

        snapshot_members: Final = set(await _get_team_member_user_ids_from_team(existing_team))
        intended_add: Final = final_members - snapshot_members
        intended_remove: Final = snapshot_members - final_members

        # Apply the metadata/displayName updates to the database
        updated_team = await _apply_group_patch_updates(group_id, update_data, prisma_client)

        refreshed_team: Final = await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": group_id})
        refreshed_current: Final = (
            set(
                await _get_team_member_user_ids_from_team(LiteLLM_TeamTable.model_validate(refreshed_team.model_dump()))
            )
            if refreshed_team
            else snapshot_members
        )

        effective_final: Final = (
            replace_target if replace_target is not None else (refreshed_current | intended_add) - intended_remove
        )

        await _handle_group_membership_changes(group_id, refreshed_current, effective_final)

        # A rename can flip whether this group matches scim_admin_group by display
        # name, so retained members must be re-resolved too, not just the ones whose
        # membership changed.
        new_alias: Final = update_data.get("team_alias", existing_team.team_alias)
        alias_changed: Final = new_alias != existing_team.team_alias
        await _recompute_scim_member_roles(
            prisma_client,
            (refreshed_current | effective_final if alias_changed else refreshed_current ^ effective_final),
        )

        # Refresh team one more time to get final state after membership changes
        final_team: Final = await _table(TeamRepository(prisma_client)).find_unique(where={"team_id": group_id})
        if final_team:
            updated_team = final_team

        if updated_team is None:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Group not found with ID: {group_id}"},  # mutable-ok: FastAPI detail contract
            )

        # Convert to SCIM format and return
        scim_group: Final = await ScimTransformations.transform_litellm_team_to_scim_group(
            LiteLLM_TeamTable.model_validate(updated_team.model_dump())
        )
        return scim_group

    except Exception as e:
        raise handle_exception_on_proxy(e)
