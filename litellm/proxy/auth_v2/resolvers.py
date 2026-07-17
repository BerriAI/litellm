from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, List, Optional, Protocol

from scim2_models import Group as ScimGroup
from scim2_models import User as ScimUser

from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.auth.auth_checks import (
    get_key_object,
    get_org_object,
    get_team_object,
    get_user_object,
)
from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.models import (
    AuthMethod,
    Credential,
    EndUserIdentity,
    OrganizationIdentity,
    Principal,
    PrincipalType,
    ProjectIdentity,
    TeamIdentity,
    TeamRole,
    UserIdentity,
)
from litellm.proxy.auth_v2.utils import (
    db_team_to_scim,
    db_user_to_scim,
    map_role,
    member_role,
    scim_group_to_db,
    scim_user_to_db,
    team_role,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.models.user import LiteLLM_UserTable
    from litellm.proxy.utils import PrismaClient


class Resolver(Protocol):
    async def resolve(self, credential: Credential) -> Principal:
        """Resolve a verified credential to a Principal.

        Must return a freshly constructed Principal, never a cached or shared
        instance. The caller stamps request-scoped state (the network context)
        onto the returned object, so handing back a shared one would leak that
        state across concurrent requests for the same identity. Cache the
        underlying identity lookups (as the DB resolver does), not the assembled
        Principal.
        """
        ...


class ProvisioningStore(Protocol):
    async def upsert_user(self, user: ScimUser) -> ScimUser: ...
    async def get_user(self, resource_id: str) -> Optional[ScimUser]: ...
    async def deactivate_user(self, resource_id: str) -> None: ...
    async def list_users(self, filter_expr: Optional[str]) -> List[ScimUser]: ...
    async def upsert_group(self, group: ScimGroup) -> ScimGroup: ...
    async def get_group(self, resource_id: str) -> Optional[ScimGroup]: ...
    async def delete_group(self, resource_id: str) -> None: ...
    async def list_groups(self, filter_expr: Optional[str]) -> List[ScimGroup]: ...


class DbResolver(Resolver, ProvisioningStore):
    """Resolves credentials against the proxy's Prisma tables and provisions
    SCIM users/groups into ``LiteLLM_UserTable`` / ``LiteLLM_TeamTable``.

    Resolution and provisioning share one object so a provisioned user is
    immediately resolvable. The Prisma client and key cache are injected so this
    stays a plain object the composition root can build once the proxy DB is
    connected.
    """

    def __init__(self, prisma_client: "PrismaClient", cache: "DualCache") -> None:
        self._prisma = prisma_client
        self._cache = cache

    async def resolve(self, credential: Credential) -> Principal:
        if credential.method == AuthMethod.API_KEY:
            return await self._resolve_api_key(credential)
        if credential.method == AuthMethod.MUTUAL_TLS:
            return self._service_account(credential)
        return await self._resolve_subject(credential)

    async def _resolve_api_key(self, credential: Credential) -> Principal:
        raw = credential.claims.get("_raw_api_key")
        if not isinstance(raw, str):
            raise errors.invalid_token()
        try:
            key = await get_key_object(hash_token(raw), self._prisma, self._cache)
        except Exception as exc:
            raise errors.invalid_token() from exc
        if key.blocked:
            raise errors.account_disabled()
        return self._principal_from_key(credential, key)

    async def _resolve_subject(self, credential: Credential) -> Principal:
        email = credential.claims.get("email")
        try:
            user = await get_user_object(
                user_id=credential.subject,
                prisma_client=self._prisma,
                user_api_key_cache=self._cache,
                user_id_upsert=False,
                sso_user_id=credential.subject,
                user_email=email if isinstance(email, str) else None,
            )
        except Exception as exc:
            raise errors.invalid_token() from exc
        if user is None:
            raise errors.invalid_token()
        return await self._principal_from_user(credential, user)

    def _service_account(self, credential: Credential) -> Principal:
        return Principal(
            principal_type=PrincipalType.SERVICE_ACCOUNT,
            subject=credential.subject,
            issuer=credential.issuer,
            audience=list(credential.audience),
            scopes=list(credential.scopes),
            auth_method=credential.method,
            credential_ref=credential.credential_ref,
        )

    def _principal_from_key(
        self, credential: Credential, key: UserAPIKeyAuth
    ) -> Principal:
        teams: List[TeamIdentity] = []
        if key.team_id is not None:
            role = (
                team_role(key.team_member.role) if key.team_member else TeamRole.MEMBER
            )
            teams.append(TeamIdentity(id=key.team_id, name=key.team_alias, role=role))
        organization = (
            OrganizationIdentity(id=key.org_id, name=key.organization_alias)
            if key.org_id is not None
            else None
        )
        user = (
            UserIdentity(id=key.user_id, email=key.user_email)
            if key.user_id is not None
            else None
        )
        project = (
            ProjectIdentity(id=key.project_id, name=key.project_alias)
            if key.project_id is not None
            else None
        )
        end_user = (
            EndUserIdentity(id=key.end_user_id) if key.end_user_id is not None else None
        )
        mapped = map_role(key.user_role)
        return Principal(
            principal_type=(
                PrincipalType.HUMAN if key.user_id else PrincipalType.SERVICE_ACCOUNT
            ),
            subject=key.user_id or key.key_alias or credential.subject,
            issuer=credential.issuer,
            user=user,
            organization=organization,
            teams=teams,
            project=project,
            end_user=end_user,
            roles=[mapped] if mapped else [],
            scopes=list(credential.scopes),
            auth_method=credential.method,
            credential_ref=credential.credential_ref,
        )

    async def _principal_from_user(
        self, credential: Credential, user: "LiteLLM_UserTable"
    ) -> Principal:
        teams: List[TeamIdentity] = []
        for team_id in user.teams or []:
            try:
                team = await get_team_object(team_id, self._prisma, self._cache)
            except Exception:
                continue
            teams.append(
                TeamIdentity(
                    id=team_id,
                    name=team.team_alias,
                    role=member_role(team.members_with_roles, user.user_id),
                )
            )

        organization = await self._organization(user)
        roles = [role for role in (map_role(user.user_role),) if role is not None]
        return Principal(
            principal_type=PrincipalType.HUMAN,
            subject=credential.subject,
            issuer=credential.issuer,
            audience=list(credential.audience),
            user=UserIdentity(
                id=user.user_id,
                external_id=user.sso_user_id,
                email=user.user_email,
                display_name=user.user_alias,
            ),
            organization=organization,
            teams=teams,
            roles=roles,
            scopes=list(credential.scopes),
            auth_method=credential.method,
            credential_ref=credential.credential_ref,
        )

    async def _organization(
        self, user: "LiteLLM_UserTable"
    ) -> Optional[OrganizationIdentity]:
        if user.organization_id is None:
            return None
        try:
            org = await get_org_object(user.organization_id, self._prisma, self._cache)
        except Exception:
            org = None
        name = org.organization_alias if org is not None else None
        return OrganizationIdentity(id=user.organization_id, name=name)

    async def upsert_user(self, user: ScimUser) -> ScimUser:
        repo = UserRepository(self._prisma)
        data = scim_user_to_db(user)
        existing = (
            await repo.table.find_unique(where={"user_id": user.id})
            if user.id
            else None
        )
        if existing is None:
            data["user_id"] = user.id or str(uuid.uuid4())
            stored = await repo.table.create(data=data)
        else:
            stored = await repo.table.update(where={"user_id": user.id}, data=data)
        return db_user_to_scim(stored)

    async def get_user(self, resource_id: str) -> Optional[ScimUser]:
        stored = await UserRepository(self._prisma).table.find_unique(
            where={"user_id": resource_id}
        )
        return db_user_to_scim(stored) if stored is not None else None

    async def deactivate_user(self, resource_id: str) -> None:
        repo = UserRepository(self._prisma)
        stored = await repo.table.find_unique(where={"user_id": resource_id})
        if stored is None:
            return
        metadata = dict(getattr(stored, "metadata", None) or {})
        metadata["scim_active"] = False
        await repo.table.update(
            where={"user_id": resource_id}, data={"metadata": metadata}
        )

    async def list_users(self, filter_expr: Optional[str]) -> List[ScimUser]:
        rows = await UserRepository(self._prisma).table.find_many()
        return [db_user_to_scim(row) for row in rows]

    async def upsert_group(self, group: ScimGroup) -> ScimGroup:
        repo = TeamRepository(self._prisma)
        data = scim_group_to_db(group)
        existing = (
            await repo.table.find_unique(where={"team_id": group.id})
            if group.id
            else None
        )
        if existing is None:
            data["team_id"] = group.id or str(uuid.uuid4())
            stored = await repo.table.create(data=data)
        else:
            stored = await repo.table.update(where={"team_id": group.id}, data=data)
        return db_team_to_scim(stored)

    async def get_group(self, resource_id: str) -> Optional[ScimGroup]:
        stored = await TeamRepository(self._prisma).table.find_unique(
            where={"team_id": resource_id}
        )
        return db_team_to_scim(stored) if stored is not None else None

    async def delete_group(self, resource_id: str) -> None:
        await TeamRepository(self._prisma).table.delete(where={"team_id": resource_id})

    async def list_groups(self, filter_expr: Optional[str]) -> List[ScimGroup]:
        rows = await TeamRepository(self._prisma).table.find_many()
        return [db_team_to_scim(row) for row in rows]
