from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from scim2_models import Group as ScimGroup
from scim2_models import User as ScimUser

from . import errors
from .models import (
    AuthMethod,
    Credential,
    Principal,
    PrincipalType,
    TeamIdentity,
    UserIdentity,
)
from .rbac import Role


@runtime_checkable
class IdentityResolver(Protocol):
    async def resolve(self, credential: Credential) -> Principal: ...


@runtime_checkable
class ProvisioningStore(Protocol):
    async def upsert_user(self, user: ScimUser) -> ScimUser: ...
    async def get_user(self, resource_id: str) -> Optional[ScimUser]: ...
    async def deactivate_user(self, resource_id: str) -> None: ...
    async def list_users(self, filter_expr: Optional[str]) -> List[ScimUser]: ...
    async def upsert_group(self, group: ScimGroup) -> ScimGroup: ...
    async def get_group(self, resource_id: str) -> Optional[ScimGroup]: ...
    async def delete_group(self, resource_id: str) -> None: ...
    async def list_groups(self, filter_expr: Optional[str]) -> List[ScimGroup]: ...


def _hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _roles_from_claims(claims: Dict[str, Any]) -> List[Role]:
    raw = claims.get("roles", [])
    if not isinstance(raw, list):
        return []
    valid = {role.value for role in Role}
    return [Role(value) for value in raw if value in valid]


def _public_claims(claims: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in claims.items() if not key.startswith("_")}


class InMemoryIdentityStore(IdentityResolver, ProvisioningStore):
    def __init__(
        self,
        api_keys: Optional[Dict[str, Principal]] = None,
        subjects: Optional[Dict[str, Principal]] = None,
        users: Optional[Dict[str, ScimUser]] = None,
        groups: Optional[Dict[str, ScimGroup]] = None,
    ) -> None:
        self._api_keys = api_keys or {}
        self._subjects = subjects or {}
        self._users = users or {}
        self._groups = groups or {}

    async def resolve(self, credential: Credential) -> Principal:
        if credential.method == AuthMethod.API_KEY:
            principal = self._resolve_api_key(credential)
        else:
            principal = self._resolve_subject(credential)
        self._reject_if_deactivated(principal)
        return principal

    def _reject_if_deactivated(self, principal: Principal) -> None:
        user = self._lookup_scim_user(principal)
        if user is not None and user.active is False:
            raise errors.account_disabled()

    def _resolve_teams(self, claims: Dict[str, Any]) -> List[TeamIdentity]:
        groups = claims.get("groups", [])
        if not isinstance(groups, list):
            return []
        teams: List[TeamIdentity] = []
        for group in groups:
            scim_group = self._find_group(str(group))
            if scim_group is not None:
                teams.append(
                    TeamIdentity(
                        id=scim_group.id or str(group),
                        name=scim_group.display_name or str(group),
                    )
                )
        return teams

    def _find_group(self, value: str) -> Optional[ScimGroup]:
        for group in self._groups.values():
            if group.id == value or group.display_name == value:
                return group
        return None

    def _lookup_scim_user(self, principal: Principal) -> Optional[ScimUser]:
        if principal.user is None:
            return None
        by_id = self._users.get(principal.user.id)
        if by_id is not None:
            return by_id
        external = principal.user.external_id
        if external:
            for user in self._users.values():
                if user.external_id == external:
                    return user
        return None

    def _resolve_api_key(self, credential: Credential) -> Principal:
        raw = credential.claims.get("_raw_api_key")
        if not isinstance(raw, str):
            raise errors.invalid_token()
        principal = self._api_keys.get(_hash_api_key(raw))
        if principal is None:
            raise errors.invalid_token()
        return principal

    def _resolve_subject(self, credential: Credential) -> Principal:
        stored = self._subjects.get(f"{credential.issuer}|{credential.subject}")
        if stored is not None:
            return stored
        return self._principal_from_claims(credential)

    def _principal_from_claims(self, credential: Credential) -> Principal:
        claims = credential.claims
        if credential.method == AuthMethod.MUTUAL_TLS:
            return Principal(
                principal_type=PrincipalType.SERVICE_ACCOUNT,
                subject=credential.subject,
                issuer=credential.issuer,
                audience=list(credential.audience),
                scopes=list(credential.scopes),
                auth_method=credential.method,
                credential_ref=credential.credential_ref,
                claims=_public_claims(claims),
            )
        return Principal(
            principal_type=PrincipalType.HUMAN,
            subject=credential.subject,
            issuer=credential.issuer,
            audience=list(credential.audience),
            user=UserIdentity(
                id=credential.subject,
                external_id=credential.subject,
                email=claims.get("email"),
                user_name=claims.get("preferred_username"),
                display_name=claims.get("name"),
            ),
            teams=self._resolve_teams(claims),
            roles=_roles_from_claims(claims),
            scopes=list(credential.scopes),
            auth_method=credential.method,
            credential_ref=credential.credential_ref,
            claims=_public_claims(claims),
        )

    async def upsert_user(self, user: ScimUser) -> ScimUser:
        if not user.id:
            user.id = str(uuid.uuid4())
        self._users[user.id] = user
        return user

    async def get_user(self, resource_id: str) -> Optional[ScimUser]:
        return self._users.get(resource_id)

    async def deactivate_user(self, resource_id: str) -> None:
        user = self._users.get(resource_id)
        if user is not None:
            user.active = False

    async def list_users(self, filter_expr: Optional[str]) -> List[ScimUser]:
        return list(self._users.values())

    async def upsert_group(self, group: ScimGroup) -> ScimGroup:
        if not group.id:
            group.id = str(uuid.uuid4())
        self._groups[group.id] = group
        return group

    async def get_group(self, resource_id: str) -> Optional[ScimGroup]:
        return self._groups.get(resource_id)

    async def delete_group(self, resource_id: str) -> None:
        self._groups.pop(resource_id, None)

    async def list_groups(self, filter_expr: Optional[str]) -> List[ScimGroup]:
        return list(self._groups.values())
