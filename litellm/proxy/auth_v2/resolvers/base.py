from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from scim2_models import Group as ScimGroup
from scim2_models import User as ScimUser

from litellm.proxy.auth_v2.models import Credential, Principal


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


@runtime_checkable
class IdentityStore(IdentityResolver, ProvisioningStore, Protocol):
    """An identity backend: resolves credentials and provisions SCIM users/groups.

    This is the single interface every implementation satisfies (in-memory,
    database, ...). Resolution and provisioning live behind one store so a
    provisioned user is immediately resolvable.
    """
