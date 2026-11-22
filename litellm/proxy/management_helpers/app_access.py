from dataclasses import dataclass
from itertools import chain
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.proxy.management_helpers.customer_hierarchy import PrismaRecord
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.table_repositories import PrismaTableRepository


class _UserRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_usertable"


class _TeamRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_teamtable"


class _OrganizationRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_organizationtable"


class _OrganizationMembershipRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_organizationmembership"


class _ObjectPermissionRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_objectpermissiontable"


class _Grantee(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    object_permission_id: str | None = None
    teams: tuple[str, ...] | None = None
    organization_id: str | None = None


class _Permission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    apps: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _Connection:
    db: object


@dataclass(frozen=True, slots=True)
class AppAccess:
    """The apps a caller may read. `app_ids` is None for a proxy admin, who reads every app."""

    app_ids: frozenset[str] | None

    @property
    def unrestricted(self) -> bool:
        return self.app_ids is None

    def allows(self, app_id: str) -> bool:
        return self.app_ids is None or app_id in self.app_ids


ALL_APPS: Final = AppAccess(None)


def _grantees(rows: tuple[PrismaRecord, ...]) -> tuple[_Grantee, ...]:
    return tuple(_Grantee.model_validate(row.model_dump()) for row in rows)


async def granted_app_ids(db: object, user_id: str | None) -> frozenset[str]:
    """Apps granted through object_permission.apps on the user, the user's teams and the user's organizations."""
    if not user_id:
        return frozenset()
    connection: Final = _Connection(db)
    user_row: Final = await _UserRepository(connection).table.find_unique(where={"user_id": user_id})
    if user_row is None:
        return frozenset()
    user: Final = _Grantee.model_validate(user_row.model_dump())
    memberships: Final = _grantees(
        tuple(await _OrganizationMembershipRepository(connection).table.find_many(where={"user_id": user_id}))
    )
    teams: Final = _grantees(tuple(await find_many_in(_TeamRepository(connection).table, "team_id", user.teams or ())))
    organizations: Final = _grantees(
        tuple(
            await find_many_in(
                _OrganizationRepository(connection).table,
                "organization_id",
                tuple(row.organization_id for row in memberships if row.organization_id),
            )
        )
    )
    permission_ids: Final = tuple(
        dict.fromkeys(
            row.object_permission_id for row in (user, *teams, *organizations) if row.object_permission_id is not None
        )
    )
    permissions: Final = await find_many_in(
        _ObjectPermissionRepository(connection).table, "object_permission_id", permission_ids
    )
    return frozenset(
        chain.from_iterable(_Permission.model_validate(row.model_dump()).apps or () for row in permissions)
    )
