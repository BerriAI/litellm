from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from idp import ADMIN_CLIENT_ID, TESTS_CLIENT_ID, Identity, Keycloak
from lifecycle import ResourceManager
from management.management_client import ManagementClient
from models import (
    KeyGenerateBody,
    KeyGenerateResponse,
    OrgDeleteBody,
    OrgDeleteResponse,
    OrgMemberAddBody,
    OrgMemberEntry,
    OrgNewBody,
    TeamDeleteBody,
    TeamMemberAddBody,
    TeamMemberEntry,
    TeamNewBody,
    UserNewBody,
    UserRole,
)
from proxy_client import Caller

ActorRole = Literal[
    "proxy_admin",
    "proxy_admin_viewer",
    "organization_admin",
    "team_admin",
    "team_member",
    "internal_user",
    "internal_user_viewer",
    "unrelated_user",
]
ActorProfile = Literal["database_role", "group_scoped"]


@dataclass(frozen=True, slots=True)
class Tenant:
    organization_id: str
    team_id: str
    group_id: str


@dataclass(frozen=True, slots=True)
class Actor:
    identity: Identity
    role: ActorRole
    global_role: UserRole
    profile: ActorProfile
    tenants: tuple[Tenant, ...]

    def mint_caller(self, idp: Keycloak) -> Caller:
        return Caller(
            credential=idp.access_token(
                self.identity, client_id=ADMIN_CLIENT_ID if self.role == "proxy_admin" else TESTS_CLIENT_ID
            ),
            kind="direct_jwt",
            role=self.role,
            tenant=self.tenants[0].organization_id if self.tenants else None,
        )


@dataclass(frozen=True, slots=True)
class ActorFactory:
    bootstrap: ManagementClient
    idp: Keycloak
    resources: ResourceManager

    def __post_init__(self) -> None:
        if self.bootstrap.proxy.caller is not None:
            raise ValueError("Actor bootstrap requires a separately held master client")

    def key(self, tenant: Tenant | None = None, *, user_id: str | None = None) -> KeyGenerateResponse:
        created: Final = unwrap(
            self.bootstrap.generate_key(
                KeyGenerateBody(
                    team_id=tenant.team_id if tenant is not None else None,
                    user_id=user_id,
                    key_alias=f"e2e-actor-key-{unique_marker()}",
                )
            )
        )
        self.resources.defer(lambda: self.bootstrap.delete_key_strict(created.key))
        return created

    def tenant(self) -> Tenant:
        marker: Final = unique_marker()
        organization_id: Final = self.bootstrap.create_org(OrgNewBody(organization_alias=f"e2e-organization-{marker}"))
        self.resources.defer(
            lambda: unwrap(
                self.bootstrap.proxy.transport.delete(
                    "/organization/delete",
                    headers=self.bootstrap.proxy.management_headers(),
                    json=OrgDeleteBody(organization_ids=[organization_id]),
                    response_type=OrgDeleteResponse,
                )
            )
        )
        team_id: Final = self.bootstrap.proxy.create_team(
            TeamNewBody(team_alias=f"e2e-team-{marker}", organization_id=organization_id)
        )
        self.resources.defer(
            lambda: unwrap(
                self.bootstrap.proxy.transport.post(
                    "/team/delete",
                    headers=self.bootstrap.proxy.management_headers(),
                    json=TeamDeleteBody(team_ids=[team_id]),
                    response_type=NoBody,
                )
            )
        )
        self.bootstrap.delete_team_member(team_id, self.bootstrap.user_info().user_id)
        group_id: Final = self.idp.create_group(team_id)
        self.resources.defer(lambda: self.idp.with_strict_cleanup().delete_group(group_id))
        return Tenant(organization_id=organization_id, team_id=team_id, group_id=group_id)

    def create(
        self, role: ActorRole, *, tenants: tuple[Tenant, ...] = (), profile: ActorProfile = "database_role"
    ) -> Actor:
        if role in ("team_admin", "team_member", "organization_admin") and not tenants:
            raise ValueError("A membership actor requires a tenant")
        identity: Final = self.idp.with_strict_cleanup().provision_user(
            marker=unique_marker(),
            groups=tuple(tenant.team_id for tenant in tenants) if profile == "group_scoped" else (),
            group_ids=tuple(tenant.group_id for tenant in tenants) if profile == "group_scoped" else (),
            defer=self.resources.defer,
        )
        global_role: Final[UserRole] = (
            role
            if role in ("proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer")
            else "internal_user"
        )
        self.bootstrap.create_user(
            UserNewBody(
                user_id=identity.user_id,
                user_email=f"{identity.username}@example.com",
                user_role=global_role,
                auto_create_key=False,
            )
        )
        self.resources.defer(lambda: self.bootstrap.delete_user_strict(identity.user_id))
        for tenant in tenants:
            unwrap(
                self.bootstrap.proxy.transport.post(
                    "/organization/member_add",
                    headers=self.bootstrap.proxy.management_headers(),
                    json=OrgMemberAddBody(
                        organization_id=tenant.organization_id,
                        member=OrgMemberEntry(
                            user_id=identity.user_id,
                            role="org_admin" if role == "organization_admin" else "internal_user",
                        ),
                    ),
                    response_type=NoBody,
                )
            )
            unwrap(
                self.bootstrap.proxy.transport.post(
                    "/team/member_add",
                    headers=self.bootstrap.proxy.management_headers(),
                    json=TeamMemberAddBody(
                        team_id=tenant.team_id,
                        member=TeamMemberEntry(
                            user_id=identity.user_id,
                            role="admin" if role == "team_admin" else "user",
                        ),
                    ),
                    response_type=NoBody,
                )
            )
        return Actor(identity=identity, role=role, global_role=global_role, profile=profile, tenants=tenants)
