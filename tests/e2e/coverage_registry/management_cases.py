from dataclasses import dataclass
from typing import Final, Literal

CredentialKind = Literal["master", "idp_admin", "direct_jwt", "virtual_key", "dashboard_session"]
DependencyProfile = Literal["management_only", "real_oidc_browser", "external_provider_required"]


@dataclass(frozen=True, slots=True)
class ManagementCase:
    node: str
    credential_kind: CredentialKind
    actor: str
    profile: str
    method: Literal["GET", "POST"]
    path: str
    operation_family: str
    dependency_profile: DependencyProfile = "management_only"


JWT_FILE: Final = "tests/e2e/management/test_jwt_management_e2e.py"
JWT_CLASS: Final = f"{JWT_FILE}::TestJwtManagement"
ACTORS: Final = (
    "proxy_admin",
    "proxy_admin_viewer",
    "organization_admin",
    "team_admin",
    "team_member",
    "internal_user",
    "internal_user_viewer",
    "unrelated_user",
)
MANAGEMENT_CASES: Final = tuple(
    ManagementCase(
        node=f"{JWT_CLASS}::test_actor_subject_and_database_role[{role}]",
        credential_kind="direct_jwt",
        actor=role,
        profile="database_role",
        method="GET",
        path="/user/info",
        operation_family="identity",
    )
    for role in ACTORS
) + (
    ManagementCase(
        node=f"{JWT_CLASS}::test_admin_viewer_reads_but_cannot_update",
        credential_kind="direct_jwt",
        actor="proxy_admin_viewer",
        profile="database_role",
        method="POST",
        path="/key/update",
        operation_family="denial",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_admin_creates_reads_updates_clears_and_deletes_a_key[direct_jwt]",
        credential_kind="direct_jwt",
        actor="proxy_admin",
        profile="group_scoped",
        method="POST",
        path="/key/generate",
        operation_family="key_lifecycle",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_admin_creates_reads_updates_clears_and_deletes_a_key[virtual_key]",
        credential_kind="virtual_key",
        actor="proxy_admin",
        profile="group_scoped",
        method="POST",
        path="/key/generate",
        operation_family="key_lifecycle",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_two_actor_sets_keep_tenants_and_keys_isolated",
        credential_kind="direct_jwt",
        actor="team_member",
        profile="group_scoped",
        method="GET",
        path="/key/info",
        operation_family="tenant_isolation",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_member_cannot_write_and_another_team_cannot_read_the_key",
        credential_kind="direct_jwt",
        actor="team_member",
        profile="group_scoped",
        method="POST",
        path="/key/update",
        operation_family="tenant_isolation",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_multi_group_actor_keeps_exact_memberships",
        credential_kind="master",
        actor="bootstrap",
        profile="group_scoped",
        method="GET",
        path="/team/info",
        operation_family="memberships",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_successful_actor_cleanup_removes_owned_state",
        credential_kind="master",
        actor="bootstrap",
        profile="failure_cleanup",
        method="GET",
        path="/team/info",
        operation_family="cleanup",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_partial_setup_removes_previously_created_identities[group]",
        credential_kind="idp_admin",
        actor="idp_admin",
        profile="failure_cleanup",
        method="POST",
        path="/groups",
        operation_family="cleanup",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_partial_setup_removes_previously_created_identities[user]",
        credential_kind="idp_admin",
        actor="idp_admin",
        profile="failure_cleanup",
        method="POST",
        path="/users",
        operation_family="cleanup",
    ),
    ManagementCase(
        node=f"{JWT_CLASS}::test_oidc_browser_profile_identity_mapping",
        credential_kind="direct_jwt",
        actor="internal_user",
        profile="oidc_configuration",
        method="GET",
        path="/protocol/openid-connect/userinfo",
        operation_family="oidc_identity",
    ),
)


def canonical_node(node: str) -> str:
    return node if node.startswith("tests/e2e/") else f"tests/e2e/{node}"


def case_properties(node: str) -> tuple[tuple[str, str], ...]:
    case: Final = next((case for case in MANAGEMENT_CASES if case.node == canonical_node(node)), None)
    if case is None:
        return ()
    return (
        ("management_node", case.node),
        ("credential_kind", case.credential_kind),
        ("actor", case.actor),
        ("auth_profile", case.profile),
        ("dependency_profile", case.dependency_profile),
    )
