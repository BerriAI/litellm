import { Member, Organization, Team } from "@/components/networking";

const ORG_ADMIN_MEMBERSHIP_ROLE = "org_admin";

interface OrganizationMembership {
  user_id?: string | null;
  user_role?: string | null;
}

// Define admin roles and permissions
export const old_admin_roles = ["Admin", "Admin Viewer"];
export const v2_admin_role_names = ["proxy_admin", "proxy_admin_viewer", "org_admin"];
export const all_admin_roles = [...old_admin_roles, ...v2_admin_role_names];

export const internalUserRoles = ["Internal User", "Internal Viewer", "internal_user", "internal_user_viewer"];
export const rolesAllowedToSeeUsage = ["Admin", "Admin Viewer", "Internal User", "Internal Viewer"];
export const rolesWithWriteAccess = ["Internal User", "Admin", "proxy_admin"];
// Admin-tier read parity: Admin Viewer sees Models + Endpoints, Agents, and
// other pages whose primary purpose is configuration/management read-only.
// Per the Admin Viewer principle: read parity with Proxy Admin, no writes,
// no cost-incurring actions (Playground stays gated by `rolesWithWriteAccess`).
export const rolesAllowedToViewWriteScopedPages = [...rolesWithWriteAccess, "Admin Viewer", "proxy_admin_viewer"];
export const viewOnlyRoles = ["Admin Viewer", "Internal Viewer"];
export const isViewOnlyRole = (role: string): boolean => viewOnlyRoles.includes(role);

// Helper function to check if a role is in all_admin_roles
export const isAdminRole = (role: string): boolean => {
  return all_admin_roles.includes(role);
};

export const isProxyAdminRole = (role: string): boolean => {
  return role === "proxy_admin" || role === "Admin";
};

export const isUserTeamAdminForAnyTeam = (teams: Team[] | null, userID: string): boolean => {
  if (teams == null) {
    return false;
  }
  return teams.some((team) => isUserTeamAdminForSingleTeam(team.members_with_roles, userID));
};

export const isUserTeamAdminForSingleTeam = (teamMemberWithRoles: Member[] | null, userID: string): boolean => {
  if (teamMemberWithRoles == null) {
    return false;
  }
  return teamMemberWithRoles.some((member) => member.user_id === userID && member.role === "admin");
};

export const isOrgAdminForAnyOrg = (
  organizations: Organization[] | null | undefined,
  userID: string | null | undefined,
): boolean => {
  if (organizations == null || !userID) {
    return false;
  }
  return organizations.some((org) => {
    const members: OrganizationMembership[] = org.members ?? [];
    return members.some((member) => member.user_id === userID && member.user_role === ORG_ADMIN_MEMBERSHIP_ROLE);
  });
};

export const formatUserRole = (userRole: string): string => {
  if (!userRole) {
    return "Undefined Role";
  }
  switch (userRole.toLowerCase()) {
    case "app_owner":
      return "App Owner";
    case "demo_app_owner":
      return "App Owner";
    case "proxy_admin":
      return "Admin";
    case "proxy_admin_viewer":
      return "Admin Viewer";
    case "org_admin":
      return "Org Admin";
    case "internal_user":
      return "Internal User";
    case "internal_user_viewer":
    case "internal_viewer": // TODO:remove if deprecated
      return "Internal Viewer";
    case "app_user":
      return "App User";
    default:
      return "Unknown Role";
  }
};

export const isOrgAdminSessionRole = (userRole?: string | null): boolean =>
  userRole === ORG_ADMIN_MEMBERSHIP_ROLE || userRole === formatUserRole(ORG_ADMIN_MEMBERSHIP_ROLE);

const viewOnlyRawRoles = ["proxy_admin_viewer", "internal_user_viewer", "internal_viewer"];

export const effectiveSessionRole = (rawUserRole?: string): string => {
  if (rawUserRole?.toLowerCase() === "proxy_admin_viewer") {
    return "Admin";
  }
  return formatUserRole(rawUserRole ?? "");
};

export const isViewOnlySessionRole = (rawUserRole?: string): boolean =>
  viewOnlyRawRoles.includes(rawUserRole?.toLowerCase() ?? "");

// Session roles (the value `useAuthorized().userRole` supplies) that /team/list and
// /v2/team/list already answer with a broad list: proxy-wide for admins, org-wide for
// org admins. Sending a user_id for those narrows the response to direct memberships,
// so only the roles the endpoints would otherwise reject carry one.
const sessionRolesWithBroadTeamList: string[] = ["Admin", "Admin Viewer", "Org Admin"];

export const teamListScopeUserId = (userRole: string | null, userId: string | null): string | null =>
  sessionRolesWithBroadTeamList.includes(userRole ?? "") ? null : userId;

// Mirrors the backend's `user_api_key_has_admin_view`, which the daily-activity endpoints gate on:
// proxy admins read every user's spend, and everyone else is forced to their own user_id. Org admin
// is deliberately absent, so this is `all_admin_roles` minus org admin rather than a reuse of it.
// Both spellings are listed because that constant carries both and either may reach a call site.
const rolesWithProxyWideSpendView: string[] = ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"];

export const hasProxyWideSpendView = (userRole: string | null): boolean =>
  rolesWithProxyWideSpendView.includes(userRole ?? "");

export const spendScopeUserId = (userRole: string | null, userId: string | null): string | null =>
  hasProxyWideSpendView(userRole) ? null : userId;
