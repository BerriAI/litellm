import { Member, Team } from "@/components/networking";

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
export const rolesAllowedToViewWriteScopedPages = [
  ...rolesWithWriteAccess,
  "Admin Viewer",
  "proxy_admin_viewer",
];

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

export const formatUserRole = (userRole: string): string => {
  if (!userRole) {
    return "Undefined Role";
  }
  switch (userRole.toLowerCase()) {
    case "app_owner":
      return "App Owner";
    case "demo_app_owner":
      return "App Owner";
    case "app_admin":
      return "Admin";
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
