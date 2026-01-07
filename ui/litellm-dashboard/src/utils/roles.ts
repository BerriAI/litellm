import { Team } from "@/components/networking";

// Define admin roles and permissions
export const old_admin_roles = ["Admin", "Admin Viewer"];
export const v2_admin_role_names = ["proxy_admin", "proxy_admin_viewer", "org_admin"];
export const all_admin_roles = [...old_admin_roles, ...v2_admin_role_names];

export const internalUserRoles = ["Internal User", "Internal Viewer"];
export const rolesAllowedToSeeUsage = ["Admin", "Admin Viewer", "Internal User", "Internal Viewer"];
export const rolesWithWriteAccess = ["Internal User", "Admin", "proxy_admin"];

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
  return teams.some((team) => isUserTeamAdminForSingleTeam(team, userID));
};

export const isUserTeamAdminForSingleTeam = (team: Team | null, userID: string): boolean => {
  if (team == null || team.members_with_roles == null) {
    return false;
  }
  return team.members_with_roles.some((member) => member.user_id === userID && member.role === "admin");
};
