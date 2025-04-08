// Define admin roles and permissions
export const old_admin_roles = ["Admin", "Admin Viewer"];
export const v2_admin_role_names = ["proxy_admin", "proxy_admin_viewer", "org_admin"];
export const all_admin_roles = [...old_admin_roles, ...v2_admin_role_names];

export const internalUserRoles = ["Internal User", "Internal Viewer"];
export const rolesAllowedToSeeUsage = ["Admin", "Admin Viewer", "Internal User", "Internal Viewer"]; 
export const rolesWithWriteAccess = ["Internal User", "Admin"]; 