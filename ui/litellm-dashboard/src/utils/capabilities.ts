import { all_admin_roles, old_admin_roles } from "./roles";

const proxyAdminOnlyRoles = [...old_admin_roles, "proxy_admin", "proxy_admin_viewer"];

const CAPABILITY_ROLES = {
  viewToolPolicies: all_admin_roles,
  viewAuditLogs: all_admin_roles,
  viewDeletedTeams: all_admin_roles,
  viewPolicies: all_admin_roles,
  viewPrompts: all_admin_roles,
  viewOrganizationUsage: all_admin_roles,
  viewAgentUsage: all_admin_roles,
  viewGlobalSpend: proxyAdminOnlyRoles,
} as const satisfies Record<string, readonly string[]>;

export type Capability = keyof typeof CAPABILITY_ROLES;

export const hasCapability = (userRole: string | null | undefined, capability: Capability): boolean =>
  userRole != null && CAPABILITY_ROLES[capability].includes(userRole);

export const rolesWithCapability = (capability: Capability): string[] => [...CAPABILITY_ROLES[capability]];
