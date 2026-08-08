import { all_admin_roles, old_admin_roles } from "./roles";

const proxyAdminOnlyRoles = [...old_admin_roles, "proxy_admin", "proxy_admin_viewer"];

const CAPABILITY_ROLES = {
  viewToolPolicies: all_admin_roles,
  viewProxyDiagnostics: proxyAdminOnlyRoles,
  viewLicenseInfo: proxyAdminOnlyRoles,
  viewPlugins: proxyAdminOnlyRoles,
} as const satisfies Record<string, readonly string[]>;

export type Capability = keyof typeof CAPABILITY_ROLES;

export const hasCapability = (userRole: string | null | undefined, capability: Capability): boolean =>
  userRole != null && CAPABILITY_ROLES[capability].includes(userRole);

export const rolesWithCapability = (capability: Capability): string[] => [...CAPABILITY_ROLES[capability]];
