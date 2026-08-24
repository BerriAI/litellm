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
  viewWorkflowRuns: proxyAdminOnlyRoles,
  viewMemory: proxyAdminOnlyRoles,
  viewGuardrailUsage: proxyAdminOnlyRoles,
  viewProxyWideCostData: proxyAdminOnlyRoles,
} as const satisfies Record<string, readonly string[]>;

export type Capability = keyof typeof CAPABILITY_ROLES;

const ORG_ADMIN_CAPABILITIES: ReadonlySet<Capability> = new Set<Capability>([
  "viewDeletedTeams",
  "viewOrganizationUsage",
]);

export const hasCapability = (
  userRole: string | null | undefined,
  capability: Capability,
  isOrgAdmin: boolean = false,
): boolean =>
  (isOrgAdmin && ORG_ADMIN_CAPABILITIES.has(capability)) ||
  (userRole != null && CAPABILITY_ROLES[capability].includes(userRole));

export const rolesWithCapability = (capability: Capability): string[] => [...CAPABILITY_ROLES[capability]];
