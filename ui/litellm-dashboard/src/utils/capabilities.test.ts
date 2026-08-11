import { describe, expect, it } from "vitest";

import { hasCapability, rolesWithCapability, type Capability } from "./capabilities";
import { effectiveSessionRole } from "./roles";

const ADMIN_ROLES = ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"];
const NON_ADMIN_ROLES = [
  "Internal User",
  "Internal Viewer",
  "internal_user",
  "App User",
  "Org Admin",
  "Unknown Role",
  "",
  null,
  undefined,
];

const ADMIN_ONLY_CAPABILITIES: Capability[] = [
  "viewToolPolicies",
  "viewAuditLogs",
  "viewDeletedTeams",
  "viewPolicies",
  "viewPrompts",
  "viewOrganizationUsage",
  "viewAgentUsage",
];

const PROXY_ADMIN_ONLY_PAGE_CAPABILITIES: Capability[] = [
  "viewWorkflowRuns",
  "viewMemory",
  "viewGuardrailUsage",
  "viewProxyWideCostData",
];

describe("hasCapability", () => {
  describe.each(ADMIN_ONLY_CAPABILITIES)("%s", (capability) => {
    it.each(ADMIN_ROLES)("should grant it to %s", (role) => {
      expect(hasCapability(role, capability)).toBe(true);
    });

    it.each(NON_ADMIN_ROLES)("should deny it to %s", (role) => {
      expect(hasCapability(role, capability)).toBe(false);
    });
  });
});

describe("hasCapability - viewGlobalSpend", () => {
  it.each(ADMIN_ROLES)("should grant it to %s", (role) => {
    expect(hasCapability(role, "viewGlobalSpend")).toBe(true);
  });

  it.each([...NON_ADMIN_ROLES, "internal_user_viewer", "org_admin"])("should deny it to %s", (role) => {
    expect(hasCapability(role, "viewGlobalSpend")).toBe(false);
  });

  it("should deny it to every role an org admin or team admin can present at runtime", () => {
    const orgAdminSessionRole = effectiveSessionRole("internal_user");
    const teamAdminSessionRole = effectiveSessionRole("internal_user");

    expect(orgAdminSessionRole).toBe("Internal User");
    expect(hasCapability(orgAdminSessionRole, "viewGlobalSpend")).toBe(false);
    expect(hasCapability(teamAdminSessionRole, "viewGlobalSpend")).toBe(false);
  });

  it.each([
    ["proxy_admin", true],
    ["proxy_admin_viewer", true],
    ["internal_user", false],
    ["internal_user_viewer", false],
  ] as const)("should match the backend for a %s session", (rawRole, expected) => {
    expect(hasCapability(effectiveSessionRole(rawRole), "viewGlobalSpend")).toBe(expected);
  });
});

// The four sidebar pages behind these capabilities call routes the proxy serves
// to proxy_admin and proxy_admin_viewer only. An org admin is denied there too,
// because `_user_is_org_admin` needs an organization_id that a page-load GET
// never carries, so `org_admin` must not grant them either.
describe.each(PROXY_ADMIN_ONLY_PAGE_CAPABILITIES)("hasCapability - %s", (capability) => {
  it.each(ADMIN_ROLES)("should grant it to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(true);
  });

  it.each([...NON_ADMIN_ROLES, "internal_user_viewer", "org_admin"])("should deny it to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(false);
  });

  it.each([
    ["proxy_admin", true],
    ["proxy_admin_viewer", true],
    ["org_admin", false],
    ["internal_user", false],
    ["internal_user_viewer", false],
  ] as const)("should match the backend for a %s session", (rawRole, expected) => {
    expect(hasCapability(effectiveSessionRole(rawRole), capability)).toBe(expected);
  });
});

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
