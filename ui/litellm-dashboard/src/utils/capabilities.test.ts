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

// Backend truth table for the `/global/spend/*` routes the Old Usage page calls
// (verified against a live proxy): only proxy_admin and proxy_admin_viewer are
// served. Org admins and team admins carry `internal_user` as their top-level
// user_role, so `effectiveSessionRole` renders them "Internal User" — an org
// admin never reaches the UI as "Org Admin" or `org_admin`.
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

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
