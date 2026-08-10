import { describe, expect, it } from "vitest";

import { hasCapability, rolesWithCapability, type Capability } from "./capabilities";

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

// An org admin is a membership, not a session role, so their JWT reads "Internal User".
// Each row was measured on a live proxy with a membership-granted org admin's key.
const ORG_ADMIN_BACKEND_ACCESS: ReadonlyArray<readonly [Capability, string, boolean]> = [
  ["viewDeletedTeams", "GET /v2/team/list?status=deleted -> 200 (scoped to their orgs)", true],
  ["viewToolPolicies", "GET /v1/tool/list -> 401", false],
  ["viewPolicies", "GET /policies/list -> 401", false],
  ["viewPrompts", "GET /prompts/list -> 401", false],
  ["viewAuditLogs", "GET /audit -> 401", false],
];

describe("hasCapability for organization admins", () => {
  it.each(ORG_ADMIN_BACKEND_ACCESS)("%s matches the backend: %s", (capability, _endpoint, isEntitled) => {
    expect(hasCapability("Internal User", capability, true)).toBe(isEntitled);
  });

  it.each(NON_ADMIN_ROLES)("grants viewDeletedTeams to an org admin whose session role is %s", (role) => {
    expect(hasCapability(role, "viewDeletedTeams", true)).toBe(true);
  });

  it.each(ADMIN_ONLY_CAPABILITIES)("leaves %s denied when the caller is not an org admin", (capability) => {
    expect(hasCapability("Internal User", capability, false)).toBe(false);
    expect(hasCapability("Internal User", capability)).toBe(false);
  });

  it("keeps the org-admin allowance opt-in per capability", () => {
    const orgAdminCapabilities = ADMIN_ONLY_CAPABILITIES.filter((capability) =>
      hasCapability("Internal User", capability, true),
    );
    expect(orgAdminCapabilities).toEqual(["viewDeletedTeams"]);
  });
});

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
