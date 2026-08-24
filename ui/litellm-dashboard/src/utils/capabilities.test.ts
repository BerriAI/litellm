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

const SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES = "Internal User";

const ORG_ADMIN_BACKEND_ACCESS: ReadonlyArray<readonly [Capability, string, boolean]> = [
  ["viewDeletedTeams", "GET /v2/team/list?status=deleted -> 200 (scoped to their orgs)", true],
  ["viewOrganizationUsage", "GET /organization/daily/activity -> 200 (scoped to orgs they administer)", true],
  ["viewToolPolicies", "GET /v1/tool/list -> 401", false],
  ["viewPolicies", "GET /policies/list -> 401", false],
  ["viewPrompts", "GET /prompts/list -> 401", false],
  ["viewAuditLogs", "GET /audit -> 401", false],
];

describe("hasCapability for organization admins", () => {
  it.each(ORG_ADMIN_BACKEND_ACCESS)("%s matches the backend: %s", (capability, _endpoint, isEntitled) => {
    expect(hasCapability(SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES, capability, true)).toBe(isEntitled);
  });

  it.each(NON_ADMIN_ROLES)("grants viewDeletedTeams to an org admin whose session role is %s", (role) => {
    expect(hasCapability(role, "viewDeletedTeams", true)).toBe(true);
  });

  // An org admin is an internal_user whose org-admin-ness lives in the
  // membership table, so their session role never distinguishes them. Gating
  // the Organization Usage view on the session role alone hid the whole view
  // from them and left the tab's data fetch disabled.
  it.each(NON_ADMIN_ROLES)("grants viewOrganizationUsage to an org admin whose session role is %s", (role) => {
    expect(hasCapability(role, "viewOrganizationUsage", true)).toBe(true);
  });

  it.each(ADMIN_ONLY_CAPABILITIES)("leaves %s denied when the caller is not an org admin", (capability) => {
    expect(hasCapability(SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES, capability, false)).toBe(false);
    expect(hasCapability(SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES, capability)).toBe(false);
  });

  it("keeps the org-admin allowance opt-in per capability", () => {
    const orgAdminCapabilities = ADMIN_ONLY_CAPABILITIES.filter((capability) =>
      hasCapability(SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES, capability, true),
    );
    expect(orgAdminCapabilities).toEqual(["viewDeletedTeams", "viewOrganizationUsage"]);
  });

  it("does not let the org-admin allowance reopen the proxy-admin-only viewGlobalSpend gate", () => {
    expect(hasCapability(SESSION_ROLE_AN_ORG_ADMIN_ACTUALLY_CARRIES, "viewGlobalSpend", true)).toBe(false);
    expect(hasCapability("Org Admin", "viewGlobalSpend", true)).toBe(false);
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
