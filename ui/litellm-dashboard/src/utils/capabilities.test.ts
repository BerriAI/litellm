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
  "viewProxyConfig",
  "viewCredentials",
];

// `/config/list` sits in `admin_viewer_routes`, which `org_admin_allowed_routes`
// includes, so an org admin looks allowed on paper. At runtime `_user_is_org_admin`
// needs an `organization_id` in the request payload, which a bare GET never carries,
// so the proxy answers 401 for org admins on both of these routes.
const PROXY_ADMIN_ONLY_CAPABILITIES: Capability[] = ["viewProxyConfig", "viewCredentials"];

// A team admin carries no distinct session role; the dashboard sees their user_role.
const TEAM_ADMIN_SESSION_ROLE = "Internal User";
const ORG_ADMIN_ROLES = ["Org Admin", "org_admin"];

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

describe.each(PROXY_ADMIN_ONLY_CAPABILITIES)("hasCapability - %s", (capability) => {
  it.each(ORG_ADMIN_ROLES)("should deny it to an org admin (%s)", (role) => {
    expect(hasCapability(role, capability)).toBe(false);
  });

  it("should deny it to a team admin", () => {
    expect(hasCapability(TEAM_ADMIN_SESSION_ROLE, capability)).toBe(false);
  });

  it.each(["internal_user", "internal_user_viewer", "Internal Viewer"])("should deny it to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(false);
  });

  it.each(["Admin", "proxy_admin", "Admin Viewer", "proxy_admin_viewer"])("should grant it to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(true);
  });
});

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
