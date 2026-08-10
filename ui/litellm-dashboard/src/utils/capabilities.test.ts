import { describe, expect, it } from "vitest";

import { hasCapability, rolesWithCapability, type Capability } from "./capabilities";

const ADMIN_ROLES = ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"];
const NON_ADMIN_ROLES = [
  "Internal User",
  "Internal Viewer",
  "App User",
  "Org Admin",
  "Unknown Role",
  "",
  null,
  undefined,
];

const ADMIN_ONLY_CAPABILITIES: Capability[] = ["viewToolPolicies", "viewOrganizationUsage", "viewAgentUsage"];

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

describe.each(["viewAuditLogs", "viewDeletedTeams"] as const)("hasCapability - %s", (capability) => {
  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])("should grant it to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(true);
  });

  it.each(["Internal User", "Internal Viewer", "App User", "Org Admin", "Unknown Role", "", null, undefined])(
    "should deny it to %s",
    (role) => {
      expect(hasCapability(role, capability)).toBe(false);
    },
  );
});

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
