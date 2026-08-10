import { describe, expect, it } from "vitest";

import { hasCapability, rolesWithCapability } from "./capabilities";

describe("hasCapability", () => {
  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])(
    "should grant viewToolPolicies to %s",
    (role) => {
      expect(hasCapability(role, "viewToolPolicies")).toBe(true);
    },
  );

  it.each(["Internal User", "Internal Viewer", "App User", "Org Admin", "Unknown Role", "", null, undefined])(
    "should deny viewToolPolicies to %s",
    (role) => {
      expect(hasCapability(role, "viewToolPolicies")).toBe(false);
    },
  );

  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])("should grant viewPolicies to %s", (role) => {
    expect(hasCapability(role, "viewPolicies")).toBe(true);
  });

  it.each([
    "Internal User",
    "Internal Viewer",
    "internal_user",
    "App User",
    "Org Admin",
    "Unknown Role",
    "",
    null,
    undefined,
  ])("should deny viewPolicies to %s", (role) => {
    expect(hasCapability(role, "viewPolicies")).toBe(false);
  });

  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])("should grant viewPrompts to %s", (role) => {
    expect(hasCapability(role, "viewPrompts")).toBe(true);
  });

  it.each([
    "Internal User",
    "Internal Viewer",
    "internal_user",
    "App User",
    "Org Admin",
    "Unknown Role",
    "",
    null,
    undefined,
  ])("should deny viewPrompts to %s", (role) => {
    expect(hasCapability(role, "viewPrompts")).toBe(false);
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

// `useAuthorized` supplies `userRole` as the formatted session role from
// `effectiveSessionRole`, which collapses proxy_admin_viewer to "Admin" and
// renders an org admin as "Org Admin". The four sidebar pages behind these
// capabilities call proxy-admin-only routes: `_user_is_org_admin` needs an
// `organization_id` in the request data, which a page-load GET never carries,
// so an org admin is denied at the proxy exactly as it is here.
describe.each(["viewWorkflowRuns", "viewMemory", "viewGuardrailUsage", "viewProxyWideCostData"] as const)(
  "hasCapability - %s",
  (capability) => {
    it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])("should grant it to %s", (role) => {
      expect(hasCapability(role, capability)).toBe(true);
    });

    it.each([
      "Internal User",
      "Internal Viewer",
      "internal_user",
      "internal_user_viewer",
      "Org Admin",
      "App User",
      "Unknown Role",
      "",
      null,
      undefined,
    ])("should deny it to %s", (role) => {
      expect(hasCapability(role, capability)).toBe(false);
    });
  },
);

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
