import { describe, expect, it } from "vitest";

import { Capability, hasCapability, rolesWithCapability } from "./capabilities";

// `/health/readiness/details`, `/health/license` and `/api/plugins` are all
// proxy-admin routes on the backend: org admins get a 401 there, unlike the
// broader `all_admin_roles` set that gates tool policies.
const proxyAdminOnlyCapabilities: Capability[] = ["viewProxyDiagnostics", "viewLicenseInfo", "viewPlugins"];

describe.each(proxyAdminOnlyCapabilities)("hasCapability(%s)", (capability) => {
  it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])("should grant to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(true);
  });

  it.each([
    "Org Admin",
    "org_admin",
    "Internal User",
    "internal_user",
    "Internal Viewer",
    "internal_user_viewer",
    "App User",
    "app_user",
    "Unknown Role",
    "",
    null,
    undefined,
  ])("should deny to %s", (role) => {
    expect(hasCapability(role, capability)).toBe(false);
  });
});

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
});

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
