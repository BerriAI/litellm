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

describe("rolesWithCapability", () => {
  it("should return a copy so callers cannot mutate the capability map", () => {
    const roles = rolesWithCapability("viewToolPolicies");
    const removed = roles.pop();
    expect(hasCapability(removed, "viewToolPolicies")).toBe(true);
  });
});
