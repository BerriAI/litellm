import { describe, it, expect } from "vitest";
import { isAdminRole, isProxyAdminRole } from "./roles";

describe("roles", () => {
  describe("isAdminRole", () => {
    it("should return true for all admin roles", () => {
      expect(isAdminRole("Admin")).toBe(true);
      expect(isAdminRole("Admin Viewer")).toBe(true);
      expect(isAdminRole("proxy_admin")).toBe(true);
      expect(isAdminRole("proxy_admin_viewer")).toBe(true);
      expect(isAdminRole("org_admin")).toBe(true);
    });

    it("should return false for non-admin roles", () => {
      expect(isAdminRole("Internal User")).toBe(false);
      expect(isAdminRole("Internal Viewer")).toBe(false);
      expect(isAdminRole("regular_user")).toBe(false);
      expect(isAdminRole("")).toBe(false);
    });
  });

  describe("isProxyAdminRole", () => {
    it("should return true for proxy_admin and Admin roles", () => {
      expect(isProxyAdminRole("proxy_admin")).toBe(true);
      expect(isProxyAdminRole("Admin")).toBe(true);
    });

    it("should return false for other admin roles", () => {
      expect(isProxyAdminRole("Admin Viewer")).toBe(false);
      expect(isProxyAdminRole("proxy_admin_viewer")).toBe(false);
      expect(isProxyAdminRole("org_admin")).toBe(false);
    });

    it("should return false for non-admin roles", () => {
      expect(isProxyAdminRole("Internal User")).toBe(false);
      expect(isProxyAdminRole("Internal Viewer")).toBe(false);
      expect(isProxyAdminRole("regular_user")).toBe(false);
      expect(isProxyAdminRole("")).toBe(false);
    });
  });
});
