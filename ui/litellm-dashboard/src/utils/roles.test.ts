import { describe, it, expect } from "vitest";
import {
  effectiveSessionRole,
  isAdminRole,
  isProxyAdminRole,
  isUserTeamAdminForAnyTeam,
  isUserTeamAdminForSingleTeam,
  isViewOnlySessionRole,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
} from "./roles";
import { Team } from "@/components/networking";

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

  describe("isUserTeamAdminForSingleTeam", () => {
    it("should return true when user is team admin", () => {
      const members_with_roles = [
        { user_id: "user-1", user_email: "user1@test.com", role: "admin" },
        { user_id: "user-2", user_email: "user2@test.com", role: "user" },
      ];
      expect(isUserTeamAdminForSingleTeam(members_with_roles, "user-1")).toBe(true);
    });

    it("should return false when user is not team admin", () => {
      const members_with_roles = [
        { user_id: "user-1", user_email: "user1@test.com", role: "user" },
        { user_id: "user-2", user_email: "user2@test.com", role: "user" },
      ];
      expect(isUserTeamAdminForSingleTeam(members_with_roles, "user-1")).toBe(false);
    });

    it("should return false when user is not in team", () => {
      const members_with_roles = [{ user_id: "user-2", user_email: "user2@test.com", role: "admin" }];
      expect(isUserTeamAdminForSingleTeam(members_with_roles, "user-1")).toBe(false);
    });

    it("should return false when members_with_roles is null", () => {
      expect(isUserTeamAdminForSingleTeam(null, "user-1")).toBe(false);
    });

    it("should return false when members_with_roles is empty array", () => {
      expect(isUserTeamAdminForSingleTeam([], "user-1")).toBe(false);
    });
  });

  describe("isUserTeamAdminForAnyTeam", () => {
    it("should return true when user is admin of at least one team", () => {
      const teams: Team[] = [
        {
          team_id: "team-1",
          team_alias: "Test Team 1",
          models: [],
          max_budget: null,
          budget_duration: null,
          tpm_limit: null,
          rpm_limit: null,
          organization_id: "org-1",
          created_at: "2024-01-01",
          keys: [],
          members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "user" }],
        },
        {
          team_id: "team-2",
          team_alias: "Test Team 2",
          models: [],
          max_budget: null,
          budget_duration: null,
          tpm_limit: null,
          rpm_limit: null,
          organization_id: "org-1",
          created_at: "2024-01-01",
          keys: [],
          members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "admin" }],
        },
      ];
      expect(isUserTeamAdminForAnyTeam(teams, "user-1")).toBe(true);
    });

    it("should return false when user is not admin of any team", () => {
      const teams: Team[] = [
        {
          team_id: "team-1",
          team_alias: "Test Team 1",
          models: [],
          max_budget: null,
          budget_duration: null,
          tpm_limit: null,
          rpm_limit: null,
          organization_id: "org-1",
          created_at: "2024-01-01",
          keys: [],
          members_with_roles: [{ user_id: "user-1", user_email: "user1@test.com", role: "user" }],
        },
        {
          team_id: "team-2",
          team_alias: "Test Team 2",
          models: [],
          max_budget: null,
          budget_duration: null,
          tpm_limit: null,
          rpm_limit: null,
          organization_id: "org-1",
          created_at: "2024-01-01",
          keys: [],
          members_with_roles: [{ user_id: "user-2", user_email: "user2@test.com", role: "admin" }],
        },
      ];
      expect(isUserTeamAdminForAnyTeam(teams, "user-1")).toBe(false);
    });

    it("should return false when teams is null", () => {
      expect(isUserTeamAdminForAnyTeam(null, "user-1")).toBe(false);
    });

    it("should return false when teams is empty array", () => {
      expect(isUserTeamAdminForAnyTeam([], "user-1")).toBe(false);
    });
  });

  describe("rolesAllowedToViewWriteScopedPages", () => {
    it("includes Admin Viewer (both display and stored forms)", () => {
      // Admin Viewer follows the read-parity rule — they must be able to
      // see Models + Endpoints and Agents read-only.
      expect(rolesAllowedToViewWriteScopedPages).toContain("Admin Viewer");
      expect(rolesAllowedToViewWriteScopedPages).toContain("proxy_admin_viewer");
    });

    it("includes everything in rolesWithWriteAccess (read parity is a superset)", () => {
      for (const role of rolesWithWriteAccess) {
        expect(rolesAllowedToViewWriteScopedPages).toContain(role);
      }
    });

    it("is a strict superset of rolesWithWriteAccess", () => {
      // Admin Viewer is added on top — the new set must be larger than
      // the write-only set, otherwise the constant has no purpose.
      expect(rolesAllowedToViewWriteScopedPages.length).toBeGreaterThan(rolesWithWriteAccess.length);
    });
  });

  describe("effectiveSessionRole", () => {
    it("normalizes proxy_admin_viewer to Admin", () => {
      expect(effectiveSessionRole("proxy_admin_viewer")).toBe("Admin");
    });

    it("keeps proxy_admin as Admin", () => {
      expect(effectiveSessionRole("proxy_admin")).toBe("Admin");
    });

    it("gives proxy_admin_viewer the same session role as proxy_admin", () => {
      expect(effectiveSessionRole("proxy_admin_viewer")).toBe(effectiveSessionRole("proxy_admin"));
    });

    it("lets a normalized proxy_admin_viewer pass admin-tier role gates", () => {
      expect(rolesWithWriteAccess).toContain(effectiveSessionRole("proxy_admin_viewer"));
    });

    it("does not collapse internal_user_viewer into an admin role", () => {
      expect(effectiveSessionRole("internal_user_viewer")).toBe("Internal Viewer");
      expect(rolesWithWriteAccess).not.toContain(effectiveSessionRole("internal_user_viewer"));
    });

    it("leaves other roles untouched", () => {
      expect(effectiveSessionRole("internal_user")).toBe("Internal User");
      expect(effectiveSessionRole("org_admin")).toBe("Org Admin");
    });

    it("returns Undefined Role for a missing role", () => {
      expect(effectiveSessionRole(undefined)).toBe("Undefined Role");
      expect(effectiveSessionRole("")).toBe("Undefined Role");
    });
  });

  describe("isViewOnlySessionRole", () => {
    it("returns true for proxy_admin_viewer", () => {
      expect(isViewOnlySessionRole("proxy_admin_viewer")).toBe(true);
    });

    it("returns false for proxy_admin", () => {
      expect(isViewOnlySessionRole("proxy_admin")).toBe(false);
    });

    it("returns true for internal_user_viewer", () => {
      expect(isViewOnlySessionRole("internal_user_viewer")).toBe(true);
    });

    it("returns false for internal_user and org_admin", () => {
      expect(isViewOnlySessionRole("internal_user")).toBe(false);
      expect(isViewOnlySessionRole("org_admin")).toBe(false);
    });

    it("returns false for a missing role", () => {
      expect(isViewOnlySessionRole(undefined)).toBe(false);
      expect(isViewOnlySessionRole("")).toBe(false);
    });

    it("stays true for proxy_admin_viewer even though its session role reads as Admin", () => {
      expect(effectiveSessionRole("proxy_admin_viewer")).toBe("Admin");
      expect(isViewOnlySessionRole("proxy_admin_viewer")).toBe(true);
    });
  });
});
