import { describe, it, expect } from "vitest";
import {
  canTeamMemberUpdateKey,
  isAdminRole,
  isProxyAdminRole,
  isUserTeamAdminForAnyTeam,
  isUserTeamAdminForSingleTeam,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
} from "./roles";
import { Member, Team } from "@/components/networking";

const makeTeam = (overrides: Partial<Team>): Team => ({
  team_id: "team-1",
  team_alias: "Test Team",
  models: [],
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "org-1",
  created_at: "2024-01-01",
  keys: [],
  members_with_roles: [],
  ...overrides,
});

const member = (user_id: string, role: string): Member => ({ user_id, user_email: `${user_id}@test.com`, role });

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

  describe("canTeamMemberUpdateKey", () => {
    it("returns true for a non-admin member when the team grants /key/update", () => {
      const team = makeTeam({
        members_with_roles: [member("user-1", "user")],
        team_member_permissions: ["/key/info", "/key/update"],
      });
      expect(canTeamMemberUpdateKey(team, "user-1")).toBe(true);
    });

    it("returns false for a non-admin member when /key/update is not granted", () => {
      const team = makeTeam({
        members_with_roles: [member("user-1", "user")],
        team_member_permissions: ["/key/info", "/key/health"],
      });
      expect(canTeamMemberUpdateKey(team, "user-1")).toBe(false);
    });

    it("returns false for a non-admin member when no permissions are set", () => {
      const team = makeTeam({
        members_with_roles: [member("user-1", "user")],
        team_member_permissions: undefined,
      });
      expect(canTeamMemberUpdateKey(team, "user-1")).toBe(false);
    });

    it("returns true for a team admin regardless of the permission list", () => {
      const team = makeTeam({
        members_with_roles: [member("user-1", "admin")],
        team_member_permissions: [],
      });
      expect(canTeamMemberUpdateKey(team, "user-1")).toBe(true);
    });

    it("returns false when the user is not a member of the team", () => {
      const team = makeTeam({
        members_with_roles: [member("user-2", "admin")],
        team_member_permissions: ["/key/update"],
      });
      expect(canTeamMemberUpdateKey(team, "user-1")).toBe(false);
    });

    it("returns false when the team is null or undefined", () => {
      expect(canTeamMemberUpdateKey(null, "user-1")).toBe(false);
      expect(canTeamMemberUpdateKey(undefined, "user-1")).toBe(false);
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
});
