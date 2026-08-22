import { describe, it, expect } from "vitest";
import {
  all_admin_roles,
  effectiveSessionRole,
  hasProxyWideSpendView,
  isAdminRole,
  spendScopeUserId,
  isOrgAdminForAnyOrg,
  isOrgAdminSessionRole,
  isProxyAdminRole,
  isUserTeamAdminForAnyTeam,
  isUserTeamAdminForSingleTeam,
  isViewOnlySessionRole,
  rolesAllowedToViewWriteScopedPages,
  rolesWithWriteAccess,
  teamListScopeUserId,
} from "./roles";
import { Organization, Team } from "@/components/networking";

const orgWithMembers = (members: { user_id: string; user_role: string }[]): Organization =>
  ({ organization_id: "org-1", members }) as unknown as Organization;

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

  describe("isOrgAdminForAnyOrg", () => {
    it("returns true when the user holds an org_admin membership in any organization", () => {
      const organizations = [
        orgWithMembers([{ user_id: "user-1", user_role: "internal_user" }]),
        orgWithMembers([{ user_id: "user-1", user_role: "org_admin" }]),
      ];
      expect(isOrgAdminForAnyOrg(organizations, "user-1")).toBe(true);
    });

    it("returns false when the user is only a plain member", () => {
      const organizations = [orgWithMembers([{ user_id: "user-1", user_role: "internal_user" }])];
      expect(isOrgAdminForAnyOrg(organizations, "user-1")).toBe(false);
    });

    it("does not credit one user with another user's org_admin membership", () => {
      const organizations = [orgWithMembers([{ user_id: "user-2", user_role: "org_admin" }])];
      expect(isOrgAdminForAnyOrg(organizations, "user-1")).toBe(false);
    });

    it("returns false for missing organizations, missing members, or a missing user id", () => {
      expect(isOrgAdminForAnyOrg(null, "user-1")).toBe(false);
      expect(isOrgAdminForAnyOrg(undefined, "user-1")).toBe(false);
      expect(isOrgAdminForAnyOrg([], "user-1")).toBe(false);
      expect(isOrgAdminForAnyOrg([{ organization_id: "org-1" } as unknown as Organization], "user-1")).toBe(false);
      expect(isOrgAdminForAnyOrg([orgWithMembers([{ user_id: "user-1", user_role: "org_admin" }])], null)).toBe(false);
      expect(isOrgAdminForAnyOrg([orgWithMembers([{ user_id: "user-1", user_role: "org_admin" }])], "")).toBe(false);
    });
  });

  describe("isOrgAdminSessionRole", () => {
    it("accepts both the raw and the formatted org admin role", () => {
      expect(isOrgAdminSessionRole("org_admin")).toBe(true);
      expect(isOrgAdminSessionRole(effectiveSessionRole("org_admin"))).toBe(true);
    });

    it("returns false for the role a membership-granted org admin actually carries", () => {
      expect(isOrgAdminSessionRole("Internal User")).toBe(false);
      expect(isOrgAdminSessionRole("internal_user")).toBe(false);
    });

    it("returns false for admin and missing roles", () => {
      expect(isOrgAdminSessionRole("Admin")).toBe(false);
      expect(isOrgAdminSessionRole("proxy_admin")).toBe(false);
      expect(isOrgAdminSessionRole(null)).toBe(false);
      expect(isOrgAdminSessionRole(undefined)).toBe(false);
      expect(isOrgAdminSessionRole("")).toBe(false);
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

  describe("teamListScopeUserId", () => {
    const SESSION_USER_ID = "user-1";

    it.each(["proxy_admin", "proxy_admin_viewer", "org_admin"])(
      "leaves %s unscoped so the endpoint keeps returning its broad list",
      (rawRole) => {
        expect(teamListScopeUserId(effectiveSessionRole(rawRole), SESSION_USER_ID)).toBeNull();
      },
    );

    it.each(["internal_user", "internal_user_viewer", "internal_viewer", "app_user"])(
      "scopes %s to its own user id, which is what the endpoint authorizes on",
      (rawRole) => {
        expect(teamListScopeUserId(effectiveSessionRole(rawRole), SESSION_USER_ID)).toBe(SESSION_USER_ID);
      },
    );

    it("also accepts the Admin Viewer label that formatUserRole emits", () => {
      expect(teamListScopeUserId("Admin Viewer", SESSION_USER_ID)).toBeNull();
    });

    it("scopes an unknown or absent role rather than assuming a broad list", () => {
      expect(teamListScopeUserId(null, SESSION_USER_ID)).toBe(SESSION_USER_ID);
      expect(teamListScopeUserId("Undefined Role", SESSION_USER_ID)).toBe(SESSION_USER_ID);
    });

    it("keeps Org Admin broad even though all_admin_roles carries only the raw org_admin", () => {
      expect(all_admin_roles).not.toContain(effectiveSessionRole("org_admin"));
      expect(isAdminRole(effectiveSessionRole("org_admin"))).toBe(false);
      expect(teamListScopeUserId(effectiveSessionRole("org_admin"), SESSION_USER_ID)).toBeNull();
    });
  });

  describe("spendScopeUserId", () => {
    const SESSION_USER_ID = "user-1234";

    it.each(["proxy_admin", "proxy_admin_viewer"])(
      "drops the user id for %s, whom the daily-activity endpoints let read every user's spend",
      (rawRole) => {
        expect(hasProxyWideSpendView(effectiveSessionRole(rawRole))).toBe(true);
        expect(spendScopeUserId(effectiveSessionRole(rawRole), SESSION_USER_ID)).toBeNull();
      },
    );

    it.each(["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer"])(
      "accepts %s in either the session-role or raw spelling, since all_admin_roles carries both",
      (role) => {
        expect(spendScopeUserId(role, SESSION_USER_ID)).toBeNull();
      },
    );

    it.each(["internal_user", "internal_user_viewer", "internal_viewer", "app_user"])(
      "scopes %s to its own user id, which is the only one the endpoint authorizes",
      (rawRole) => {
        expect(spendScopeUserId(effectiveSessionRole(rawRole), SESSION_USER_ID)).toBe(SESSION_USER_ID);
      },
    );

    it("scopes an unknown or absent role rather than asking for the whole proxy", () => {
      expect(spendScopeUserId(null, SESSION_USER_ID)).toBe(SESSION_USER_ID);
      expect(spendScopeUserId("Undefined Role", SESSION_USER_ID)).toBe(SESSION_USER_ID);
    });

    // The backend's user_api_key_has_admin_view covers PROXY_ADMIN and PROXY_ADMIN_VIEW_ONLY only,
    // so an org admin asking for the whole proxy is silently narrowed to its own rows and the
    // figures would read as the key's total. Both spellings have to scope, unlike teamListScopeUserId.
    it.each(["org_admin", "Org Admin"])("scopes %s, whom the backend does not grant an admin view", (role) => {
      expect(hasProxyWideSpendView(role)).toBe(false);
      expect(spendScopeUserId(role, SESSION_USER_ID)).toBe(SESSION_USER_ID);
    });

    it("differs from all_admin_roles by exactly org admin, which is the whole point of not reusing it", () => {
      const scopedByAllAdminRoles = all_admin_roles.filter((role) => spendScopeUserId(role, SESSION_USER_ID) !== null);

      expect(scopedByAllAdminRoles).toEqual(["org_admin"]);
    });
  });
});
