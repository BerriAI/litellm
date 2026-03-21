import { processSSOSettingsPayload } from "./utils";
import { describe, it, expect } from "vitest";

describe("processSSOSettingsPayload", () => {
  describe("without role mappings", () => {
    it("should return all fields except role mapping fields when use_role_mappings is false", () => {
      const formValues = {
        proxy_admin_teams: "team1, team2",
        admin_viewer_teams: "viewer1",
        internal_user_teams: "user1",
        internal_viewer_teams: "viewer1",
        default_role: "proxy_admin",
        group_claim: "groups",
        use_role_mappings: false,
        use_team_mappings: false,
        team_ids_jwt_field: "teams",
        other_field: "value",
        another_field: 123,
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        other_field: "value",
        another_field: 123,
      });
      expect(result.role_mappings).toBeUndefined();
      expect(result.team_mappings).toBeUndefined();
    });

    it("should return all fields except role mapping fields when use_role_mappings is not present", () => {
      const formValues = {
        proxy_admin_teams: "team1",
        admin_viewer_teams: "viewer1",
        internal_user_teams: "user1",
        internal_viewer_teams: "viewer1",
        default_role: "proxy_admin",
        group_claim: "groups",
        use_team_mappings: false,
        team_ids_jwt_field: "teams",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        other_field: "value",
      });
      expect(result.role_mappings).toBeUndefined();
      expect(result.team_mappings).toBeUndefined();
    });
  });

  describe("with role mappings enabled", () => {
    it("should create role mappings with all team types populated", () => {
      const formValues = {
        proxy_admin_teams: "admin1, admin2",
        admin_viewer_teams: "viewer1, viewer2, viewer3",
        internal_user_teams: "user1",
        internal_viewer_teams: "internal_viewer1, internal_viewer2",
        default_role: "proxy_admin",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.other_field).toBe("value");
      expect(result.role_mappings).toEqual({
        provider: "generic",
        group_claim: "groups",
        default_role: "proxy_admin",
        roles: {
          proxy_admin: ["admin1", "admin2"],
          proxy_admin_viewer: ["viewer1", "viewer2", "viewer3"],
          internal_user: ["user1"],
          internal_user_viewer: ["internal_viewer1", "internal_viewer2"],
        },
      });
    });

    it("should handle empty team strings", () => {
      const formValues = {
        proxy_admin_teams: "",
        admin_viewer_teams: "",
        internal_user_teams: "",
        internal_viewer_teams: "",
        default_role: "internal_user",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.roles).toEqual({
        proxy_admin: [],
        proxy_admin_viewer: [],
        internal_user: [],
        internal_user_viewer: [],
      });
    });

    it("should handle undefined team fields", () => {
      const formValues = {
        default_role: "internal_user_viewer",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.roles).toEqual({
        proxy_admin: [],
        proxy_admin_viewer: [],
        internal_user: [],
        internal_user_viewer: [],
      });
    });

    it("should handle whitespace-only team strings", () => {
      const formValues = {
        proxy_admin_teams: "   ",
        admin_viewer_teams: ", , ,",
        internal_user_teams: "user1, , user2",
        internal_viewer_teams: "viewer1,   ,viewer2",
        default_role: "proxy_admin_viewer",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.roles).toEqual({
        proxy_admin: [],
        proxy_admin_viewer: [],
        internal_user: ["user1", "user2"],
        internal_user_viewer: ["viewer1", "viewer2"],
      });
    });

    it("should trim whitespace from team names", () => {
      const formValues = {
        proxy_admin_teams: " admin1 , admin2 ",
        admin_viewer_teams: " viewer1 ",
        internal_user_teams: "  user1  ,  user2  ",
        internal_viewer_teams: "viewer1,viewer2",
        default_role: "internal_user",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.roles).toEqual({
        proxy_admin: ["admin1", "admin2"],
        proxy_admin_viewer: ["viewer1"],
        internal_user: ["user1", "user2"],
        internal_user_viewer: ["viewer1", "viewer2"],
      });
    });

    it("should filter out empty strings after trimming", () => {
      const formValues = {
        proxy_admin_teams: "admin1,,admin2, , admin3",
        default_role: "internal_user",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.roles.proxy_admin).toEqual(["admin1", "admin2", "admin3"]);
    });
  });

  describe("default role mapping", () => {
    it("should map internal_user_viewer correctly", () => {
      const formValues = {
        default_role: "internal_user_viewer",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("internal_user_viewer");
    });

    it("should map internal_user correctly", () => {
      const formValues = {
        default_role: "internal_user",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("internal_user");
    });

    it("should map proxy_admin_viewer correctly", () => {
      const formValues = {
        default_role: "proxy_admin_viewer",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("proxy_admin_viewer");
    });

    it("should map proxy_admin correctly", () => {
      const formValues = {
        default_role: "proxy_admin",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("proxy_admin");
    });

    it("should default to internal_user for unknown roles", () => {
      const formValues = {
        default_role: "unknown_role",
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("internal_user");
    });

    it("should default to internal_user for undefined default_role", () => {
      const formValues = {
        group_claim: "groups",
        use_role_mappings: true,
        sso_provider: "generic",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.role_mappings.default_role).toBe("internal_user");
    });
  });

  describe("without team mappings", () => {
    it("should return all fields except team mapping fields when use_team_mappings is false", () => {
      const formValues = {
        use_team_mappings: false,
        team_ids_jwt_field: "teams",
        sso_provider: "okta",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        sso_provider: "okta",
        other_field: "value",
      });
      expect(result.team_mappings).toBeUndefined();
    });

    it("should return all fields except team mapping fields when use_team_mappings is not present", () => {
      const formValues = {
        team_ids_jwt_field: "teams",
        sso_provider: "generic",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        sso_provider: "generic",
        other_field: "value",
      });
      expect(result.team_mappings).toBeUndefined();
    });

    it("should not include team mappings for unsupported providers even when use_team_mappings is true", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "teams",
        sso_provider: "google",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        sso_provider: "google",
        other_field: "value",
      });
      expect(result.team_mappings).toBeUndefined();
    });

    it("should not include team mappings for microsoft provider even when use_team_mappings is true", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "teams",
        sso_provider: "microsoft",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        sso_provider: "microsoft",
        other_field: "value",
      });
      expect(result.team_mappings).toBeUndefined();
    });
  });

  describe("with team mappings enabled", () => {
    it("should create team mappings for okta provider when use_team_mappings is true", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "teams",
        sso_provider: "okta",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.other_field).toBe("value");
      expect(result.team_mappings).toEqual({
        team_ids_jwt_field: "teams",
      });
    });

    it("should create team mappings for generic provider when use_team_mappings is true", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "custom_teams",
        sso_provider: "generic",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.other_field).toBe("value");
      expect(result.team_mappings).toEqual({
        team_ids_jwt_field: "custom_teams",
      });
    });

    it("should exclude team mapping fields from payload when team mappings are included", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "teams",
        sso_provider: "okta",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.use_team_mappings).toBeUndefined();
      expect(result.team_ids_jwt_field).toBeUndefined();
    });

    it("should handle team mappings and role mappings together", () => {
      const formValues = {
        use_team_mappings: true,
        team_ids_jwt_field: "teams",
        use_role_mappings: true,
        group_claim: "groups",
        default_role: "internal_user",
        sso_provider: "okta",
        other_field: "value",
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result.team_mappings).toEqual({
        team_ids_jwt_field: "teams",
      });
      expect(result.role_mappings).toBeDefined();
      expect(result.role_mappings.group_claim).toBe("groups");
    });
  });

  describe("edge cases", () => {
    it("should handle empty form values", () => {
      const result = processSSOSettingsPayload({});

      expect(result).toEqual({});
    });

    it("should preserve other fields in the payload", () => {
      const formValues = {
        use_role_mappings: false,
        use_team_mappings: false,
        sso_provider: "google",
        client_id: "123",
        client_secret: "secret",
        redirect_url: "http://example.com",
        custom_field: { nested: "value" },
        array_field: [1, 2, 3],
      };

      const result = processSSOSettingsPayload(formValues);

      expect(result).toEqual({
        sso_provider: "google",
        client_id: "123",
        client_secret: "secret",
        redirect_url: "http://example.com",
        custom_field: { nested: "value" },
        array_field: [1, 2, 3],
      });
    });
  });
});
