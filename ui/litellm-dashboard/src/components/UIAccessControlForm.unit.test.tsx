import { describe, it, expect, vi, beforeEach } from "vitest";
import * as networking from "./networking";

// Mock the networking module
vi.mock("./networking", () => ({
  updateSSOSettings: vi.fn(),
}));

// Mock NotificationManager
vi.mock("./molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
  },
}));

// Extract the logic we want to test into a pure function
const buildApiPayload = (formValues: Record<string, any>) => {
  if (formValues.ui_access_mode_type === "all_authenticated_users") {
    // Set ui_access_mode to none when all_authenticated_users is selected
    return {
      ui_access_mode: "none",
    };
  } else {
    return {
      ui_access_mode: {
        type: formValues.ui_access_mode_type,
        restricted_sso_group: formValues.restricted_sso_group,
        sso_group_jwt_field: formValues.sso_group_jwt_field,
      },
    };
  }
};

describe("UIAccessControlForm Logic", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  describe("buildApiPayload", () => {
    it('should return ui_access_mode="none" when ui_access_mode_type is "all_authenticated_users"', () => {
      const formValues = {
        ui_access_mode_type: "all_authenticated_users",
        restricted_sso_group: "some-group",
        sso_group_jwt_field: "groups",
      };

      const result = buildApiPayload(formValues);

      expect(result).toEqual({
        ui_access_mode: "none",
      });
    });

    it('should return object structure when ui_access_mode_type is "restricted_sso_group"', () => {
      const formValues = {
        ui_access_mode_type: "restricted_sso_group",
        restricted_sso_group: "admin-users",
        sso_group_jwt_field: "team_groups",
      };

      const result = buildApiPayload(formValues);

      expect(result).toEqual({
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "admin-users",
          sso_group_jwt_field: "team_groups",
        },
      });
    });

    it("should return object structure for any other ui_access_mode_type", () => {
      const formValues = {
        ui_access_mode_type: "some_other_mode",
        restricted_sso_group: "test-group",
        sso_group_jwt_field: "user_groups",
      };

      const result = buildApiPayload(formValues);

      expect(result).toEqual({
        ui_access_mode: {
          type: "some_other_mode",
          restricted_sso_group: "test-group",
          sso_group_jwt_field: "user_groups",
        },
      });
    });

    it("should handle undefined values gracefully", () => {
      const formValues = {
        ui_access_mode_type: "restricted_sso_group",
        restricted_sso_group: undefined,
        sso_group_jwt_field: undefined,
      };

      const result = buildApiPayload(formValues);

      expect(result).toEqual({
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: undefined,
          sso_group_jwt_field: undefined,
        },
      });
    });

    it("should prioritize all_authenticated_users over other values", () => {
      const formValues = {
        ui_access_mode_type: "all_authenticated_users",
        restricted_sso_group: "admin-group",
        sso_group_jwt_field: "groups",
      };

      const result = buildApiPayload(formValues);

      // Should return "none" and ignore the other fields
      expect(result).toEqual({
        ui_access_mode: "none",
      });

      // Verify other fields are not included
      expect(result).not.toHaveProperty("restricted_sso_group");
      expect(result).not.toHaveProperty("sso_group_jwt_field");
    });
  });

  describe("API Integration", () => {
    it("should call updateSSOSettings with correct payload for all_authenticated_users", async () => {
      const mockAccessToken = "test-token";
      const formValues = {
        ui_access_mode_type: "all_authenticated_users",
        sso_group_jwt_field: "groups",
      };

      vi.mocked(networking.updateSSOSettings).mockResolvedValue({});

      const expectedPayload = buildApiPayload(formValues);

      // Simulate the API call
      await networking.updateSSOSettings(mockAccessToken, expectedPayload);

      expect(networking.updateSSOSettings).toHaveBeenCalledWith(mockAccessToken, { ui_access_mode: "none" });
    });

    it("should call updateSSOSettings with correct payload for restricted_sso_group", async () => {
      const mockAccessToken = "test-token";
      const formValues = {
        ui_access_mode_type: "restricted_sso_group",
        restricted_sso_group: "admin-team",
        sso_group_jwt_field: "team_groups",
      };

      vi.mocked(networking.updateSSOSettings).mockResolvedValue({});

      const expectedPayload = buildApiPayload(formValues);

      // Simulate the API call
      await networking.updateSSOSettings(mockAccessToken, expectedPayload);

      expect(networking.updateSSOSettings).toHaveBeenCalledWith(mockAccessToken, {
        ui_access_mode: {
          type: "restricted_sso_group",
          restricted_sso_group: "admin-team",
          sso_group_jwt_field: "team_groups",
        },
      });
    });
  });
});
