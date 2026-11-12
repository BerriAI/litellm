import { describe, expect, it } from "vitest";
import { simplifyKeyGenerateError } from "./utils";

describe("simplifyKeyGenerateError", () => {
  const expectedSimplifiedMessage =
    "Team member does not have permission to generate key for this team. Ask your proxy admin to configure the team member permission settings.";

  describe("when error is NOT related to /key/generate", () => {
    it("should return the original error message for generic errors", () => {
      const error = "Some random error";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: Some random error");
    });

    it("should return the original error message for other endpoint errors", () => {
      const error = "Error: Failed to fetch /key/list";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: Error: Failed to fetch /key/list");
    });

    it("should handle Error objects for non-/key/generate errors", () => {
      const error = new Error("Database connection failed");
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: Error: Database connection failed");
    });
  });

  describe("when error is related to /key/generate team member permission error", () => {
    it("should simplify JSON error with team_member_permission_error", () => {
      const error = JSON.stringify({
        error: {
          message:
            "Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE. You only have access to the following endpoints: ['/key/info', '/key/health'] for team 60f6c0db-3ed9-4112-a2ef-8d838b2c8679. To create keys for this team, please ask your proxy admin to check the team member permission settings and update the settings to allow team member users to create keys.",
          type: "team_member_permission_error",
          param: "/key/generate",
          code: "401",
        },
      });
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should simplify error string containing /key/generate and team_member_permission_error", () => {
      const error =
        'Error creating the key: {"error":{"message":"Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE. You only have access to the following endpoints: [\'/key/info\', \'/key/health\'] for team 60f6c0db-3ed9-4112-a2ef-8d838b2c8679. To create keys for this team, please ask your proxy admin to check the team member permission settings and update the settings to allow team member users to create keys.","type":"team_member_permission_error","param":"/key/generate","code":"401"}}';
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should simplify error with KeyManagementRoutes.KEY_GENERATE in message", () => {
      const error =
        "Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE. Ask your proxy admin to check the team member permission settings.";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should handle nested error object structure", () => {
      const error = {
        error: {
          message:
            "Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE. You only have access to the following endpoints: ['/key/info', '/key/health'].",
          type: "team_member_permission_error",
        },
      };
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should handle error with team_member_permission_error type in string", () => {
      const error =
        'Some prefix {"error":{"type":"team_member_permission_error","message":"Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE"}} some suffix';
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });
  });

  describe("when error is related to /key/generate but NOT a permission error", () => {
    it("should return original error message for /key/generate validation errors", () => {
      const error = "Invalid request to /key/generate: missing required field";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: Invalid request to /key/generate: missing required field");
    });

    it("should return original error message for /key/generate server errors", () => {
      const error = JSON.stringify({
        error: {
          message: "Internal server error occurred while processing /key/generate",
          type: "server_error",
          code: "500",
        },
      });
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(`Error creating the key: ${error}`);
    });
  });

  describe("edge cases", () => {
    it("should handle malformed JSON gracefully", () => {
      const error =
        '{"error":{"message":"Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE", invalid json';
      const result = simplifyKeyGenerateError(error);
      // Should still detect the permission error from the string content
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should handle empty string", () => {
      const error = "";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: ");
    });

    it("should handle null", () => {
      const error = null;
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: null");
    });

    it("should handle undefined", () => {
      const error = undefined;
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe("Error creating the key: undefined");
    });

    it("should handle Error object with /key/generate permission error", () => {
      const error = new Error(
        '{"error":{"message":"Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE","type":"team_member_permission_error"}}',
      );
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should handle error with multiple JSON objects", () => {
      const error =
        'Prefix {"error":{"message":"Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE","type":"team_member_permission_error"}} suffix {"other":"data"}';
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });

    it("should handle error message without explicit team_member_permission_error type but with permission text", () => {
      const error =
        "Team member does not have permissions for endpoint: KeyManagementRoutes.KEY_GENERATE. Please contact admin.";
      const result = simplifyKeyGenerateError(error);
      expect(result).toBe(expectedSimplifiedMessage);
    });
  });
});
