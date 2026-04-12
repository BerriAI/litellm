import { describe, it, expect } from "vitest";
import { FIELD_GROUPS, MCP_REQUIRED_FIELD_DEFS, SETTINGS_KEY } from "./MCPStandardsSettings";
import { MCPServer } from "./types";

const makeServer = (overrides: Partial<MCPServer> = {}): MCPServer => ({
  server_id: "s1",
  created_at: "2024-01-01",
  created_by: "user",
  updated_at: "2024-01-01",
  updated_by: "user",
  ...overrides,
});

describe("FIELD_GROUPS", () => {
  it("should contain four groups", () => {
    expect(FIELD_GROUPS).toHaveLength(4);
    expect(FIELD_GROUPS.map((g) => g.label)).toEqual([
      "Documentation",
      "Source",
      "Connection",
      "Security",
    ]);
  });
});

describe("MCP_REQUIRED_FIELD_DEFS", () => {
  it("should flatten all fields from groups", () => {
    const totalFields = FIELD_GROUPS.reduce((sum, g) => sum + g.fields.length, 0);
    expect(MCP_REQUIRED_FIELD_DEFS).toHaveLength(totalFields);
  });
});

describe("field check functions", () => {
  const findCheck = (key: string) =>
    MCP_REQUIRED_FIELD_DEFS.find((f) => f.key === key)!.check;

  it("should pass description check when description is present", () => {
    expect(findCheck("description")(makeServer({ description: "A service" }))).toBe(true);
  });

  it("should fail description check when description is empty", () => {
    expect(findCheck("description")(makeServer({ description: "  " }))).toBe(false);
  });

  it("should pass auth check when auth_type is not none", () => {
    expect(findCheck("auth_type")(makeServer({ auth_type: "oauth2" }))).toBe(true);
  });

  it("should fail auth check when auth_type is none", () => {
    expect(findCheck("auth_type")(makeServer({ auth_type: "none" }))).toBe(false);
  });

  it("should fail auth check when auth_type is missing", () => {
    expect(findCheck("auth_type")(makeServer())).toBe(false);
  });
});

describe("SETTINGS_KEY", () => {
  it("should equal mcp_required_fields", () => {
    expect(SETTINGS_KEY).toBe("mcp_required_fields");
  });
});
