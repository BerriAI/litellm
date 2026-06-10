import { describe, it, expect } from "vitest";
import {
  getFieldGroups,
  getMcpRequiredFieldDefs,
  SETTINGS_KEY,
  FieldGroup,
  RequiredFieldDef,
} from "./MCPStandardsSettings";
import type { TFunction } from "i18next";
import { MCPServer } from "./types";

const t = ((key: string) => key) as unknown as TFunction;

const makeServer = (overrides: Partial<MCPServer> = {}): MCPServer => ({
  server_id: "s1",
  created_at: "2024-01-01",
  created_by: "user",
  updated_at: "2024-01-01",
  updated_by: "user",
  ...overrides,
});

describe("getFieldGroups", () => {
  it("should contain four groups", () => {
    const groups = getFieldGroups(t);
    expect(groups).toHaveLength(4);
  });
});

describe("getMcpRequiredFieldDefs", () => {
  it("should flatten all fields from groups", () => {
    const groups = getFieldGroups(t);
    const defs = getMcpRequiredFieldDefs(t);
    const totalFields = groups.reduce((sum: number, g: FieldGroup) => sum + g.fields.length, 0);
    expect(defs).toHaveLength(totalFields);
  });
});

describe("field check functions", () => {
  const findCheck = (key: string) => getMcpRequiredFieldDefs(t).find((f: RequiredFieldDef) => f.key === key)!.check;

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
