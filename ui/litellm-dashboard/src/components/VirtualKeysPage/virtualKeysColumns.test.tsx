import { describe, expect, it, vi } from "vitest";
import { getVirtualKeysColumns } from "./virtualKeysColumns";

describe("getVirtualKeysColumns", () => {
  const mockOptions = {
    setSelectedKey: vi.fn(),
    teams: [{ team_id: "team-1", team_alias: "Test Team" }],
    expandedAccordions: {} as Record<string, boolean>,
    setExpandedAccordions: vi.fn(),
  };

  it("should return an array of column definitions", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    expect(Array.isArray(columns)).toBe(true);
    expect(columns.length).toBeGreaterThan(0);
  });

  it("should include Key ID column with correct header", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    const keyIdColumn = columns.find((col) => col.id === "token");
    expect(keyIdColumn).toBeDefined();
    expect(keyIdColumn?.header).toBe("Key ID");
    expect(keyIdColumn?.enableSorting).toBe(true);
  });

  it("should include Key Alias column", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    const keyAliasColumn = columns.find((col) => col.id === "key_alias");
    expect(keyAliasColumn).toBeDefined();
    expect(keyAliasColumn?.header).toBe("Key Alias");
  });

  it("should include Secret Key column", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    const secretKeyColumn = columns.find((col) => col.id === "key_name");
    expect(secretKeyColumn).toBeDefined();
    expect(secretKeyColumn?.header).toBe("Secret Key");
  });

  it("should include Team Alias and Team ID columns", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    expect(columns.find((col) => col.id === "team_alias")).toBeDefined();
    expect(columns.find((col) => col.id === "team_id")).toBeDefined();
  });

  it("should include User Email column", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    const userEmailColumn = columns.find((col) => col.id === "user_email");
    expect(userEmailColumn).toBeDefined();
    expect(userEmailColumn?.header).toBe("User Email");
  });

  it("should include Spend (USD) and Budget (USD) columns", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    expect(columns.find((col) => col.id === "spend")?.header).toBe("Spend (USD)");
    expect(columns.find((col) => col.id === "max_budget")?.header).toBe("Budget (USD)");
  });

  it("should include Models and Rate Limits columns", () => {
    const columns = getVirtualKeysColumns(mockOptions);
    expect(columns.find((col) => col.id === "models")).toBeDefined();
    expect(columns.find((col) => col.id === "rate_limits")?.header).toBe("Rate Limits");
  });

  it("should work with null teams", () => {
    const columns = getVirtualKeysColumns({
      ...mockOptions,
      teams: null,
    });
    expect(columns.length).toBeGreaterThan(0);
    const teamAliasColumn = columns.find((col) => col.id === "team_alias");
    expect(teamAliasColumn).toBeDefined();
  });
});
