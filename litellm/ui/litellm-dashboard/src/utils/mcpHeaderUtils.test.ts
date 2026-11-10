import { describe, expect, it } from "vitest";
import { sanitizeMcpAliasForHeader } from "./mcpHeaderUtils";

describe("sanitizeMcpAliasForHeader", () => {
  it("lowercases and replaces spaces with underscores", () => {
    expect(sanitizeMcpAliasForHeader("My Server")).toBe("my_server");
  });

  it("replaces invalid characters for header token segments", () => {
    expect(sanitizeMcpAliasForHeader("GitHub-MCP!")).toBe("github_mcp");
  });

  it("preserves underscores and digits", () => {
    expect(sanitizeMcpAliasForHeader("github_mcp2")).toBe("github_mcp2");
  });
});
