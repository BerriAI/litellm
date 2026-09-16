import { describe, expect, it } from "vitest";
import { parseConnectorConfig } from "./importConnectorConfig";

describe("parseConnectorConfig", () => {
  it("accepts a Claude Desktop mcpServers mapping", () => {
    const result = parseConnectorConfig(
      JSON.stringify({
        mcpServers: {
          github: { url: "https://api.example.com/mcp", authorization_token: "tok" },
          local: { command: "npx", args: ["-y", "@example/server"] },
        },
      }),
    );
    expect(result).toEqual({
      ok: true,
      payload: {
        mcpServers: {
          github: { url: "https://api.example.com/mcp", authorization_token: "tok" },
          local: { command: "npx", args: ["-y", "@example/server"] },
        },
      },
      connectorCount: 2,
    });
  });

  it("accepts an Anthropic Messages API mcp_servers array", () => {
    const result = parseConnectorConfig(
      JSON.stringify({
        mcp_servers: [{ type: "url", url: "https://mcp.example.com/sse", name: "deepwiki" }],
      }),
    );
    expect(result).toEqual({
      ok: true,
      payload: { mcp_servers: [{ type: "url", url: "https://mcp.example.com/sse", name: "deepwiki" }] },
      connectorCount: 1,
    });
  });

  it("rejects empty input", () => {
    const result = parseConnectorConfig("   ");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("Paste your connector JSON");
  });

  it("rejects malformed JSON", () => {
    const result = parseConnectorConfig("{ not json");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("Invalid JSON");
  });

  it("rejects objects without a recognized key", () => {
    const result = parseConnectorConfig(JSON.stringify({ servers: {} }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("mcpServers or mcp_servers");
  });

  it("rejects an empty mcpServers mapping", () => {
    const result = parseConnectorConfig(JSON.stringify({ mcpServers: {} }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("no connectors");
  });

  it("rejects a non-array mcp_servers", () => {
    const result = parseConnectorConfig(JSON.stringify({ mcp_servers: { a: 1 } }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("must be an array");
  });

  it("rejects an array mcpServers", () => {
    const result = parseConnectorConfig(JSON.stringify({ mcpServers: [] }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("must be an object");
  });
});
