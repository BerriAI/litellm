import { afterEach, describe, expect, it, vi } from "vitest";

import type { ModelGroupInfo, MCPServerData } from "@/components/PublicModelHubTableColumns";

import { curlSnippet, marketplaceCommand, mcpEndpoint, mcpSnippet, modelSnippet } from "./hubSnippets";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => "https://gateway.example.com",
  getGlobalLitellmHeaderName: () => "Authorization",
}));

const model = (overrides: Partial<ModelGroupInfo>): ModelGroupInfo => ({
  model_group: "anthropic/*",
  providers: ["anthropic"],
  supports_parallel_function_calling: false,
  supports_vision: false,
  supports_function_calling: false,
  ...overrides,
});

const server = { server_id: "s1", name: "github", server_name: "github" } as MCPServerData;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("model snippets", () => {
  it("points the SDK snippet at the gateway and a concrete model name", () => {
    vi.stubGlobal("window", { location: { origin: "http://ui.example.com" } });
    const snippet = modelSnippet(model({ mode: "chat" }));
    expect(snippet).toContain('base_url="https://gateway.example.com"');
    expect(snippet).toContain("anthropic/<model-name>");
    expect(snippet).not.toContain("ui.example.com");
  });

  it("offers cURL only for chat models", () => {
    expect(curlSnippet(model({ mode: "chat" }))).toContain("curl https://gateway.example.com/v1/chat/completions");
    expect(curlSnippet(model({ mode: undefined }))).not.toBeNull();
    expect(curlSnippet(model({ mode: "embedding" }))).toBeNull();
  });
});

describe("MCP and skill snippets", () => {
  it("builds the MCP endpoint from the gateway and the server name", () => {
    expect(mcpEndpoint(server)).toBe("https://gateway.example.com/github/mcp");
    expect(mcpSnippet(server)).toContain('"url": "https://gateway.example.com/github/mcp"');
  });

  it("routes a server without a server name through its listed name", () => {
    const aliased = { server_id: "s2", name: "my_alias", server_name: null } as unknown as MCPServerData;
    expect(mcpEndpoint(aliased)).toBe("https://gateway.example.com/my_alias/mcp");
  });

  it("adds the gateway's marketplace", () => {
    expect(marketplaceCommand()).toBe(
      "/plugin marketplace add https://gateway.example.com/claude-code/marketplace.json",
    );
  });
});
