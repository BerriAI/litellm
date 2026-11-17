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

  it("offers cURL for chat models but not embeddings", () => {
    expect(curlSnippet(model({ mode: "chat" }))).toContain("curl https://gateway.example.com/v1/chat/completions");
    expect(curlSnippet(model({ mode: undefined }))).not.toBeNull();
    expect(curlSnippet(model({ mode: "embedding" }))).toBeNull();
  });

  it.each([null, undefined, [], ["POST"], ["GET", "POST"]])(
    "uses the pass-through route with a JSON body when its methods are %j",
    (methods) => {
      const endpoint = model({
        mode: "passthrough",
        pass_through_path: "/clinical-ner",
        pass_through_methods: methods,
      });
      const expected = [
        'curl -X "POST" "https://gateway.example.com/clinical-ner" \\',
        '  -H "Authorization: Bearer your_api_key" \\',
        '  -H "Content-Type: application/json" \\',
        `  -d '{"input": "Hello, how are you?"}'`,
      ].join("\n");
      expect(curlSnippet(endpoint)).toBe(expected);
      expect(modelSnippet(endpoint)).toBe(expected);
    },
  );

  it.each([["GET"], ["PUT", "GET"]])("uses the first non-POST method with no body for %j", (...methods) => {
    const endpoint = model({
      mode: "passthrough",
      pass_through_path: "/lpr-catalog",
      pass_through_methods: methods,
    });
    const expected = [
      `curl -X "${methods[0]}" "https://gateway.example.com/lpr-catalog" \\`,
      '  -H "Authorization: Bearer your_api_key"',
    ].join("\n");
    expect(curlSnippet(endpoint)).toBe(expected);
    expect(modelSnippet(endpoint)).toBe(expected);
  });

  it("does not offer a chat SDK example for a pass-through row without its route", () => {
    const endpoint = model({ mode: "passthrough" });
    expect(curlSnippet(endpoint)).toBeNull();
    expect(modelSnippet(endpoint)).toBe("");
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
