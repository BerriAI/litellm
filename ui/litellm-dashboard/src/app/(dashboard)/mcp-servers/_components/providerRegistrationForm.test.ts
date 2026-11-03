import { describe, expect, it } from "vitest";
import type { MCPServer } from "@/components/mcp_tools/types";
import {
  buildProviderRegistrationPayload,
  providerRegistrationDefaults,
  providerRegistrationSchema,
} from "./providerRegistrationForm";

const values = {
  ...providerRegistrationDefaults(),
  server_name: "docs_native",
  url: "https://docs.example.com/mcp",
  tools_text: " search_docs \nfetch_doc\nsearch_docs\n\n",
};

const server: MCPServer = {
  server_id: "provider-docs",
  execution_mode: "provider",
  server_name: "docs_native",
  url: values.url,
  allowed_tools: ["search_docs", "fetch_doc"],
  mcp_info: { tool_allowlist_enforced: true, logo_url: "https://example.com/icon.svg" },
  tool_name_to_description: { search_docs: "Search documentation", fetch_doc: "Fetch a document" },
  tool_name_to_display_name: { search_docs: "Search", fetch_doc: "Fetch" },
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
  created_by: "admin",
  updated_by: "admin",
};

describe("provider registration payload", () => {
  it("sends finite deduplicated tools and only provider registration fields", () => {
    expect(buildProviderRegistrationPayload(providerRegistrationSchema.parse(values))).toEqual({
      execution_mode: "provider",
      server_name: "docs_native",
      alias: null,
      description: null,
      url: values.url,
      allowed_tools: ["search_docs", "fetch_doc"],
      mcp_access_groups: [],
      allow_all_keys: false,
      mcp_info: { tool_allowlist_enforced: true },
    });
  });

  it("clears tools with an enforced empty list and removes stale tool metadata", () => {
    const edited = { ...providerRegistrationDefaults(server), tools_text: " \n " };
    const payload = buildProviderRegistrationPayload(edited, server);
    expect(payload).not.toHaveProperty("execution_mode");
    expect(payload).toMatchObject({
      server_id: server.server_id,
      allowed_tools: [],
      mcp_info: server.mcp_info,
      tool_name_to_description: {},
      tool_name_to_display_name: {},
    });
  });

  it("preserves descriptions for retained tools and hydrates stored permissions", () => {
    const stored = { ...server, mcp_access_groups: ["engineering"], allow_all_keys: true };
    const defaults = providerRegistrationDefaults(stored);
    expect(defaults).toMatchObject({
      tools_text: "search_docs\nfetch_doc",
      mcp_access_groups: ["engineering"],
      allow_all_keys: true,
    });
    expect(buildProviderRegistrationPayload({ ...defaults, tools_text: "search_docs" }, stored)).toMatchObject({
      allowed_tools: ["search_docs"],
      tool_name_to_description: { search_docs: "Search documentation" },
      tool_name_to_display_name: { search_docs: "Search" },
    });
  });

  it.each([
    "http://example.com/mcp",
    "https:example.com/mcp",
    "https://user:password@example.com/mcp",
    "https://@example.com/mcp",
    "https://example.com/mcp#tools",
    "https://example.com/with space",
    "https://example.com/with\u0001control",
    "https://example.com\\mcp",
  ])("rejects unsupported endpoint %s", (url) => {
    expect(providerRegistrationSchema.safeParse({ ...values, url }).success).toBe(false);
  });

  it.each(["*", "search?", "search[docs]"])("rejects wildcard tool %s", (tools_text) => {
    expect(providerRegistrationSchema.safeParse({ ...values, tools_text }).success).toBe(false);
  });
});
