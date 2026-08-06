import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeInteractionsRequest } from "./interactions_api";
import type { MCPServer } from "@/components/mcp_tools/types";

vi.mock("@/components/networking", () => ({
  getGlobalLitellmHeaderName: vi.fn(() => "Authorization"),
  getProxyBaseUrl: vi.fn(() => "https://example.com"),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

describe("makeInteractionsRequest", () => {
  const mockUpdateUI = vi.fn();
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      body: {
        getReader: () => ({
          read: vi.fn().mockResolvedValue({ done: true, value: undefined }),
        }),
      },
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("forwards MCP tool blocks in the interactions request body", async () => {
    const selectedMCPServers = ["server-1"];
    const mcpServers = [
      {
        server_id: "server-1",
        server_name: "deepwiki",
        alias: "wiki",
        url: "http://example.com",
        created_at: "2024-01-01",
        created_by: "test",
        updated_at: "2024-01-01",
        updated_by: "test",
      },
    ] as MCPServer[];

    await makeInteractionsRequest(
      "list tools",
      mockUpdateUI,
      "gpt-4o",
      "test-token",
      undefined,
      undefined,
      undefined,
      undefined,
      selectedMCPServers,
      mcpServers,
    );

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body)) as {
      model: string;
      input: string;
      tools: Array<Record<string, unknown>>;
    };

    expect(body.model).toBe("gpt-4o");
    expect(body.input).toBe("list tools");
    expect(body.tools).toEqual([
      {
        type: "mcp",
        server_label: "deepwiki",
        server_url: "litellm_proxy/mcp/deepwiki",
        require_approval: "never",
      },
    ]);
  });

  it("omits tools when no MCP servers are selected", async () => {
    await makeInteractionsRequest("hello", mockUpdateUI, "gpt-4o", "test-token");

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(body.tools).toBeUndefined();
  });
});
