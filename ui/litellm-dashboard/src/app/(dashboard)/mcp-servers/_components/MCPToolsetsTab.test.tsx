import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MCPToolsetsTab } from "./MCPToolsetsTab";
import useProxySettings from "@/app/(dashboard)/hooks/proxySettings/useProxySettings";

const DOC_BASE_URL = "https://gateway.public.example.com";
const PROXY_BASE_URL = "http://proxy.internal:4000";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: () => PROXY_BASE_URL,
  createMCPToolset: vi.fn(),
  updateMCPToolset: vi.fn(),
  deleteMCPToolset: vi.fn(),
  listMCPTools: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

vi.mock("@/app/(dashboard)/hooks/proxySettings/useProxySettings", () => ({
  default: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: () => ({
    data: [
      {
        toolset_id: "ts-1",
        toolset_name: "github-tools",
        description: "GitHub helpers",
        tools: [{ server_id: "srv-1", tool_name: "create_issue" }],
        created_at: "2026-01-01T00:00:00Z",
      },
    ],
    isLoading: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", alias: "github", server_name: "github" }] }),
}));

const mockedUseProxySettings = vi.mocked(useProxySettings);

function renderTab(docBaseUrl: string | null) {
  mockedUseProxySettings.mockReturnValue({
    PROXY_BASE_URL,
    PROXY_LOGOUT_URL: "",
    LITELLM_UI_API_DOC_BASE_URL: docBaseUrl,
  });
  render(
    <QueryClientProvider client={new QueryClient()}>
      <MCPToolsetsTab accessToken="sk-test" userRole="Admin" />
    </QueryClientProvider>,
  );
}

describe("MCPToolsetsTab", () => {
  it("builds the usage guide and row endpoint urls from LITELLM_UI_API_DOC_BASE_URL when set", () => {
    renderTab(DOC_BASE_URL);

    expect(screen.getByText(new RegExp(`${DOC_BASE_URL}/toolset/<toolset-name>/mcp`))).toBeInTheDocument();
    expect(screen.getByText(`${DOC_BASE_URL}/toolset/github-tools/mcp`)).toBeInTheDocument();
    expect(screen.queryAllByText(/proxy\.internal/)).toHaveLength(0);
  });

  it("falls back to the proxy base url when LITELLM_UI_API_DOC_BASE_URL is unset", () => {
    renderTab(null);

    expect(screen.getByText(new RegExp(`${PROXY_BASE_URL}/toolset/<toolset-name>/mcp`))).toBeInTheDocument();
    expect(screen.getByText(`${PROXY_BASE_URL}/toolset/github-tools/mcp`)).toBeInTheDocument();
  });
});
