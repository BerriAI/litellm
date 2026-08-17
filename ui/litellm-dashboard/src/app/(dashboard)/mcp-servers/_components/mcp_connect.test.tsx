import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import MCPConnect from "./mcp_connect";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue("http://runtime-proxy.local:4000"),
  getProxyUISettings: vi.fn(),
}));

function renderConnect() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MCPConnect accessToken="test-token" />
    </QueryClientProvider>,
  );
}

describe("MCPConnect", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getProxyBaseUrl).mockReturnValue("http://runtime-proxy.local:4000");
  });

  it("prefers LITELLM_UI_API_DOC_BASE_URL over PROXY_BASE_URL for Server URL snippets", async () => {
    vi.mocked(networking.getProxyUISettings).mockResolvedValue({
      PROXY_BASE_URL: "https://proxy.example.com",
      PROXY_LOGOUT_URL: "",
      LITELLM_UI_API_DOC_BASE_URL: "https://docs.example.com",
    });

    renderConnect();

    await waitFor(() => {
      expect(screen.getAllByText("https://docs.example.com/mcp").length).toBeGreaterThan(0);
    });
    expect(screen.queryByText("https://proxy.example.com/mcp")).not.toBeInTheDocument();
    expect(screen.queryByText("http://runtime-proxy.local:4000/mcp")).not.toBeInTheDocument();
  });

  it("falls back to PROXY_BASE_URL when docs URL is unset", async () => {
    vi.mocked(networking.getProxyUISettings).mockResolvedValue({
      PROXY_BASE_URL: "https://proxy.example.com",
      PROXY_LOGOUT_URL: "",
      LITELLM_UI_API_DOC_BASE_URL: null,
    });

    renderConnect();

    await waitFor(() => {
      expect(screen.getAllByText("https://proxy.example.com/mcp").length).toBeGreaterThan(0);
    });
  });
});
