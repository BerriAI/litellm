import React from "react";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, it, expect, vi } from "vitest";
import { getProxyUISettings } from "@/components/networking";
import { copyToClipboard } from "@/utils/dataUtils";
import MCPConnect from "./mcp_connect";

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn().mockReturnValue("https://backend.example.com"),
  getGatewayBaseUrl: vi.fn().mockReturnValue("https://gateway.example.com"),
  getProxyUISettings: vi.fn(),
}));

vi.mock("@/utils/dataUtils", () => ({ copyToClipboard: vi.fn().mockResolvedValue(true) }));

const mockUiSettings = (docBaseUrl: string | null) =>
  vi.mocked(getProxyUISettings).mockResolvedValue({
    PROXY_BASE_URL: "https://backend.example.com",
    PROXY_LOGOUT_URL: "",
    LITELLM_UI_API_DOC_BASE_URL: docBaseUrl,
  });

const renderConnect = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MCPConnect accessToken="sk-test" />
    </QueryClientProvider>,
  );

const activePanel = () => screen.getByRole("tabpanel");

beforeEach(() => {
  mockUiSettings(null);
});

describe("MCPConnect with LITELLM_UI_API_DOC_BASE_URL", () => {
  it.each(["https://docs.example.com", " https://docs.example.com/ "])(
    "shows %j instead of the backend and gateway URLs on every tab",
    async (docBaseUrl) => {
      mockUiSettings(docBaseUrl);
      const user = userEvent.setup();
      renderConnect();

      expect(
        await within(activePanel()).findByText(/"server_url": "https:\/\/docs\.example\.com\/mcp"/),
      ).toBeInTheDocument();
      expect(within(activePanel()).getByText("https://docs.example.com/mcp")).toBeInTheDocument();

      await user.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));
      expect(within(activePanel()).getByText("https://docs.example.com/mcp")).toBeInTheDocument();
      expect(
        within(activePanel()).getByText(/curl --location 'https:\/\/docs\.example\.com\/v1\/responses'/),
      ).toBeInTheDocument();

      await user.click(screen.getByRole("tab", { name: "Cursor" }));
      expect(within(activePanel()).getByText(/"url": "https:\/\/docs\.example\.com\/mcp"/)).toBeInTheDocument();

      await user.click(screen.getByRole("tab", { name: "Streamable HTTP" }));
      expect(within(activePanel()).getByText("https://docs.example.com/mcp")).toBeInTheDocument();

      expect(document.body).not.toHaveTextContent("backend.example.com");
      expect(document.body).not.toHaveTextContent("gateway.example.com");
    },
  );
});

describe("MCPConnect routing", () => {
  it("uses the gateway for Responses and the backend for MCP server URLs", async () => {
    const user = userEvent.setup();
    renderConnect();

    expect(within(activePanel()).getByText(/"server_url": "https:\/\/backend\.example\.com\/mcp"/)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(within(activePanel()).getByText("https://backend.example.com/mcp")).toBeInTheDocument();
    expect(
      within(activePanel()).getByText(/curl --location 'https:\/\/gateway\.example\.com\/v1\/responses'/),
    ).toBeInTheDocument();
  });
});

describe("MCPConnect (tab mount contract)", () => {
  it.each([
    ["OpenAI API", '"server_url": "https://docs.example.com/mcp",'],
    ["LiteLLM Proxy", "curl --location 'https://docs.example.com/v1/responses' \\"],
    ["Cursor", '"url": "https://docs.example.com/mcp",'],
  ])("preserves the %s server filter when settings arrive and the snippet is copied", async (tab, expectedUrlLine) => {
    const settings = Promise.withResolvers<Awaited<ReturnType<typeof getProxyUISettings>>>();
    vi.mocked(getProxyUISettings).mockReturnValue(settings.promise);
    const user = userEvent.setup();
    renderConnect();

    await user.click(screen.getByRole("tab", { name: tab }));
    await user.click(within(activePanel()).getByRole("switch"));
    expect(within(activePanel()).getByRole("switch")).toBeChecked();

    await act(async () => {
      settings.resolve({
        PROXY_BASE_URL: "https://backend.example.com",
        PROXY_LOGOUT_URL: "",
        LITELLM_UI_API_DOC_BASE_URL: "https://docs.example.com",
      });
    });
    await waitFor(() => {
      expect(
        within(activePanel())
          .getByText(/"headers":/)
          .textContent?.split("\n")
          .map((line) => line.trim()),
      ).toContain(expectedUrlLine);
    });
    const snippet = within(activePanel()).getByText(/"headers":/);
    expect(within(activePanel()).getByRole("switch")).toBeChecked();
    expect(snippet).toHaveTextContent('"x-mcp-servers":');

    await user.click(within(activePanel()).getByRole("button", { name: "Copy code" }));
    expect(copyToClipboard).toHaveBeenLastCalledWith(snippet.textContent);
    expect(within(activePanel()).getByRole("switch")).toBeChecked();

    await user.click(screen.getByRole("tab", { name: "Streamable HTTP" }));
    await user.click(screen.getByRole("tab", { name: tab }));
    expect(within(activePanel()).getByRole("switch")).toBeChecked();
    expect(
      within(activePanel())
        .getByText(/"x-mcp-servers":/)
        .textContent?.split("\n")
        .map((line) => line.trim()),
    ).toContain(expectedUrlLine);
  });

  it("keeps the x-mcp-servers header toggle on after switching tabs away and back", async () => {
    renderConnect();

    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));
    expect(within(activePanel()).queryByText(/"x-mcp-servers":/)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("switch"));
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Cursor" }));
    await userEvent.click(screen.getByRole("tab", { name: "LiteLLM Proxy" }));

    expect(screen.getByRole("switch")).toBeChecked();
    expect(within(activePanel()).getByText(/"x-mcp-servers": "Zapier_MCP,dev-group"/)).toBeInTheDocument();
  });
});
