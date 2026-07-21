import React from "react";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import MCPAppsPanel from "./MCPAppsPanel";
import * as networking from "../networking";

vi.mock("../networking", () => ({
  fetchMCPServers: vi.fn(),
  getMCPOAuthUserCredentialStatus: vi.fn(),
  deleteMCPOAuthUserCredential: vi.fn(),
  listMCPTools: vi.fn(),
  getProxyBaseUrl: vi.fn().mockReturnValue("http://localhost:4000"),
}));

const idJagServer = {
  server_id: "srv-ema",
  server_name: "ema_upstream",
  alias: "ema_upstream",
  url: "https://up.example.com/mcp",
  transport: "http",
  auth_type: "oauth2_id_jag",
  created_at: "2024-01-01T00:00:00Z",
  created_by: "u",
  updated_at: "2024-01-01T00:00:00Z",
  updated_by: "u",
};

const plainServer = {
  ...idJagServer,
  server_id: "srv-plain",
  server_name: "plain_upstream",
  alias: "plain_upstream",
  auth_type: "none",
};

const renderPanel = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MCPAppsPanel accessToken="sk-test" selectedServers={[]} onChange={() => {}} />
    </QueryClientProvider>,
  );
};

describe("MCPAppsPanel enterprise-managed authorization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([idJagServer, plainServer] as never);
    vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: [] } as never);
  });

  it("renders an id_jag server as already connected and never fetches its credential status", async () => {
    const { container } = renderPanel();

    await waitFor(() => {
      expect(screen.getByText("ema_upstream")).toBeInTheDocument();
    });

    const tile = screen.getByText("ema_upstream").closest("div[class*='cursor-pointer']") ?? container;
    expect(tile.querySelector("svg.text-emerald-600")).toBeTruthy();
    expect(networking.getMCPOAuthUserCredentialStatus).not.toHaveBeenCalled();
  });

  it("shows the connected badge and no connect affordance in the id_jag detail pane", async () => {
    renderPanel();

    await waitFor(() => {
      expect(screen.getByText("ema_upstream")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("ema_upstream"));

    await waitFor(() => {
      expect(screen.getByText("Connected via your organization sign-in")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^Connect$/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Disconnect$/ })).not.toBeInTheDocument();
  });

  it("keeps the plain Connect toggle for a non id_jag server", async () => {
    renderPanel();

    await waitFor(() => {
      expect(screen.getByText("plain_upstream")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("plain_upstream"));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^Connect$/ })).toBeInTheDocument();
    });
    expect(screen.queryByText("Connected via your organization sign-in")).not.toBeInTheDocument();
  });
});
